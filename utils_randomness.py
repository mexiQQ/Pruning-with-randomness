import torch
import argparse
import logging
import math
import os
import random
import torch

import datasets
from datasets import load_metric

import transformers
from accelerate import Accelerator
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
    set_seed,
)

from transformers.utils.versions import require_version
from torch.nn import MSELoss
from torch.utils.data import (
    DataLoader, 
    TensorDataset
)

import time
from glue2 import *
from data_processing import *
from datetime import datetime,timedelta

def train_one_phase(args, model, teacher, train_dataloader, accelerator, optimizer, lr_scheduler, eval_dataloader, eval_dataset, tokenizer, pruner, epochs=1, logger=None, model_soup_id=-1):

    completed_steps = 0
    tr_att_loss = 0
    tr_rep_loss = 0
    tr_cls_loss = 0
    tr_adv_loss = 0
    tr_loss = 0
    loss_mse = MSELoss()

    es = None
    early_stop_trigger = False
    if args.early_stop:
        es = EarlyStopping(patience=100, mode='max')
    
    for epoch in range(epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.cuda() for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            outputs = model(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                labels=label_ids,
                output_hidden_states=True, 
                output_attentions=True)

            student_loss = outputs.loss
            student_logits = outputs.logits
            student_hidden_states = outputs.hidden_states
            student_attentions = outputs.attentions

            if teacher:
                with torch.no_grad():
                    teacher_outputs = teacher(
                        input_ids=input_ids,
                        attention_mask=input_mask,
                        token_type_ids=segment_ids,
                        labels=label_ids,
                        output_hidden_states=True, 
                        output_attentions=True)
                    teacher_logits = teacher_outputs.logits
                    teacher_hidden_states = teacher_outputs.hidden_states
                    teacher_attentions = teacher_outputs.attentions

            att_loss = torch.zeros(1).cuda()
            rep_loss = torch.zeros(1).cuda()
            cls_loss = torch.zeros(1).cuda()
            adv_loss = torch.zeros(1).cuda()
            loss = torch.zeros(1).cuda()
            
            if args.kd:
                for student_att, teacher_att in zip(student_attentions, teacher_attentions):
                    tmp_loss = loss_mse(student_att, teacher_att)
                    att_loss += tmp_loss

                for student_rep, teacher_rep in zip(student_hidden_states, teacher_hidden_states):
                    tmp_loss = loss_mse(student_rep, teacher_rep)
                    rep_loss += tmp_loss                        

                if args.task_name == "stsb":
                    cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))
                else:
                    cls_loss = soft_cross_entropy(
                    student_logits / args.temperature, teacher_logits / args.temperature)
            
                loss = rep_loss + att_loss + cls_loss
            else:
                cls_loss = student_loss 
                loss += cls_loss
                assert loss, "The switch of loss computation is closed because of kd"                

            loss = loss / args.gradient_accumulation_steps

            tr_att_loss += att_loss.item()
            tr_rep_loss += rep_loss.item()
            tr_cls_loss += cls_loss.item()
            tr_adv_loss += adv_loss.item()
            tr_loss += loss.item()

            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                completed_steps += 1

                if pruner:
                    pruner.prune()

            if completed_steps % args.print_step == 0:
                if accelerator.is_main_process:
                    logger.info("{:0>6d}/{:0>6d},lr: {:.6f}, loss: {:.6f}, adv_loss: {:.6f}, att_loss: {:.6f}, rep_loss: {:.6f}, cls_loss: {:.6f}, avg_loss: {:.6f}, avg_adv_loss: {:.6f}, avg_att_loss: {:.6f}, avg_rep_loss: {:.6f}, avg_cls_loss: {:.6f}".format(
                        completed_steps,
                        args.max_train_steps, 
                        get_lr(optimizer),
                        loss.item(),
                        adv_loss.item(),
                        att_loss.item(),
                        rep_loss.item(),
                        cls_loss.item(),
                        tr_loss / completed_steps,
                        tr_adv_loss / completed_steps,
                        tr_att_loss / completed_steps,
                        tr_rep_loss / completed_steps,
                        tr_cls_loss / completed_steps,
                        )
                    )

            if completed_steps % args.eval_step == 0:
                if pruner and accelerator.is_main_process:
                    layer_sparse_rate, total_sparse_rate = pruner.prune_sparsity()
                    logger.info('\nepoch %d; step=%d; weight sparsity=%s' % (epoch, completed_steps, total_sparse_rate))

            if completed_steps % args.eval_step == 0: 
                eval_metric = evaluate_data(
                    eval_dataloader,
                    eval_dataset,
                    model,
                    args,
                    logger,
                    mode="dev"
                )
                logger.info(f"epoch {epoch}, step {completed_steps}/{args.max_train_steps}, patience:{es.num_bad_epochs}/{es.patience}, Best: {es.best} : {eval_metric}")

                if args.early_stop:
                    assert args.early_stop_metric in eval_metric, "Early stop metric is not in evaluation result"
                    if es.step(eval_metric[args.early_stop_metric]):
                        early_stop_trigger = True
                        logger.info("****\nEarly Stop is Triggered\n****")
                        break
                    else:
                        if es.best == eval_metric[args.early_stop_metric]:
                            es.record = eval_metric
                            if args.output_dir is not None:
                                accelerator.wait_for_everyone()
                                logger.info(f"**** Save model with best result: {es.record} ****")

                                if model_soup_id == -1:
                                    dir_path = args.output_dir
                                else: 
                                    dir_path = f"{args.output_dir}/soup_{model_soup_id}"

                                if accelerator.is_main_process:
                                    if args.output_dir is not None:
                                        os.makedirs(dir_path, exist_ok=True)
                                unwrapped_model = accelerator.unwrap_model(model)
                                unwrapped_model.save_pretrained(dir_path, save_function=accelerator.save)
                                torch.save(optimizer.state_dict(), f"{dir_path}/optimizer.pth")
                                if accelerator.is_main_process:
                                    tokenizer.save_pretrained(dir_path) 
            model.train()
 
        if early_stop_trigger:
           break
    return es.best
            

def quickcheck(args, model, teacher, train_dataloader, accelerator, optimizer, lr_scheduler, eval_dataloader, eval_dataset, pruner, logger, model_soup_id, identity):

    if model_soup_id == -1:
        dir_path = args.output_dir
    else: 
        dir_path = f"{args.output_dir}/soup_{model_soup_id}"

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(dir_path, exist_ok=True)

    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(f'{dir_path}/logfile_{identity}.log')
    formatter    = logging.Formatter('%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    logger.handlers = []
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    completed_steps = 0
    tr_att_loss = 0
    tr_rep_loss = 0
    tr_cls_loss = 0
    tr_adv_loss = 0
    tr_loss = 0
    loss_mse = MSELoss()
    metrics = []

    model.train()
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        outputs = model(
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            labels=label_ids,
            output_hidden_states=True, 
            output_attentions=True)

        student_loss = outputs.loss
        student_logits = outputs.logits
        student_hidden_states = outputs.hidden_states
        student_attentions = outputs.attentions

        if teacher:
            with torch.no_grad():
                teacher_outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=input_mask,
                    token_type_ids=segment_ids,
                    labels=label_ids,
                    output_hidden_states=True, 
                    output_attentions=True)
                teacher_logits = teacher_outputs.logits
                teacher_hidden_states = teacher_outputs.hidden_states
                teacher_attentions = teacher_outputs.attentions

        att_loss = torch.zeros(1).cuda()
        rep_loss = torch.zeros(1).cuda()
        cls_loss = torch.zeros(1).cuda()
        adv_loss = torch.zeros(1).cuda()
        loss = torch.zeros(1).cuda()
        
        if args.kd:
            for student_att, teacher_att in zip(student_attentions, teacher_attentions):
                tmp_loss = loss_mse(student_att, teacher_att)
                att_loss += tmp_loss

            for student_rep, teacher_rep in zip(student_hidden_states, teacher_hidden_states):
                tmp_loss = loss_mse(student_rep, teacher_rep)
                rep_loss += tmp_loss                        

            if args.task_name == "stsb":
                cls_loss = loss_mse(student_logits.view(-1), label_ids.view(-1))
            else:
                cls_loss = soft_cross_entropy(
                student_logits / args.temperature, teacher_logits / args.temperature)
        
            loss = rep_loss + att_loss + cls_loss
        else:
            cls_loss = student_loss 
            loss += cls_loss
            assert loss, "The switch of loss computation is closed because of kd"                

        loss = loss / args.gradient_accumulation_steps

        tr_att_loss += att_loss.item()
        tr_rep_loss += rep_loss.item()
        tr_cls_loss += cls_loss.item()
        tr_adv_loss += adv_loss.item()
        tr_loss += loss.item()

        accelerator.backward(loss)
        if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1

            if pruner:
                pruner.prune()

        if completed_steps % args.print_step == 0:
            if accelerator.is_main_process:
                logger.info("{:0>6d}/{:0>6d},lr: {:.6f}, loss: {:.6f}, adv_loss: {:.6f}, att_loss: {:.6f}, rep_loss: {:.6f}, cls_loss: {:.6f}, avg_loss: {:.6f}, avg_adv_loss: {:.6f}, avg_att_loss: {:.6f}, avg_rep_loss: {:.6f}, avg_cls_loss: {:.6f}".format(
                    completed_steps,
                    args.max_train_steps, 
                    get_lr(optimizer),
                    loss.item(),
                    adv_loss.item(),
                    att_loss.item(),
                    rep_loss.item(),
                    cls_loss.item(),
                    tr_loss / completed_steps,
                    tr_adv_loss / completed_steps,
                    tr_att_loss / completed_steps,
                    tr_rep_loss / completed_steps,
                    tr_cls_loss / completed_steps,
                    )
                )

        if completed_steps % args.eval_step == 0: 
            eval_metric = evaluate_data(
                eval_dataloader,
                eval_dataset,
                model,
                args,
                logger,
                mode="dev"
            )
            logger.info(f"step {completed_steps}/{args.max_train_steps}: {eval_metric}")
            metrics.append(eval_metric[args.early_stop_metric])

        model.train()

    torch.save(pruner._mask, f"{dir_path}/mask.pt")
    metric = sum(metrics[-10:])/10 

    logger.info("#" * 30)
    logger.info(f"Current Mask Metric: {metric}")
    logger.info("#" * 30 + "\n\n\n")
    return metric
    
def get_tensor_data(output_mode, features):
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    tensor_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return tensor_data, all_label_ids


def evaluate_data(dataloader, dataset, model, args, logger, mode="train"):
    model.eval()
    logger.info(f"***** Running {mode} evaluation *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_eval_batch_size}")

    metric = None
    if args.task_name is not None:
        #metric = load_metric("glue", args.task_name, keep_in_memory=True)
        metric = Glue(args.task_name)
    else:
        metric = load_metric("accuracy", keep_in_memory=True)

    for step, batch in enumerate(dataloader):
        batch = tuple(t.cuda() for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch
        outputs = model( 
            input_ids=input_ids,
            attention_mask=input_mask,
            token_type_ids=segment_ids,
            labels=label_ids)
        predictions = outputs.logits.argmax(dim=-1) if not args.is_regression else outputs.logits.squeeze()
        metric.add_batch(
            predictions=predictions,
            references=label_ids,
        )
    eval_metric = metric.compute()
    return eval_metric


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self.record = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print('terminating because of early stopping!')
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']