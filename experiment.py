# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Finetuning a 🤗 Transformers model for sequence classification on GLUE."""

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
from one_shot_pruner import Prune
from torch.utils.data import (
    DataLoader, 
    TensorDataset
)

import time
from data_processing import *
from datetime import datetime,timedelta
from sklearn.model_selection import train_test_split

from utils_imp import *
from parameters import *

logger = logging.getLogger(__name__)
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


def main():
    args = parse_args()
    start_time = time.time()

    accelerator = Accelerator(fp16=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            os.makedirs(f"{args.output_dir}/last", exist_ok=True)

    if accelerator.is_main_process:
        logfilename = 'log_bs{}_lr{}_sp{}_time_{}.txt'.format(args.per_device_train_batch_size, args.learning_rate, args.pruning_sparsity, datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        logfilename = os.path.join(args.output_dir, logfilename)
        handler = logging.FileHandler(logfilename)
        logger.addHandler(handler)
        logger.info('------------> log file =={}'.format(logfilename))
        logger.info(args)
    logger.info(accelerator.state)
    accelerator.wait_for_everyone()


############################################################################################      
############################################################################################      
############################################################################################      
############################################################################################      
############################################################################################      
############################################################################################      

    args.is_regression = args.task_name == "stsb"

    train_dataloader, eval_dataloader, eval_train_dataloader, train_dataset, eval_dataset, eval_train_dataset, num_labels, tokenizer = get_dataset_and_dataloader(args)

    model, teacher = get_teacher_and_student(args, num_labels)

    optimizer = get_optimizer(args, model)

    model, teacher, optimizer, train_dataloader = accelerator.prepare(
        model, teacher, optimizer, train_dataloader
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=int(args.max_train_steps * 0.1),
        num_training_steps=args.max_train_steps,
    )

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {1}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    train_one_phase(args, model, teacher, train_dataloader, accelerator, optimizer, lr_scheduler, eval_dataloader, eval_dataset, tokenizer, None, 1, logger)
    eval_metric = evaluate_data(
        eval_dataloader,
        eval_dataset,
        model,
        args,
        logger,
        mode="dev"
    )
    logger.info(f"Dev Dataset Result: {eval_metric}")

    logger.info("")
    logger.info("")
    logger.info("")
    logger.info("*" * 30)
    logger.info("*" * 30)
    logger.info("*" * 30)
    logger.info("*" * 30)
    logger.info("")
    logger.info("")
    logger.info("")

    pruner = None
    final_sparsity = args.pruning_sparsity 
    sparsities_schedule = [0.54, 0.83, 0.92, 0.9375]
    for idx,s in enumerate(sparsities_schedule):
        del pruner
        del lr_scheduler

        pruner = get_pruner(args, model, s)
        args.max_train_steps = num_update_steps_per_epoch * args.num_train_epochs
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type if idx != len(sparsities_schedule) - 1 else "linear",
            optimizer=optimizer,
            num_warmup_steps=int(args.max_train_steps * 0.1),
            num_training_steps=args.max_train_steps,
        )
        
        model = train_one_phase(args, model, teacher, train_dataloader, accelerator, optimizer, lr_scheduler, eval_dataloader, eval_dataset, tokenizer, pruner, args.num_train_epochs, logger)
        eval_metric = evaluate_data(
            eval_dataloader,
            eval_dataset,
            model,
            args,
            logger,
            mode="dev"
        )
        logger.info(f"Dev Dataset Result: {eval_metric}")
           
        logger.info("")
        logger.info("")
        logger.info("")
        logger.info("*" * 30)
        logger.info("*" * 30)
        logger.info("*" * 30)
        logger.info("*" * 30)
        logger.info("")
        logger.info("")
        logger.info("")
        logger.info("")

    eval_metric = evaluate_data(
        eval_dataloader,
        eval_dataset,
        model,
        args,
        logger,
        mode="dev"
    )
    logger.info(f"Dev Dataset Result: {eval_metric}")

    eval_metric = evaluate_data(
        eval_train_dataloader,
        eval_train_dataset,
        model,
        args,
        logger,
        mode="train"
    )
    logger.info(f"Train Dataset Result: {eval_metric}")      
            
    if accelerator.is_main_process:
        total_time = time.time() - start_time
        total_time_str = str(timedelta(seconds=int(total_time)))
        logger.info('Training time {}'.format(total_time_str))

    logger.info('Totoal Variance: {}'.format(pruner._variance))
    logger.info("Success")

def get_dataset_and_dataloader(args):
    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mnli-mm": MnliMismatchedProcessor,
        "mrpc": MrpcProcessor,
        "sst2": Sst2Processor,
        "stsb": StsbProcessor,
        "qqp": QqpProcessor,
        "qnli": QnliProcessor,
        "rte": RteProcessor,
        "wnli": WnliProcessor
    }

    output_modes = {
        "cola": "classification",
        "mnli": "classification",
        "mrpc": "classification",
        "sst2": "classification",
        "stsb": "regression",
        "qqp": "classification",
        "qnli": "classification",
        "rte": "classification",
        "wnli": "classification"
    }

    default_params = {
        "cola": {"num_train_epochs": 50, "max_seq_length": 64},
        "mnli": {"num_train_epochs": 5, "max_seq_length": 128},
        "mrpc": {"num_train_epochs": 20, "max_seq_length": 128},
        "sst2": {"num_train_epochs": 10, "max_seq_length": 64},
        "stsb": {"num_train_epochs": 20, "max_seq_length": 128},
        "qqp": {"num_train_epochs": 5, "max_seq_length": 128},
        "qnli": {"num_train_epochs": 10, "max_seq_length": 128},
        "rte": {"num_train_epochs": 20, "max_seq_length": 128}
    }

    task_name = args.task_name
    processor = processors[task_name]()
    output_mode = output_modes[task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)
    max_seq_length = default_params[task_name]["max_seq_length"]
    
    if not args.aug_train:
        train_examples = processor.get_train_examples(args.data_dir)
        eval_train_examples = train_examples
    else:
        train_examples = processor.get_aug_examples(args.data_dir)
        eval_train_examples = processor.get_train_examples(args.data_dir)

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)

    train_features = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, output_mode)
    train_dataset, train_labels = get_tensor_data(output_mode, train_features)

    eval_train_features = convert_examples_to_features(eval_train_examples, label_list, max_seq_length, tokenizer, output_mode)
    eval_train_dataset, eval_train_labels = get_tensor_data(output_mode, eval_train_features)

    eval_examples = processor.get_dev_examples(args.data_dir)
    eval_features = convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer, output_mode)
    eval_dataset, eval_labels = get_tensor_data(output_mode, eval_features)

    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.per_device_eval_batch_size)
    eval_train_dataloader = DataLoader(eval_train_dataset, batch_size=args.per_device_eval_batch_size)

    return train_dataloader, eval_dataloader, eval_train_dataloader, train_dataset, eval_dataset, eval_train_dataset, num_labels, tokenizer

def get_teacher_and_student(args, num_labels):
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )

    teacher = None
    if args.kd:
        teacher = AutoModelForSequenceClassification.from_pretrained(
            args.teacher,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )
        teacher.eval()

    return model, teacher

def get_optimizer(args, model):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    return optimizer

def get_pruner(args, model, sparsity):

    prune_dict = {}
    for k, v in model.named_parameters():
        if ('intermediate.dense.weight' in k or 'output.dense.weight' in k) and ('attention.output.dense.weight' not in k):
            prune_dict[k] = sparsity
        if 'attention.self.query.weight' in k or 'attention.self.key.weight' in k or 'attention.self.value.weight' in k or 'attention.output.dense.weight' in k:
            prune_dict[k] = sparsity
            
    pruner = None
    if args.prune:
        pruner = Prune(
            model=model, 
            prune_type=args.prune_type,
            pretrain_step=0,
            sparse_step=1,
            current_step=args.current_step,
            frequency=args.pruning_frequency,
            prune_dict=prune_dict,
            prune_device='default',
            fix_sparsity=args.fix_sparsity,
            restore_sparsity=args.restore_sparsity,
            sample_count=args.sample_count,
            sample_ratio=args.sample_ratio,
            sample_layer=args.sample_layer,
            logger=logger
        )
    
    pruner.prune()
    layer_sparse_rate, total_sparse_rate = pruner.prune_sparsity()
    logger.info('\nepoch %d; step=%d; weight sparsity=%s; layer weight sparsity=%s\n' % (0, 0, total_sparse_rate, layer_sparse_rate))

    return pruner

if __name__ == "__main__":
    main()
