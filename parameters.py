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
from data_processing import *
from datetime import datetime,timedelta

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="The name of the glue task to train on.",
        choices=list(task_to_keys.keys()),
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
        ),
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )    
    parser.add_argument(
        "--teacher",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument('--data_dir', type=str, default="")
    parser.add_argument("--eval_step", default=200, type=int, help="eval step.")
    parser.add_argument("--print_step", default=10, type=int, help="print step.")
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--kd', action='store_true')
    parser.add_argument('--aug_train', action='store_true')
    parser.add_argument('--prune',action='store_true')
    parser.add_argument('--prune_type',type=int, default=0)
    parser.add_argument('--sample_count',type=int, default=400)
    parser.add_argument('--sample_ratio',type=float, default=1e-5)
    parser.add_argument('--sample_layer',type=str, default="attention.output.dense.weight")
    parser.add_argument('--fix_sparsity', action='store_true')
    parser.add_argument('--restore_sparsity', action='store_true')
    parser.add_argument('--pruning_sparsity',type=float, default=0.875, help='sparsity')
    parser.add_argument("--current_step", default=0, type=int, help="current step.")
    parser.add_argument("--start_epoch", default=0, type=int, help="current epoch.")
    parser.add_argument('--pruning_frequency',type=int, default=800, help='also known as bank_size')
    parser.add_argument('--pruning_epochs',type=int, default=0, help='pruning epochs')
    parser.add_argument('--local_rank',type=int, default=0, help='rank')
    parser.add_argument('--do_eval',action='store_true')
    parser.add_argument('--early_stop',action='store_true')
    parser.add_argument('--early_stop_metric',default='accuracy', type=str, help="early stop metric")
    parser.add_argument('--save_last',action='store_true')
    parser.add_argument('--one_shot_prune',action='store_true')
    parser.add_argument('--sample_mask_count_1',type=int, default=2)
    parser.add_argument('--sample_mask_count_2',type=int, default=4)

    parser.add_argument('--adv_norm_type',type=str, default="l2")
    parser.add_argument('--adv_init_mag',type=float, default=0.05)
    parser.add_argument('--adv_steps',type=int, default=5)
    parser.add_argument('--adv_lr',type=float, default=0.03)
    parser.add_argument('--adv_max_norm', default=0, type=float,
                        help='adv_max_norm = 0 means unlimited')
    parser.add_argument('--max_grad_norm', default=1, type=float, help='max gradient norm')

    args = parser.parse_args()

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    args.lr_scheduler_type = SchedulerType[args.lr_scheduler_type.upper()]
    return args
