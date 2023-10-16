#/bin/bash
gpu=$1
prune_type=2
sample_ratio=0.0001 # deprecated

TASK_NAME=mrpc

CUDA_VISIBLE_DEVICES=$gpu python ../experiment_soups_quickcall.py \
  --teacher textattack/bert-base-uncased-MRPC \
  --model_name_or_path textattack/bert-base-uncased-MRPC \
  --data_dir /mnt/d/workspace/Dataset/nlp/glue_data/MRPC \
  --task_name $TASK_NAME \
  --aug_train \
  --max_length 128 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --learning_rate 5e-5 \
  --eval_step 50 \
  --print_step 10 \
  --num_train_epochs 10 \
  --early_stop \
  --early_stop_metric f1 \
  --kd \
  --prune \
  --lr_scheduler_type constant_with_warmup \
  --output_dir research/rand/3 \
  --pruning_sparsity 0.9375 \
  --prune_type $prune_type \
  --pruning_epochs 0 \
  --pruning_frequency 0 \
  --sample_ratio $sample_ratio \
  --sample_mask_count_1 8 \
  --sample_mask_count_2 8 \
  # --do_eval 
