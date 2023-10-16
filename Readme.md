# Random Pruning Method

This is the official implementation of paper "Breaking through Deterministic Barriers: Randomized Pruning Mask Generation and Selection"

## Data Preparation

please refer to the data preparation of TinyBert

https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT

## Run Experiments for MRPC

```
cd mrpc
bash run_experiment_mrpc_determinstic.sh 0
bash run_experiment_mrpc_random_quickcall.sh 0
```

please replace the dataset file path with your own.

## Note

This repo is not perfect so far, more details will be added recently.
