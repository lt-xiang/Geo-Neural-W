#!/bin/bash

set -x
set -u

now=$(date +"%Y%m%d_%H%M%S")
jobname="brandenburg_gate-scale1-$now"

echo "job name is $jobname"

config_file='config/train_brandenburg_gate.yaml'

mkdir -p log
mkdir -p logs/${jobname}
cp ${config_file} logs/${jobname}

#export CUDA_VISIBLE_DEVICES=0

python train.py --cfg_path ${config_file} \
    --num_gpus 4 \
    --num_nodes 1 \
    --num_epochs 20 \
    --batch_size 8192 \
    --test_batch_size 512 \
    --num_workers 16 \
    --exp_name ${jobname} 2>&1|tee log/${jobname}.log 
