#!/bin/bash

set -x
set -u

now=$(date +"%Y%m%d_%H%M%S")
jobname="sdf-eval-$now"
echo "job name is $jobname"

config_file="config/train_brandenburg_gate.yaml"
ckpt_path="checkpoints/xxx.ckpt"
eval_level=10

python -m torch.distributed.launch \
       --nproc_per_node=1 tools/extract_mesh.py \
                                --cfg_path ${config_file} \
                                --mesh_size 1024 \
                                --chunk 102144 \
                                --ckpt_path $ckpt_path \
                                --mesh_radius 1 \
                                --mesh_origin "0, 0, 0" \
                                --chunk_rgb 1024 \
                                --vertex_color \
                                --eval_level ${eval_level} \
                                2>&1|tee log/${jobname}.log
