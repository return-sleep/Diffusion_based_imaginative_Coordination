#!/bin/bash
#SBATCH --job-name=act-eval       # name
#SBATCH --mem=100G # memory pool for all cores`
#SBATCH --nodes=1                    # nodes
#SBATCH --cpus-per-gpu=10           # number of cores per tasks
#SBATCH --gres=gpu:a100:1            # number of gpus
#SBATCH --time 23:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=./checkpoints/outputs/act_%x-%j.out           # output file name
#SBATCH --error=./checkpoints/outputs/act_%x-%j.err            # error file nameW
#SBATCH --mail-user=xuh0e@kaust.edu.sa #Your Email address assigned for your job
#SBATCH --mail-type=ALL #Receive an email for ALL Job S
#SBATCH --account conf-icml-2025.01.31-elhosemh

task_name=$1
chunk_size=$2
seed=$3
eval_ckpts=0
lr_schedule_type=cosine_warmup

# 
echo "Processing $task_name"
python3 eval_policy_robotwin.py \
        --task_name  $task_name \
        --ckpt_dir checkpoints_tmp/$task_name/act_${chunk_size}_${lr_schedule_type}/seed_$seed \
        --policy_class ACT --hidden_dim 512 --dim_feedforward 3200 \
        --chunk_size $chunk_size \
        --num_epochs 300 \
        --norm_type gaussian \
        --lr 1e-4 \
        --seed 0 \
        --kl_weight 10 \
        --eval_ckpts $eval_ckpts #--eval_video_log 

# python3 eval_policy_robotwin.py \
#         --task_name  $task_name \
#         --ckpt_dir checkpoints/$task_name/act_${chunk_size}/seed_$seed \
#         --policy_class ACT --hidden_dim 512 --dim_feedforward 3200 \
#         --chunk_size $chunk_size \
#         --num_epochs 300 \
#         --norm_type gaussian \
#         --lr 1e-4 \
#         --seed 0 \
#         --kl_weight 10 \
#         --eval_ckpts 0

# python3 eval_policy_robotwin.py \
#         --task_name  $task_name \
#         --ckpt_dir checkpoints/$task_name/act_${chunk_size}/seed_$seed \
#         --policy_class ACT --hidden_dim 512 --dim_feedforward 3200 \
#         --chunk_size $chunk_size \
#         --num_epochs 300 \
#         --norm_type gaussian \
#         --lr 1e-4 \
#         --seed 0 \
#         --kl_weight 10 \
#         --eval_ckpts 1

