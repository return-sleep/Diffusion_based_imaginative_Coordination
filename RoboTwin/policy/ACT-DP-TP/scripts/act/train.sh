#!/bin/bash
#SBATCH --job-name=act        # name
#SBATCH --mem=100G # memory pool for all cores`
#SBATCH --nodes=1                    # nodes
#SBATCH --cpus-per-gpu=10           # number of cores per tasks
#SBATCH --gres=gpu:a100:1            # number of gpus
#SBATCH --time 24:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=./checkpoints/outputs/act_%x-%j.out           # output file name
#SBATCH --error=./checkpoints/outputs/act_%x-%j.err            # error file nameW
#SBATCH --mail-user=xuh0e@kaust.edu.sa #Your Email address assigned for your job
#SBATCH --mail-type=ALL #Receive an email for ALL Job S


task_name=$1
num_epochs=$2
chunk_size=$3
seed=$4
lr_schedule_type=cosine_warmup # cosine_warmup 
backbone=resnet18
echo "Processing $task_name"
python3 train_policy_robotwin.py \
        --task_name  $task_name \
        --ckpt_dir checkpoints/$task_name/act_noaug_${chunk_size}_${lr_schedule_type}/seed_$seed \
        --policy_class ACT --hidden_dim 512 --batch_size 256  --dim_feedforward 3200 \
        --chunk_size $chunk_size \
        --num_epochs $num_epochs \
        --norm_type gaussian \
        --lr 1e-4 \
        --seed $seed \
        --kl_weight 10 \
        --dist-url 'tcp://localhost:10001' \
        --world-size 1 \
        --rank 0 \
        --lr_schedule_type $lr_schedule_type --backbone $backbone \
        # --is_wandb \
python3 eval_policy_robotwin.py \
        --task_name  $task_name \
        --ckpt_dir checkpoints/$task_name/act_noaug_${chunk_size}_${lr_schedule_type}/seed_$seed \
        --policy_class ACT --hidden_dim 512 --dim_feedforward 3200 \
        --chunk_size $chunk_size \
        --num_epochs 300 \
        --norm_type gaussian \
        --lr 1e-4 \
        --seed $seed \
        --kl_weight 10 \
        --eval_ckpts 0 --backbone $backbone