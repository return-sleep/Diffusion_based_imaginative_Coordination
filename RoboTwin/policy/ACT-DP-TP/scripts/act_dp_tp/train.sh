#!/bin/bash
#SBATCH --job-name=act_dp_tp       # name
#SBATCH --mem=100G # memory pool for all cores`
#SBATCH --nodes=1                    # nodes
#SBATCH --cpus-per-task=10           # number of cores per tasks
#SBATCH --gres=gpu:a100:1            # number of gpus
#SBATCH --time 23:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=./outputs/act_dp_tp_%x-%j.out           # output file name
#SBATCH --error=./outputs/act_dp_tp_%x-%j.err            # error file nameW
#SBATCH --mail-user=xuh0e@kaust.edu.sa #Your Email address assigned for your job
#SBATCH --mail-type=ALL #Receive an email for ALL Job S

task_name=$1
cuda_id=$2
seed=$3

num_epochs=300
chunk_size=20
predict_frame=20
temporal_downsample_rate=5
tokenizer_model_temporal_rate=4
lr_schedule_type=cosine_warmup
attention_type=v3
export CUDA_VISIBLE_DEVICES=$cuda_id
echo "Processing $task_name"
python3 train_policy_robotwin.py \
    --task_name  $task_name \
    --ckpt_dir checkpoints/$task_name/act_dp_tp_${chunk_size}_${predict_frame}_${temporal_downsample_rate}_${tokenizer_model_temporal_rate}_${lr_schedule_type}/seed_$seed/num_epochs_$num_epochs \
    --policy_class ACT_diffusion_tp --hidden_dim 512  --batch_size 32 --dim_feedforward 3200 \
    --chunk_size $chunk_size  --norm_type minmax --disable_vae_latent \
    --predict_frame $predict_frame --patch_size 5 --temporal_downsample_rate $temporal_downsample_rate   --prediction_weight 0.2  \
    --tokenizer_model_temporal_rate $tokenizer_model_temporal_rate --tokenizer_model_spatial_rate 8 \
    --num_epochs  $num_epochs --token_pe_type fixed --diffusion_timestep_type cat  \
    --lr 1e-4  --lr_schedule_type $lr_schedule_type  \
    --seed $seed \
    --kl_weight 10 \
    --share_decoder --attention_type $attention_type \
