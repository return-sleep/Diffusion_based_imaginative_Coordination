export MUJOCO_GL='egl'
samples_per_epoch=100 

task_name=$1
num_epochs=$2
seed=$3
cuda_id=$4
chunk_size=100
predict_frame=40
attention_type=v3
temporal_downsample_rate=5
tokenizer_model_temporal_rate=4
lr=5e-5
lr_schedule_type=cosine_warmup
diffusion_timestep_type=cat

echo "Processing $task_name" "$seed"
CUDA_VISIBLE_DEVICES=$cuda_id python3 imitate_episodes_multi_gpu.py \
        --task_name  $task_name \
        --ckpt_dir checkpoints/${task_name}/act_dp_tp_${attention_type}_${chunk_size}_${predict_frame}_${temporal_downsample_rate}_${tokenizer_model_temporal_rate}/seed_${seed}_${lr}_${lr_schedule_type}/${num_epochs} \
        --policy_class ACT_diffusion_tp --hidden_dim 512  --batch_size 32 --samples_per_epoch $samples_per_epoch --dim_feedforward 3200 \
        --chunk_size $chunk_size  --norm_type minmax --disable_vae_latent \
        --predict_frame $predict_frame --patch_size 5 --temporal_downsample_rate $temporal_downsample_rate   --prediction_weight 0.2  \
        --tokenizer_model_temporal_rate $tokenizer_model_temporal_rate --tokenizer_model_spatial_rate 8 \
        --num_epochs  $(($num_epochs / samples_per_epoch)) --token_pe_type fixed --diffusion_timestep_type $diffusion_timestep_type  \
        --lr $lr  --lr_schedule_type $lr_schedule_type  \
        --seed $seed \
        --kl_weight 10 \
        --share_decoder --attention_type $attention_type \
