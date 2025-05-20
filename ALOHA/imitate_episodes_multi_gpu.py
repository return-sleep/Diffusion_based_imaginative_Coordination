import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import json
import pickle
import argparse
from copy import deepcopy
import matplotlib.pyplot as plt
from tqdm import tqdm
from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from sim_env import BOX_POSE
from policy import *
from utils import load_data, load_data_unified_preload# data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict, plot_history, get_image, convert_weigt # helper functions
from utils import get_cosine_schedule_with_warmup,get_constant_schedule
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import IPython
e = IPython.embed

# os.environ['MUJOCO_GL'] = 'egl'

def main_worker(args):
    set_seed(1)
    # command line parameters
    args = vars(args)
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size'] 
    num_epochs = args['num_epochs']
    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
        
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes'] if args['num_episodes'] == 0 else args['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    disable_vae_latent = args['disable_vae_latent']
    num_inference_steps = args['num_inference_steps']
    num_train_steps = args['num_train_steps']

    #######################  fixed parameters ####################### 
    state_dim = 14
    lr_backbone = 1e-5 
    backbone = args['backbone'] 
    if 'ACT' in policy_class: 
        enc_layers = 4 
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'is_wandb': args['is_wandb'],
                         'task_name':args['task_name'],
                         'save_epoch':  min(args['num_epochs'] // 20,10), 
                         'num_queries': args['chunk_size'],
                         'predict_frame': args['predict_frame'],
                         'history_step': args['history_step'],
                         'resize_rate': args['resize_rate'],
                         'image_downsample_rate': args['image_downsample_rate'],
                         'temporal_downsample_rate': args['temporal_downsample_rate'],
                         'image_height': 480, # Hard code
                         'image_width': 640, # Hard code
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'norm_type': args['norm_type'],
                         'disable_vae_latent': disable_vae_latent,
                         'disable_resnet': args['disable_resnet'],
                         # design diffusion parameters
                         'num_inference_steps':num_inference_steps,
                         'num_train_timesteps':num_train_steps,
                         'prediction_type':args['prediction_type'],
                         'loss_type':args['loss_type'],
                         'schedule_type':args['schedule_type'],
                         'beta_schedule':args['beta_schedule'],
                         'diffusion_timestep_type':args['diffusion_timestep_type'],
                         'attention_type':args['attention_type'],
                         'causal_mask':args['causal_mask'],
                         'predict_only_last':args['predict_only_last'],
                         'share_decoder':args['share_decoder'],
                         # visual toknizer
                         'tokenizer_model_temporal_rate': args['tokenizer_model_temporal_rate'],
                         'tokenizer_model_spatial_rate': args['tokenizer_model_spatial_rate'],
                         'tokenizer_model_name': args['tokenizer_model_name'],
                         'prediction_weight': args['prediction_weight'],
                         'imitate_weight': args['imitate_weight'],
                         'token_dim': args['token_dim'],
                         'patch_size': args['patch_size'],
                         'token_pe_type': args['token_pe_type'],
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names,}
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        'onscreen_render': onscreen_render,
        'policy_config': policy_config,
        'task_name': task_name,
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': not is_sim,
        'batch_size_train': batch_size_train,
        'lr_backbone': lr_backbone,
        'norm_type': args['norm_type'], # FOR action normalization
        'lr_scheduler': args['lr_schedule_type'],
        'share_decoder': args['share_decoder'],
        'image_downsample_rate': args['image_downsample_rate'],
        'temporal_downsample_rate': args['temporal_downsample_rate'],
        'patch_size': args['patch_size'],
        'diffusion_timestep_type': args['diffusion_timestep_type'],
        'token_pe_type': args['token_pe_type'],
        'attention_type': args['attention_type'],
    }
    
    #######################  Evalutation ############################
    # For evaluation on aloha
    if is_eval:
        start_time = time.time()
        ckpt_names = []
        if args['eval_ckpts'] == 1:
            ckpt_names = ['policy_last.ckpt']
        elif args['eval_ckpts'] > 0:
            ckpt_names.append('policy_epoch_'+str(args['eval_ckpts'])+'_seed_'+str(args['seed'])+'.ckpt')
        elif args['eval_ckpts'] == 0:
            files = sorted(os.listdir(config['ckpt_dir']))
            ckpt_names = [file for file in files if file.endswith('.ckpt') and 'lastest_seed' not in file]
        print(f'ckpt_names: {ckpt_names}')
        results = []
        success_rate_list = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=False)
            results.append([ckpt_name, success_rate, avg_return])
            success_rate_list.append(success_rate)
        
        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate} {avg_return}')
            
        top_success_rate = max(success_rate_list)
        top_k_success_rate = sorted(success_rate_list, reverse=True)[:3]
        top_k_mean = np.mean(top_k_success_rate)
        # write to file
        result_file_name = 'result.txt'
        with open(os.path.join(ckpt_dir, result_file_name), 'a') as f:
            f.write(f'Best ckpt: {ckpt_names[success_rate_list.index(top_success_rate)]}\n')
            f.write(f'Top success rate: {top_success_rate}\n')
            f.write(f'Top 3 success rate: {top_k_success_rate}\n')
            f.write(f'Top 3 Mean success rate : {top_k_mean}\n')
        print(f'Evaluation Time: {time.time()-start_time}')
        print()
        exit()

    #######################  load data ############################
    distributed = False
    if (policy_class == 'ACT' or policy_class == 'ACT_diffusion'):
        print('Load data')
        train_dataloader, val_dataloader, train_sampler, stats, _ = load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, samples_per_epoch=args['samples_per_epoch'],distributed=distributed)
    else: 
        print('Load data unified')
        train_dataloader, val_dataloader, train_sampler, stats, _ = load_data_unified_preload(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, args['chunk_size'], history_step = args['history_step'], predict_frame = args['predict_frame'], image_downsample_rate=args['image_downsample_rate'],temporal_downsample_rate= args['temporal_downsample_rate'], predict_only_last= args['predict_only_last'], samples_per_epoch=args['samples_per_epoch'], distributed=distributed)  
    
    #######################  Train Phase ############################
    # save dataset stats
    os.makedirs(ckpt_dir, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config, train_sampler,  stats)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    log_message = f'Best ckpt, val loss {min_val_loss:.6f} @ epoch {best_epoch}'
    print(log_message)
    # save training log
    log_dir = os.path.join(ckpt_dir, 'training_log.txt')
    with open(log_dir, "a") as file:  
        file.write(log_message + "\n")

def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == 'ACT_diffusion_tp':
        policy = ACTPolicyDiffusion_with_Token_Prediction(policy_config)
    else:
        raise NotImplementedError
    print(f'Policy: {policy_class}')
    return policy


def forward_pass(config, data, policy, stats=None, is_training = True, downsample_rate=1, temporal_downsample_rate=1): # TODO discard downsample_rate and temporal_downsample_rate and predict_only_last
    # TODO: add ACT_DP_policy
    if isinstance(policy, (ACTPolicy, CNNMLPPolicy)):
        image_data, qpos_data, action_data, is_pad = data[:4] # normalized action
        image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
        action_data  = normalize_data(action_data,stats,config['norm_type'],data_type='action')
        image_data = image_data[...,::downsample_rate,::downsample_rate]
        return policy(qpos_data, image_data, action_data, is_pad, is_training) # 
    else:
        # unify the forward pass, dataloader ouput preprocessing
        image_data, qpos_data, action_data, is_pad, future_imgs_data, is_pad_img = data # raw action
        image_data, qpos_data, action_data, is_pad, future_imgs_data, is_pad_img = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda(), future_imgs_data.cuda(), is_pad_img.cuda()
        action_data  = normalize_data(action_data,stats,config['norm_type'],data_type='action')
        return policy(qpos_data, image_data, action_data, is_pad, future_imgs_data, is_pad_img, is_training) # ,is_training TODO remove None


def train_bc(train_dataloader, val_dataloader, config, train_sampler=None, stats=None):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']
    downsample_rate = config['image_downsample_rate'] # downsample rate for image space
    temporal_downsample_rate = config['temporal_downsample_rate'] # downsample rate for temporal space
    print('ckpt_dir:', ckpt_dir)
    set_seed(seed)

    # make policy
    policy = make_policy(policy_class, policy_config)
    policy.cuda()

    # make optimizer
    param_dicts = [
        {"params": [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": policy_config['lr_backbone'],
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=policy_config['lr'],
                                  weight_decay=1e-4)
    if config['lr_scheduler'] == 'constant':
        scheduler = get_constant_schedule(optimizer)
    elif config['lr_scheduler'] == 'cosine_warmup':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=(num_epochs+1)*len(train_dataloader))
    
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    update_step = 0
    start_epoch = 0
    for epoch in tqdm(range(start_epoch, num_epochs+1)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(policy_config, data, policy, stats, is_training=False,downsample_rate = downsample_rate, temporal_downsample_rate = temporal_downsample_rate)
                epoch_dicts.append(forward_dict)
            # log
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)
            epoch_val_loss = epoch_summary['loss']
            print(f'Val loss:   {epoch_val_loss:.5f}')
            summary_string = ''
            for k, v in epoch_summary.items():
                summary_string += f'{k}: {v.item():.4f} '
            print(summary_string)
            # save best val ckpt
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
                torch.save(policy.state_dict(), ckpt_path)
              
        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(tqdm(train_dataloader, desc=f"Train Epoch {epoch+1}/{num_epochs+1}", leave=False)):
            forward_dict = forward_pass(policy_config, data, policy,stats, is_training=True, downsample_rate = downsample_rate,temporal_downsample_rate = temporal_downsample_rate)
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            update_step += 1
            scheduler.step()
        # log    
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*(epoch-start_epoch):(batch_idx+1)*(epoch+1-start_epoch)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.4f} '
        print(summary_string)
        print('lr:', optimizer.param_groups[0]['lr'])
        # save train ckpt
        if epoch % config['policy_config']['save_epoch'] == 0 and epoch > 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)
            
        lastest_ckpt = {
            'epoch': epoch,
            'state_dict': policy.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        lastest_ckpt_path = os.path.join(ckpt_dir, f'policy_lastest_seed_{seed}.ckpt') # avoid interrupt
        torch.save(lastest_ckpt, lastest_ckpt_path)
    
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')
    
    return best_ckpt_info


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    real_robot = config['real_robot']
    policy_class = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config = config['policy_config']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    state_dict = torch.load(ckpt_path, map_location='cuda')
    loading_status = policy.load_state_dict(convert_weigt(state_dict))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    if config['norm_type'] == 'minmax':
        post_process = lambda a: (a+1)/2 * (stats['action_max'] - stats['action_min']) + stats['action_min']
    else:
        post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    # load environment
    if real_robot:
        from aloha_scripts.robot_utils import move_grippers # requires aloha
        from aloha_scripts.real_env import make_real_env # requires aloha
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries'] 
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    print(f'query_frequency: {query_frequency}')
    max_timesteps = int(max_timesteps * 1) 
    success_num = 0
    num_rollouts =50
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                ### process previous timestep to get qpos and image_list
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)
                if config['image_downsample_rate'] > 1:
                    curr_image = curr_image[...,::config['image_downsample_rate'],::config['image_downsample_rate']]

                ### query policy
                if "ACT" in config['policy_class']:
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == "CNNMLP":
                    raw_action = policy(qpos, curr_image)
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                ts = env.step(target_qpos)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
 
            plt.close()
            
        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
            pass

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(int(episode_return))
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(int(episode_highest_reward))
        if episode_highest_reward == env_max_reward:
            success_num += 1
        
        print(
                f"Rollout {rollout_id}\n"
                f"episode_return={int(episode_return)}, "
                f"episode_highest_reward={int(episode_highest_reward)}, "
                f"env_max_reward={env_max_reward}, "
                f"Success: {int(episode_highest_reward) == env_max_reward}"
            )
        print(f'Rollout {rollout_id+1} : success_num: {success_num} / {rollout_id+1}')
                
    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'query_frequency: {query_frequency}\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(task_name, summary_str)

    # save success rate to txt
    result_file_name = 'result.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'a') as f:
        f.write('\n' + ckpt_name.split('.')[0]+ '\n')
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n')
        f.write(repr(highest_rewards))
        f.write('\n\n')
    print(ckpt_name.split('.')[0], 'saved to', os.path.join(ckpt_dir, result_file_name))
      
    return success_rate, avg_return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--eval_ckpts', default=0,type=int)
    parser.add_argument('--is_wandb', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--num_episodes', type=int, help='num_epochs',default=0, required=False)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--samples_per_epoch', default=1, type=int, help='samples_per_epoch')
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--lr_schedule_type', default='constant', type=str, help='lr_schedule_type')
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--save_epoch', action='store', type=int, help='save_epoch', default=500, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--history_step',default=0 , type=int, help='history_step', required=False)
    parser.add_argument('--predict_frame',default=0 , type=int, help='predict_frame', required=False)
    parser.add_argument('--image_downsample_rate',default=1 , type=int, help='image_downsample_rate', required=False)
    parser.add_argument('--resize_rate',default=1 , type=int, help='resize_rate for future image prediction', required=False)
    parser.add_argument('--temporal_downsample_rate',default=1 , type=int, help='temporal_downsample_rate', required=False)
    parser.add_argument('--predict_only_last', action='store_true') # only predict the last #predict_frame frame
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    # prediction_type for diffusion
    parser.add_argument('--norm_type', default='minmax', type=str, help='norm_type')
    parser.add_argument('--num_train_steps', default=100, type=int, help='num_train_steps')
    parser.add_argument('--num_inference_steps', default=10, type=int, help='num_inference_steps')
    parser.add_argument('--imitate_weight', default=1, type=int, help='imitate Weight', required=False)
    parser.add_argument('--schedule_type', default='DDIM', type=str, help='scheduler_type')
    parser.add_argument('--prediction_type', default='sample', type=str, help='prediction_type')
    parser.add_argument('--beta_schedule', default='squaredcos_cap_v2', type=str, help='prediction_type')
    parser.add_argument('--loss_type', default='l1', type=str, help='loss_type')
    parser.add_argument('--diffusion_timestep_type', default='cat', type=str, help='diffusion_timestep_type, cat or add, how to combine timestep')
    parser.add_argument('--attention_type', default='v0', help='decoder attention type')
    parser.add_argument('--causal_mask', action='store_true', help='use causal mask for diffusion')
    parser.add_argument('--disable_vae_latent', action='store_true', help='Use VAE latent space by default')
    parser.add_argument('--disable_resnet', action='store_true', help='Use resnet to encode obs image  by default')
    parser.add_argument('--share_decoder', action='store_true', help='jpeg and action share decoder')
    # visual tokenizer
    parser.add_argument('--tokenizer_model_type', default='DV', type=str, help='tokenizer_model_type, DV,CV,DI,CI')
    parser.add_argument('--tokenizer_model_temporal_rate', default=8, type=int, help='tokenizer_model_temporal_rate, 4,8')
    parser.add_argument('--tokenizer_model_spatial_rate', default=16, type=int, help='tokenizer_model_spatial_rate, 8,16')
    parser.add_argument('--prediction_weight', default=1, type=float, help='pred token Weight', required=False)
    parser.add_argument('--patch_size', default=5, type=int, help='patch_size', required=False)
    parser.add_argument('--token_pe_type', default='learned', type=str, help='token_pe_type', required=False)
    args = parser.parse_args()
    if args.tokenizer_model_type in ['DV', 'CV']: # Video tokenizer
        args.tokenizer_model_name = f'Cosmos-Tokenizer-{args.tokenizer_model_type}{args.tokenizer_model_temporal_rate}x{args.tokenizer_model_spatial_rate}x{args.tokenizer_model_spatial_rate}'
    elif args.tokenizer_model_type in ['DI', 'CI']: # Image tokenizer
        args.tokenizer_model_name = f'Cosmos-Tokenizer-{args.tokenizer_model_type}{args.tokenizer_model_spatial_rate}x{args.tokenizer_model_spatial_rate}'
        args.tokenizer_model_temporal_rate = 1
    else:
        raise NotImplementedError
    
    if args.tokenizer_model_type in ['DV', 'DI']:
        args.token_dim = 6
    else:
        args.token_dim = 16
        
    os.makedirs(args.ckpt_dir, exist_ok=True)
    if args.eval:
        with open(os.path.join(args.ckpt_dir,"eval_args_config.json"), "w") as json_file:
            json.dump(vars(args), json_file, indent=4)
    else:
        with open(os.path.join(args.ckpt_dir,"train_args_config.json"), "w") as json_file:
            json.dump(vars(args), json_file, indent=4)
    
    main_worker(args)

    # main(vars(parser.parse_args()))
