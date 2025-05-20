import sys
sys.path.append('./') 
sys.path.insert(0, './policy/ACT-DP-TP')
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import json
import pickle
import argparse
from copy import deepcopy
from tqdm import tqdm
import yaml
import traceback
import warnings
import importlib
from collections import OrderedDict
from datetime import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import IPython
e = IPython.embed
import sys
from policy import * 
from utils_robotwin import set_seed, convert_weigt

# global variables
CAMERA_NAMES = ['head_camera']


def class_decorator(task_name):
    envs_module = importlib.import_module(f'envs.{task_name}')
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def get_camera_config(camera_type):
    camera_config_path ='task_config/_camera_config.yml'

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f'camera {camera_type} is not defined'
    return args[camera_type]

def main_worker(args):
    set_seed(1)
    # command line parameters
    args = vars(args)
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    seed = args['seed']
    head_camera_type = 'D435'
    expert_data_num = 50
        
    #######################  policy parameters ####################### 
    print('####################### Step 1: get policy #######################')
    lr_backbone = 1e-5 
    backbone = args['backbone']
    # backbone = 'resnet18' # TODO maybe change to frozed tokenizer
    if 'ACT' in policy_class: 
        enc_layers = 4 
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'task_name': task_name,
                         'num_queries': args['chunk_size'],
                         'predict_frame': args['predict_frame'],
                         'history_step': args['history_step'],
                         'image_downsample_rate': args['image_downsample_rate'],
                         'resize_rate': args['resize_rate'],
                         'temporal_downsample_rate': args['temporal_downsample_rate'],
                         'image_height': 240, # Hard code
                         'image_width': 320, # Hard code
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': CAMERA_NAMES,
                         'norm_type': args['norm_type'],
                         'disable_vae_latent': args['disable_vae_latent'],
                         'disable_resnet': args['disable_resnet'],
                         # design diffusion parameters
                         'num_inference_steps':args['num_inference_steps'],
                         'num_train_timesteps':args['num_train_steps'],
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
                         # multitask
                        'use_film': False,
                        'multitask': False,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': CAMERA_NAMES,}
    else:
        raise NotImplementedError
    policy = make_policy(policy_class, policy_config)
    usr_args = deepcopy(args)
    
    #######################  env parameters ####################### 
    print('####################### Step 2: get env #######################')
    
    with open(f'./task_config/{task_name}.yml', 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)
    args['head_camera_type'] = head_camera_type
    head_camera_config = get_camera_config(args['head_camera_type'])
    args['head_camera_fovy'] = head_camera_config['fovy']
    args['head_camera_w'] = head_camera_config['w']
    args['head_camera_h'] = head_camera_config['h']
    head_camera_config = 'fovy' + str(args['head_camera_fovy']) + '_w' + str(args['head_camera_w']) + '_h' + str(args['head_camera_h'])
    
    wrist_camera_config = get_camera_config(args['wrist_camera_type'])
    args['wrist_camera_fovy'] = wrist_camera_config['fovy']
    args['wrist_camera_w'] = wrist_camera_config['w']
    args['wrist_camera_h'] = wrist_camera_config['h']
    wrist_camera_config = 'fovy' + str(args['wrist_camera_fovy']) + '_w' + str(args['wrist_camera_w']) + '_h' + str(args['wrist_camera_h'])

    front_camera_config = get_camera_config(args['front_camera_type'])
    args['front_camera_fovy'] = front_camera_config['fovy']
    args['front_camera_w'] = front_camera_config['w']
    args['front_camera_h'] = front_camera_config['h']
    front_camera_config = 'fovy' + str(args['front_camera_fovy']) + '_w' + str(args['front_camera_w']) + '_h' + str(args['front_camera_h'])

    # output camera config
    print('============= Camera Config =============\n')
    print('Head Camera Config:\n    type: '+ str(args['head_camera_type']) + '\n    fovy: ' + str(args['head_camera_fovy']) + '\n    camera_w: ' + str(args['head_camera_w']) + '\n    camera_h: ' + str(args['head_camera_h']))
    print('Wrist Camera Config:\n    type: '+ str(args['wrist_camera_type']) + '\n    fovy: ' + str(args['wrist_camera_fovy']) + '\n    camera_w: ' + str(args['wrist_camera_w']) + '\n    camera_h: ' + str(args['wrist_camera_h']))
    print('Front Camera Config:\n    type: '+ str(args['front_camera_type']) + '\n    fovy: ' + str(args['front_camera_fovy']) + '\n    camera_w: ' + str(args['front_camera_w']) + '\n    camera_h: ' + str(args['front_camera_h']))
    print('\n=======================================')
    
    task = class_decorator(args['task_name'])
    stats_path = os.path.join('/mnt/mydisk/Project_folder/0_RoboTwin/RoboTwin',ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f) # for normalization
    print(f'Load dataset stats from {stats_path}')
    args['norm_type'] = policy_config['norm_type'] 
    args['expert_data_num'] = expert_data_num 
    args['expert_seed'] = seed
    args['embedding_type'] = usr_args['embedding_type']
    args['eval_video_log'] = usr_args['eval_video_log']
    print('norm_type:', args['norm_type'])
    print('####################### Step 3: begin eval #######################')
    st_seed = 100000 * (1+seed)
    suc_nums = []
    test_num = 50
    topk = 1
    start_time = time.time()
    ckpt_names = []# TODO: change to model_best.pth
    if usr_args['eval_ckpts'] == 0:
            ckpt_names = ['policy_last.ckpt'] #,'policy_best.ckpt'
    elif usr_args['eval_ckpts'] == 1:
            ckpt_names = ['policy_best.ckpt'] #,'policy_best.ckpt'
    elif usr_args['eval_ckpts'] > 0: 
        ckpt_names.append('policy_epoch_'+str(usr_args['eval_ckpts'])+'_seed_'+str(seed)+'.ckpt')
    for ckpt_name in ckpt_names:
        print(f'Load checkpoint: {ckpt_name}')    
        ckpt_name_starts = ckpt_name.split('.')[0]
        ckpt_path = os.path.join('policy/ACT-DP-TP',ckpt_dir, ckpt_name) # 
        state_dict = torch.load(ckpt_path, map_location='cpu') # TODO
        sub_state_dict = OrderedDict()
        for k, v in state_dict.items(): 
            if k.startswith('model.'):
                sub_state_dict[k[len('model.'):]] = v
        loading_status = policy.model.load_state_dict(convert_weigt(sub_state_dict))
        print(loading_status)
        policy.cuda()
        policy.eval()
        
        start_time = time.time()
        st_seed, suc_num, success_list = test_policy(task_name, stats, task, args, policy, st_seed, test_num=test_num)
        file_path = os.path.join(ckpt_dir,  'results.txt')
        cost_time = time.time() - start_time
        print(f'Cost time: {cost_time} s')
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f'Checkpoint: {ckpt_name_starts}')
        with open(file_path, 'a') as file:
            file.write('\n\n')
            file.write(f'Task: {task_name}\n')
            file.write(f'Policy: {policy_class}\n')
            file.write(f'diffusion inference step: {usr_args["num_inference_steps"]}\n')
            file.write(f'Timestamp: {current_time}\n')
            file.write(f'cost time: {cost_time} s\n')
            success_rate = suc_num / test_num
            file.write(f'Checkpoint: {ckpt_name_starts}\n')
            file.write(f'Success rate: {suc_num}/{test_num} = {success_rate }\n')
            file.write(str(success_list))
            file.write('\n\n')
            # file.write('Successful Rate of Diffenent checkpoints:\n')
            # file.write('\n'.join(map(str, np.array(suc_nums) / test_num)))
            # file.write('\n\n')
    suc_nums.append(suc_num)
    topk_success_rate = sorted(suc_nums, reverse=True)[:topk]
    print(f'TopK {topk} Success Rate (every):')
    print('best ckpt name', ckpt_names[np.argmax(topk_success_rate)])
    with open(file_path, 'a') as file:
        file.write(f'TopK {topk} Success Rate (every):\n')
        file.write('\n'.join(map(str, np.array(topk_success_rate) / test_num)))
        file.write('\n\n')

    print(f'Data has been saved to {file_path}')
    return 0
    
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


def test_policy(task_name, stats, Demo_class, args, policy, st_seed, test_num):
    expert_check = True
    print("Task name: ", args["task_name"])


    Demo_class.suc = 0
    Demo_class.test_num =0

    now_id = 0
    succ_seed = 0
    suc_test_seed_list = []
    success_list = []
    now_seed = st_seed
    while succ_seed < test_num: # test test_num rollouts
        print('now_seed:', now_seed)
        render_freq = args['render_freq']
        args['render_freq'] = 0
        expert_check = True
        if expert_check:
            try:
                # print('start setup_demo')
                Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args) # setup the task env? 
                # print('end setup_demo')
                Demo_class.play_once()
                # print('play_once')
                Demo_class.close()
                # print('close')
            except Exception as e:
                stack_trace = traceback.format_exc()
                print(' -------------')
                print('Error: ', stack_trace)
                print(' -------------')
                Demo_class.close()
                now_seed += 2
                args['render_freq'] = render_freq
                print('error occurs !')
                continue

        if (not expert_check) or ( Demo_class.plan_success and Demo_class.check_success() ):
            succ_seed +=1 # scripted policy is successful to test the task env
            suc_test_seed_list.append(now_seed) #save the seed of successful test
        else:
            now_seed += 1
            args['render_freq'] = render_freq
            continue


        args['render_freq'] = render_freq
        Demo_class.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
        Demo_class.apply_my_policy(args, policy, stats, args['norm_type']) # apply policy to the task apply_my_policy

        now_id += 1 # update the episode number
        Demo_class.close()
        if Demo_class.render_freq:
            Demo_class.viewer.close()
        # dp.runner.reset_obs() # reset the observation
        print(f"{task_name} success rate: {Demo_class.suc}/{Demo_class.test_num}, current seed: {now_seed}\n")
        Demo_class._take_picture()
        now_seed += 1
        success_list.append(Demo_class.suc)
        
    return now_seed, Demo_class.suc, success_list

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_ckpts', default=0,type=int)
    parser.add_argument('--eval_video_log', action='store_true')
    parser.add_argument('--no_sigmoid', action='store_true')
    parser.add_argument('--action_interval', default=1, type=int)
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--num_episodes', type=int, help='num_episodes',default=0, required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', default=100,type=int, help='num_epochs', required=False)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)
    parser.add_argument('--lr_schedule_type', default='cosine_warmup', type=str, help='lr_schedule_type')
    parser.add_argument('--backbone', default='resnet18', type=str, help='backbone')
    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--save_epoch', action='store', type=int, help='save_epoch', default=500, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--history_step',default=0 , type=int, help='history_step', required=False)
    parser.add_argument('--predict_frame',default=0 , type=int, help='predict_frame', required=False)
    parser.add_argument('--resize_rate',default=1 , type=int, help='resize_rate for future image prediction', required=False)
    parser.add_argument('--image_downsample_rate',default=1 , type=int, help='image_downsample_rate', required=False)
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
    # parser.add_argument('--tokenizer_model_name', default='Cosmos-Tokenizer-DV8x16x16', type=str, help='tokenizer_model_name')
    parser.add_argument('--prediction_weight', default=1, type=float, help='pred token Weight', required=False)
    parser.add_argument('--patch_size', default=5, type=int, help='patch_size', required=False)
    parser.add_argument('--token_pe_type', default='learned', type=str, help='token_pe_type', required=False)
    parser.add_argument(
        "--embedding_type",
        default="cls_embedding",
        type=str,
        help="task_embedding_type",
        required=False,
    )
    args = parser.parse_args()
    if args.tokenizer_model_type in ['DV', 'CV']:
        args.tokenizer_model_name = f'Cosmos-Tokenizer-{args.tokenizer_model_type}{args.tokenizer_model_temporal_rate}x{args.tokenizer_model_spatial_rate}x{args.tokenizer_model_spatial_rate}'
    elif args.tokenizer_model_type in ['DI', 'CI']:
        args.tokenizer_model_name = f'Cosmos-Tokenizer-{args.tokenizer_model_type}{args.tokenizer_model_spatial_rate}x{args.tokenizer_model_spatial_rate}'
        args.tokenizer_model_temporal_rate = 1
    else:
        raise NotImplementedError
    
    if args.tokenizer_model_type in ['DV', 'DI']:
        args.token_dim = 6
    else:
        args.token_dim = 16
            
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(os.path.join(args.ckpt_dir,"eval_args_config.json"), "w") as json_file:
        json.dump(vars(args), json_file, indent=4)
    main_worker(args) #  separate the main_worker to the script
