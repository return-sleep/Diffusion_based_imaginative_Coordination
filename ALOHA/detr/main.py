# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import torch
from .models import *

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--eval_ckpts', default=0, type=int, help='eval_ckpts')
    parser.add_argument('--lr', default=1e-4, type=float) # will be overridden
    parser.add_argument('--lr_backbone', default=1e-5, type=float) # will be overridden
    parser.add_argument('--lr_schedule_type', default='constant', type=str, help='lr_schedule_type')
    parser.add_argument('--num_episodes', type=int, help='num_epochs',default=0, required=False)
    parser.add_argument('--batch_size', default=2, type=int) # not used
    parser.add_argument('--samples_per_epoch', default=1, type=int, help='samples_per_epoch', required=False)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int) # not used
    parser.add_argument('--lr_drop', default=200, type=int) # not used
    parser.add_argument('--clip_max_norm', default=0.1, type=float, # not used
                        help='gradient clipping max norm')
    parser.add_argument('--norm_type', default='meanstd', type=str, help='norm_type')
    parser.add_argument('--num_train_steps', default=50, type=int, help='num_train_steps')
    parser.add_argument('--num_inference_steps', default=10, type=int, help='num_inference_steps')
    parser.add_argument('--schedule_type', default='DDIM', type=str, help='scheduler_type')
    parser.add_argument('--imitate_weight', default=1, type=int, help='imitate Weight', required=False)
    parser.add_argument('--prediction_type', default='sample', type=str, help='prediction_type')
    parser.add_argument('--beta_schedule', default='squaredcos_cap_v2', type=str, help='prediction_type')
    parser.add_argument('--diffusion_timestep_type', default='cat', type=str, help='diffusion_timestep_type, cat or add, how to combine timestep')
    parser.add_argument('--attention_type', default='v0', help='decoder attention type')
    parser.add_argument('--causal_mask', action='store_true', help='use causal mask for diffusion')
    parser.add_argument('--loss_type', default='l2', type=str, help='loss_type')
    parser.add_argument('--disable_vae_latent', action='store_true', help='Use VAE latent space by default')
    parser.add_argument('--disable_resnet', action='store_true', help='Use resnet to encode obs image  by default')
    parser.add_argument('--inference_num_queries', default=0, type=int, help='inference_num_queries', required=False) #predict_frame
    parser.add_argument('--disable_resize', action='store_true',help='if resize jpeg image')
    parser.add_argument('--share_decoder', action='store_true', help='jpeg and action share decoder')
    parser.add_argument('--resize_rate',default=1 , type=int, help='resize rate for pixel prediction', required=False)
    parser.add_argument('--image_downsample_rate',default=1 , type=int, help='image_downsample_rate', required=False)
    parser.add_argument('--temporal_downsample_rate',default=1 , type=int, help='temporal_downsample_rate', required=False)
    # Model parameters external
    parser.add_argument('--test_num', default=50, type=int, help='test_num')
    parser.add_argument('--save_episode', action='store_true')
    parser.add_argument('--depth_mode', default='None', type=str, help='use depth/depth+coordinate/None. ALL/Single/None')
    parser.add_argument('--pc_mode', default='pc_camera', type=str, help='pc_world/pc_camera')
    parser.add_argument('--disable_multi_view', action='store_true', help='Use multi-view rgb images')
    # * Backbone
    parser.add_argument('--backbone', default='resnet18', type=str, # will be overridden
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list, # will be overridden
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int, # will be overridden
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int, # will be overridden
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, # will be overridden
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int, # will be overridden
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, # will be overridden
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=400, type=int, # will be overridden
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # repeat args in imitate_episodes just to avoid error. Will not be used
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--save_epoch', action='store', type=int, help='save_epoch', default=500, required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--history_step',default=0 , type=int, help='history_step', required=False)
    parser.add_argument('--predict_frame',default=0 , type=int, help='predict_frame', required=False)
    parser.add_argument('--predict_only_last', action='store_true') # only predict the last #predict_frame frame
    parser.add_argument('--temporal_agg', action='store_true')
    # visual tokenizer
    parser.add_argument('--tokenizer_model_type', default='DV', type=str, help='tokenizer_model_type, DV,CV,DI,CI')
    parser.add_argument('--tokenizer_model_temporal_rate', default=8, type=int, help='tokenizer_model_temporal_rate, 4,8')
    parser.add_argument('--tokenizer_model_spatial_rate', default=16, type=int, help='tokenizer_model_spatial_rate, 8,16')
    parser.add_argument('--tokenizer_model_name', default='Cosmos-Tokenizer-DV4x8x8', type=str, help='tokenizer_model_name')
    parser.add_argument('--prediction_weight', default=1, type=float, help='pred token Weight', required=False)
    parser.add_argument('--token_dim', default=6, type=int, help='token_dim', required=False) #token_pe_type
    parser.add_argument('--patch_size', default=5, type=int, help='patch_size', required=False) #token_pe_type
    parser.add_argument('--token_pe_type', default='learned', type=str, help='token_pe_type', required=False)
    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)

    return model, optimizer


def build_diffusion_tp_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    for k, v in args_override.items():
        setattr(args, k, v)
    # print('args',args) # get
    model = build_ACTDiffusion_tp_model(args)
    model.cuda()

    return model #, optimizer

