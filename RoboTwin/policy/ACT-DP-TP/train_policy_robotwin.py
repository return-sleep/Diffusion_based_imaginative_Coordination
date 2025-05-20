import torch
import numpy as np
import os
import json
import pickle
import argparse
from copy import deepcopy
from tqdm import tqdm
import torch.multiprocessing as mp
import builtins
import torch.distributed as dist
import wandb
from utils_robotwin import load_data_unified,normalize_data
from utils_robotwin import compute_dict_mean, set_seed, detach_dict
from utils_robotwin import plot_history
from utils_robotwin import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule,
    convert_weigt,
)
from policy import *
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import IPython

e = IPython.embed

# Global variables
current_dir = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(current_dir, "data")
CAMERA_NAMES = ["head_camera"]


def main_worker(args):
    set_seed(args.seed)
    # command line parameters
    args = vars(args)
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]
    num_epochs = args["num_epochs"]
   
    # get task parameters
    dataset_dir = DATA_DIR
    camera_names = CAMERA_NAMES
    head_camera_type = args["head_camera_type"]
    num_episodes = args["num_episodes"]
    disable_vae_latent = args["disable_vae_latent"]
    num_inference_steps = args["num_inference_steps"]
    num_train_steps = args["num_train_steps"]
    #######################  fixed parameters #######################
    state_dim = 14
    lr_backbone = 1e-5
    backbone = args["backbone"]
    print("backbone:", backbone)
    if "diffusion" in policy_class or "ACT" in policy_class:
        enc_layers = 4  # TODO scale model
        dec_layers = 7  # TODO scale model
        nheads = 8  # TODO scale model
        policy_config = {
            "lr": args["lr"],
            "task_name": task_name,
            "is_wandb": args["is_wandb"],
            "save_epoch": args["save_epoch"],
            "num_queries": args["chunk_size"],
            "predict_frame": args["predict_frame"],
            "history_step": args["history_step"],
            "resize_rate": args["resize_rate"],
            "image_downsample_rate": args["image_downsample_rate"],
            "temporal_downsample_rate": args["temporal_downsample_rate"],
            "image_height": 240,  # Hard code
            "image_width": 320,  # Hard code
            "kl_weight": args["kl_weight"],
            "hidden_dim": args["hidden_dim"],
            "dim_feedforward": args["dim_feedforward"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "enc_layers": enc_layers,
            "dec_layers": dec_layers,
            "nheads": nheads,
            "camera_names": camera_names,
            "norm_type": args["norm_type"],
            "disable_vae_latent": disable_vae_latent,
            "disable_resnet": args["disable_resnet"],
            # design diffusion parameters
            "num_inference_steps": num_inference_steps,
            "num_train_timesteps": num_train_steps,
            "prediction_type": args["prediction_type"],
            "loss_type": args["loss_type"],
            "schedule_type": args["schedule_type"],
            "beta_schedule": args["beta_schedule"],
            "diffusion_timestep_type": args["diffusion_timestep_type"],
            "attention_type": args["attention_type"],
            "causal_mask": args["causal_mask"],
            "predict_only_last": args["predict_only_last"],
            "share_decoder": args["share_decoder"],
            # visual toknizer
            "tokenizer_model_temporal_rate": args["tokenizer_model_temporal_rate"],
            "tokenizer_model_spatial_rate": args["tokenizer_model_spatial_rate"],
            "tokenizer_model_name": args["tokenizer_model_name"],
            "prediction_weight": args["prediction_weight"],
            "imitate_weight": args["imitate_weight"],
            "token_dim": args["token_dim"],
            "patch_size": args["patch_size"],
            "token_pe_type": args["token_pe_type"],
            'use_film': False,
            'multitask': False,
        }
    elif policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "num_queries": 1,
            "camera_names": camera_names,
        }
    else:
        raise NotImplementedError

    config = {
        "num_epochs": num_epochs,
        "ckpt_dir": ckpt_dir,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "policy_config": policy_config,
        "task_name": task_name,
        "seed": args["seed"],
        "camera_names": camera_names,
        "batch_size_train": batch_size_train,
        "lr_backbone": lr_backbone,
        "norm_type": args["norm_type"],  # FOR action normalization
        "lr_scheduler": args["lr_schedule_type"],
        "share_decoder": args["share_decoder"],
        "image_downsample_rate": args["image_downsample_rate"],
        "temporal_downsample_rate": args["temporal_downsample_rate"],
        "patch_size": args["patch_size"],
        "diffusion_timestep_type": args["diffusion_timestep_type"],
        "token_pe_type": args["token_pe_type"],
        "attention_type": args["attention_type"],
    }

    #######################  load data ############################
    print("####################### Step 1: get dataloader #######################")
    distributed = False
    train_dataloader, val_dataloader, train_sampler, stats = load_data_unified(
        dataset_dir,
        task_name,
        head_camera_type,
        num_episodes,
        args["train_ratio"],
        batch_size_train,
        batch_size_val,
        args["chunk_size"],
        args["history_step"],
        args["predict_frame"],
        args["temporal_downsample_rate"],
        args["predict_only_last"],
        distributed,
    )

    #######################  Train Phase ############################
    os.makedirs(ckpt_dir, exist_ok=True)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)
    best_ckpt_info = train_bc(
        train_dataloader, val_dataloader, config, train_sampler, stats
    )

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint no need for scripted data, all in
    ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
    torch.save(best_state_dict, ckpt_path)
    log_message = f"Best ckpt, val loss {min_val_loss:.6f} @ epoch {best_epoch}"
    print(log_message)

    log_dir = os.path.join(ckpt_dir, "training_log.txt")
    with open(log_dir, "a") as file:
        file.write(log_message + "\n")


def train_bc(train_dataloader, val_dataloader, config, train_sampler=None, stats=None):
    num_epochs = config["num_epochs"]
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]
    downsample_rate = config["image_downsample_rate"]  # downsample rate for image space
    print("ckpt_dir:", ckpt_dir)

    print(
        "####################### Step 2: get policy",
        policy_class,
        "#######################",
    )
    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    
    # make optimizer
    param_dicts = [
        {
            "params": [
                p
                for n, p in policy.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in policy.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": policy_config["lr_backbone"],
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=policy_config["lr"], weight_decay=1e-4
    )
    if config["lr_scheduler"] == "constant":
        scheduler = get_constant_schedule(optimizer)
    elif config["lr_scheduler"] == "cosine_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=500,
            num_training_steps=(num_epochs + 1) * len(train_dataloader),
        )

    # load checkpoint
    start_epoch = 0
    # start training
    print("####################### Step 3: start training #######################")
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    update_step = 0

    for epoch in tqdm(range(start_epoch, num_epochs + 1)):
        print(f"\nEpoch {epoch}")
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(
                    policy_config,
                    data,
                    policy,
                    stats,
                    is_training=False,
                    downsample_rate=downsample_rate,
                )
                epoch_dicts.append(forward_dict)

            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
                torch.save(policy.state_dict(), ckpt_path)
                   

           
            print(f"Val loss:   {epoch_val_loss:.5f}")
            summary_string = ""
            for k, v in epoch_summary.items():
                summary_string += f"{k}: {v.item():.3f} "
            print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()

        for batch_idx, data in enumerate(
            tqdm(
                train_dataloader,
                desc=f"Train Epoch {epoch+1}/{num_epochs+1}",
                leave=False,
            )
        ):
            forward_dict = forward_pass(
                policy_config,
                data,
                policy,
                stats,
                is_training=True,
                downsample_rate=downsample_rate,
            )
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
            update_step += 1
            scheduler.step()
            
        epoch_summary = compute_dict_mean(
            train_history[
                (batch_idx + 1)
                * (epoch - start_epoch) : (batch_idx + 1)
                * (epoch + 1 - start_epoch)
            ]
        )
        epoch_train_loss = epoch_summary["loss"]
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)
        print("lr:", optimizer.param_groups[0]["lr"])

        if epoch % config["policy_config"]["save_epoch"] == 0 and epoch > 0:
            ckpt_path = os.path.join(
                ckpt_dir, f"policy_epoch_{epoch}_seed_{seed}.ckpt"
            )
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

        lastest_ckpt = {
            "epoch": epoch,
            "state_dict": policy.state_dict(),
            "optimizer": optimizer.state_dict(),
            "stats": stats,
        }
        lastest_ckpt_path = os.path.join(
            ckpt_dir, f"policy_lastest_seed_{seed}.ckpt"
        )  # avoid interrupt
        torch.save(lastest_ckpt, lastest_ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)
    ckpt_path = os.path.join(
        ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt"
    )
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )
    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)   
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    elif policy_class == "ACT_diffusion_tp":
        policy = ACTPolicyDiffusion_with_Token_Prediction(policy_config)
    else:
        raise NotImplementedError
    print(f"Policy: {policy_class}")
    return policy


def forward_pass(config, data, policy, stats=None, is_training=True, downsample_rate=1):
    image_data, qpos_data, action_data, is_pad, future_imgs_data, is_pad_img = (
        data  # raw action
    )

    image_data, qpos_data, action_data, is_pad, future_imgs_data, is_pad_img = (
        image_data.cuda(),
        qpos_data.cuda(),
        action_data.cuda(),
        is_pad.cuda(),
        future_imgs_data.cuda(),
        is_pad_img.cuda(),
    )
    action_data = normalize_data(
        action_data, stats, config["norm_type"], data_type="action"
    )

    if isinstance(policy, (ACTPolicy, CNNMLPPolicy)) or isinstance(
        getattr(policy, "module", None), (ACTPolicy, CNNMLPPolicy)
    ):  # for mutli-gpu
        image_data = image_data[:, -1]  # B, N, C, H, W no history TODO
        qpos_data = qpos_data[:, -1]  # B, N , C, H, W
        return policy(qpos_data, image_data, action_data, is_pad, is_training)  #
    else:
        return policy(
            qpos_data,
            image_data,
            action_data,
            is_pad,
            future_imgs_data,
            is_pad_img,
            is_training,  # aug or not
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_wandb", action="store_true")
    parser.add_argument(
        "--ckpt_dir", default="checkpoints", type=str, help="ckpt_dir", required=False
    )
    parser.add_argument(
        "--policy_class",
        default="ACT",
        type=str,
        help="policy_class, capitalize",
        required=False,
    )
    parser.add_argument(
        "--task_name",
        default="bottle_adjust",
        type=str,
        help="task_name",
        required=False,
    )
    parser.add_argument(
        "--head_camera_type",
        default="D435",
        type=str,
        help="head_camera_type",
        required=False,
    )
    parser.add_argument(
        "--num_episodes", default=100, type=int, help="num_epochs", required=False
    )
    parser.add_argument(
        "--train_ratio", default=0.9, type=float, help="train_ratio", required=False
    )
    parser.add_argument(
        "--batch_size", default=64, type=int, help="batch_size", required=False
    )
    parser.add_argument("--seed", default=0, type=int, help="seed", required=False)
    parser.add_argument(
        "--num_epochs", default=100, type=int, help="num_epochs", required=False
    )
    parser.add_argument("--lr", default=5e-5, type=float, help="lr", required=False)
    parser.add_argument(
        "--lr_schedule_type", default="constant", type=str, help="lr_schedule_type"
    )
    parser.add_argument("--backbone", default="resnet18", type=str, help="backbone")
    # for ACT
    parser.add_argument(
        "--kl_weight", action="store", type=int, help="KL Weight", required=False
    )
    parser.add_argument(
        "--save_epoch",
        action="store",
        type=int,
        help="save_epoch",
        default=200,
        required=False,
    )
    parser.add_argument(
        "--chunk_size", action="store", type=int, help="chunk_size", required=False
    )
    parser.add_argument(
        "--history_step", default=0, type=int, help="history_step", required=False
    )
    parser.add_argument(
        "--predict_frame", default=0, type=int, help="predict_frame", required=False
    )
    parser.add_argument(
        "--image_downsample_rate",
        default=1,
        type=int,
        help="image_downsample_rate",
        required=False,
    )
    parser.add_argument(
        "--resize_rate",
        default=1,
        type=int,
        help="resize_rate for future image prediction",
        required=False,
    )
    parser.add_argument(
        "--temporal_downsample_rate",
        default=1,
        type=int,
        help="temporal_downsample_rate",
        required=False,
    )
    parser.add_argument(
        "--predict_only_last", action="store_true"
    )  # only predict the last #predict_frame frame
    parser.add_argument(
        "--hidden_dim",
        action="store",
        default=512,
        type=int,
        help="hidden_dim",
        required=False,
    )
    parser.add_argument(
        "--dim_feedforward",
        action="store",
        default=3200,
        type=int,
        help="dim_feedforward",
        required=False,
    )

    # prediction_type for diffusion
    parser.add_argument("--norm_type", default="minmax", type=str, help="norm_type")
    parser.add_argument(
        "--num_train_steps", default=100, type=int, help="num_train_steps"
    )
    parser.add_argument(
        "--num_inference_steps", default=10, type=int, help="num_inference_steps"
    )
    parser.add_argument(
        "--imitate_weight", default=1, type=int, help="imitate Weight", required=False
    )
    parser.add_argument(
        "--schedule_type", default="DDIM", type=str, help="scheduler_type"
    )
    parser.add_argument(
        "--prediction_type", default="sample", type=str, help="prediction_type"
    )
    parser.add_argument(
        "--beta_schedule", default="squaredcos_cap_v2", type=str, help="prediction_type"
    )
    parser.add_argument("--loss_type", default="l1", type=str, help="loss_type")
    parser.add_argument(
        "--diffusion_timestep_type",
        default="cat",
        type=str,
        help="diffusion_timestep_type, cat or add, how to combine timestep",
    )
    parser.add_argument("--attention_type", default="v0", help="decoder attention type")
    parser.add_argument(
        "--causal_mask", action="store_true", help="use causal mask for diffusion"
    )
    parser.add_argument(
        "--disable_vae_latent",
        action="store_true",
        help="Use VAE latent space by default",
    )
    parser.add_argument(
        "--disable_resnet",
        action="store_true",
        help="Use resnet to encode obs image  by default",
    )
    parser.add_argument(
        "--share_decoder", action="store_true", help="jpeg and action share decoder"
    )

    # visual tokenizer
    parser.add_argument(
        "--tokenizer_model_type",
        default="DV",
        type=str,
        help="tokenizer_model_type, DV,CV,DI,CI",
    )
    parser.add_argument(
        "--tokenizer_model_temporal_rate",
        default=8,
        type=int,
        help="tokenizer_model_temporal_rate, 4,8",
    )
    parser.add_argument(
        "--tokenizer_model_spatial_rate",
        default=16,
        type=int,
        help="tokenizer_model_spatial_rate, 8,16",
    )
    parser.add_argument(
        "--prediction_weight",
        default=1,
        type=float,
        help="pred token Weight",
        required=False,
    )
    parser.add_argument(
        "--patch_size", default=5, type=int, help="patch_size", required=False
    )
    parser.add_argument(
        "--token_pe_type",
        default="learned",
        type=str,
        help="token_pe_type",
        required=False,
    )
    # for next frame or segmentation setting
    args = parser.parse_args()
    if args.tokenizer_model_type in ["DV", "CV"]:
        args.tokenizer_model_name = f"Cosmos-Tokenizer-{args.tokenizer_model_type}{args.tokenizer_model_temporal_rate}x{args.tokenizer_model_spatial_rate}x{args.tokenizer_model_spatial_rate}"
    elif args.tokenizer_model_type in ["DI", "CI"]:
        args.tokenizer_model_name = f"Cosmos-Tokenizer-{args.tokenizer_model_type}{args.tokenizer_model_spatial_rate}x{args.tokenizer_model_spatial_rate}"
        args.tokenizer_model_temporal_rate = 1
    else:
        raise NotImplementedError

    if args.tokenizer_model_type in ["DV", "DI"]:
        args.token_dim = 6
    else:
        args.token_dim = 16

    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(os.path.join(args.ckpt_dir, "train_args_config.json"), "w") as json_file:
        json.dump(vars(args), json_file, indent=4)

    main_worker(args)
