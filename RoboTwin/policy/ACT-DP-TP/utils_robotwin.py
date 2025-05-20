import torch
from torch.utils.data import Dataset, DataLoader
import zarr
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import math
from torch.nn.modules.batchnorm import _BatchNorm
from collections import OrderedDict
from torch.optim.lr_scheduler import LambdaLR

def normalize_data(action_data, stats, norm_type, data_type="action"):

    if norm_type == "minmax":
        action_max = torch.from_numpy(stats[data_type + "_max"]).float().to(action_data.device)
        action_min = torch.from_numpy(stats[data_type + "_min"]).float().to(action_data.device)
        action_data = (action_data - action_min) / (action_max - action_min) * 2 - 1
    elif norm_type == "gaussian":
        action_mean = torch.from_numpy(stats[data_type + "_mean"]).float().to(action_data.device)
        action_std = torch.from_numpy(stats[data_type + "_std"]).float().to(action_data.device)
        action_data = (action_data - action_mean) / action_std
    return action_data


class EpisodicDataset_Unified(torch.utils.data.Dataset):
    """
    Args:
        norm_stats: normalization stats for qpos and action
        chunksize: chunk size
        history_steps: number of history steps
        predict_frame: number of future frames to predict
    Output:
        A unified dataset for all the datasets
        image_data [0~1]: history_steps+1 Num_view C H W
        qpos_data [normalized]: history_steps+1 D
        action_data [raw]: chunk_size D
        is_pad: chunk_size
        future_imgs_data [0~1]: predict_frame Num_view C H W
        is_pad_img : predict_frame
    """

    def __init__(
        self,
        head_camera,
        state,
        action,
        episode_ends,
        stats,
        chunk_size=50,
        history_steps=0,
        predict_frame=0,
        temporal_downsample_rate=1,
        predict_only_last=False,
    ):
        super(EpisodicDataset_Unified).__init__()
        self.head_camera = head_camera
        self.state = state
        self.action = action
        self.episode_ends = episode_ends
        self.norm_stats = stats
        self.chunk_size = chunk_size
        self.history_steps = history_steps
        self.predict_frame = predict_frame
        self.temporal_downsample_rate = temporal_downsample_rate
        self.predict_only_last = predict_only_last

    def __len__(self):
        return len(self.head_camera)

    def _get_episode_idx(self, idx):
        for i, end_idx in enumerate(self.episode_ends):
            if idx < end_idx:
                start_idx = self.episode_ends[i - 1] if i > 0 else 0
                end_idx = end_idx
                return start_idx, end_idx

    def __getitem__(self, index):
        start_idx, end_idx = self._get_episode_idx(index)
        index = min(index, end_idx - 2)  # Avoid the last index of episode
        # get observation qpos and image data
        past_start_ts = max(start_idx, index - self.history_steps)
        past_padding_needed = self.history_steps - (index - past_start_ts)
        obs_qpos = self.state[past_start_ts : index + 1]
        obs_img = self.head_camera[past_start_ts : index + 1]
        if past_padding_needed > 0:
            padding_qpos = np.tile(self.state[start_idx], (past_padding_needed, 1))
            padding_image = np.tile(
                self.head_camera[start_idx], (past_padding_needed, 1, 1, 1)
            )
            obs_qpos = np.concatenate([padding_qpos, obs_qpos], axis=0)
            obs_img = np.concatenate(
                [padding_image, obs_img], axis=0
            )  # (history_steps+1, 3, H, W)
        # get action chunk data
        original_action_shape = (self.chunk_size, *self.action.shape[1:])
        gt_action = np.zeros(original_action_shape)
        action_len = min(self.chunk_size, end_idx - index - 1)
        # print(action_len)
        # print(self.action[index+1:index+1+action_len].shape)
        gt_action[:action_len] = self.action[
            index + 1 : index + 1 + action_len
        ]  # move left one step,due to action = qpos
        is_pad = np.zeros(self.chunk_size)
        is_pad[action_len:] = 1
        # get future image data
        if self.predict_frame > 0:
            future_frames_len = min(self.predict_frame, end_idx - index - 1)
            future_images = self.head_camera[
                index : index + future_frames_len + 1
            ]  # (future_frames_len+1, 3, H, W)
            is_pad_img = np.zeros(self.predict_frame + 1)
            future_padding_needed = self.predict_frame - future_frames_len

            if future_padding_needed > 0:
                is_pad_img[-future_padding_needed:] = 1
                last_frame = self.head_camera[end_idx - 1]  # (3, H, W)
                padding_future_frames = np.broadcast_to(
                    last_frame, (future_padding_needed, *last_frame.shape)
                )
                future_images = np.concatenate(
                    [future_images, padding_future_frames], axis=0
                )

            future_images = future_images[:: self.temporal_downsample_rate][
                1:
            ]  # (predict_frames, 3, H, W)
            is_pad_img = is_pad_img[:: self.temporal_downsample_rate][
                1:
            ]  # (predict_frames)

            if self.predict_only_last:
                future_images = future_images[-1:]
                is_pad_img = is_pad_img[-1:]

        # construct observations
        image_data = (
            torch.from_numpy(obs_img).unsqueeze(1).float()
        )  # (history_steps+1, 1, 3, H, W) add num_view
        qpos_data = torch.from_numpy(obs_qpos).float()
        action_data = torch.from_numpy(gt_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        if self.predict_frame > 0:
            future_imgs_data = torch.from_numpy(future_images).unsqueeze(1).float()
            is_pad_img = torch.from_numpy(is_pad_img).bool()
        else:
            future_imgs_data = 0
            is_pad_img = 0

        # normalize image and change dtype to float
        image_data = image_data / 255.0  # history_steps+1 N C H W
        future_imgs_data = (
            future_imgs_data / 255.0 if future_imgs_data is not None else None
        )
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats[
            "qpos_std"
        ]

        return image_data, qpos_data, action_data, is_pad, future_imgs_data, is_pad_img


def get_norm_stats(state, action):
    all_qpos_data = torch.from_numpy(np.array(state))
    all_action_data = torch.from_numpy(np.array(action))
    # normalize action data
    action_mean = all_action_data.mean(dim=[0], keepdim=True)
    action_std = all_action_data.std(dim=[0], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf)  # clipping
    action_max = torch.amax(all_action_data, dim=[0], keepdim=True)
    action_min = torch.amin(all_action_data, dim=[0], keepdim=True)

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf)  # clipping

    stats = {
        "action_mean": action_mean.numpy().squeeze(),
        "action_std": action_std.numpy().squeeze(),
        "action_max": action_max.numpy().squeeze(),
        "action_min": action_min.numpy().squeeze(),
        "qpos_mean": qpos_mean.numpy().squeeze(),
        "qpos_std": qpos_std.numpy().squeeze(),
    }

    return stats


def split_episode_indices(head_camera, state, action, episode_ends, train_ratio=0.98):
    num_episodes = len(episode_ends)
    shuffledm_episodes_indices = np.random.permutation(num_episodes)
    train_episode_index = shuffledm_episodes_indices[: int(train_ratio * num_episodes)]
    val_episode_index = shuffledm_episodes_indices[int(train_ratio * num_episodes) :]

    train_head_camera = []
    train_state = []
    train_action = []
    train_episode_ends = []
    train_samples = 0

    val_head_camera = []
    val_state = []
    val_action = []
    val_episode_ends = []
    val_samples = 0

    prev_end = 0
    for index, end in enumerate(episode_ends):
        sampled_head_camera = head_camera[prev_end:end]
        sampled_state = state[prev_end:end]
        sampled_action = action[prev_end:end]
        if index in train_episode_index:
            train_head_camera.append(sampled_head_camera)
            train_state.append(sampled_state)
            train_action.append(sampled_action)
            train_samples += len(sampled_action)
            train_episode_ends.append(train_samples)
        else:
            val_head_camera.append(sampled_head_camera)
            val_state.append(sampled_state)
            val_action.append(sampled_action)
            val_samples += len(sampled_action)
            val_episode_ends.append(val_samples)
        prev_end = end
    train_head_camera = np.concatenate(train_head_camera, axis=0)
    train_state = np.concatenate(train_state, axis=0)
    train_action = np.concatenate(train_action, axis=0)
    train_episode_ends = np.stack(train_episode_ends)

    val_head_camera = np.concatenate(val_head_camera, axis=0)
    val_state = np.concatenate(val_state, axis=0)
    val_action = np.concatenate(val_action, axis=0)
    val_episode_ends = np.stack(val_episode_ends)

    return (
        train_head_camera,
        train_state,
        train_action,
        train_episode_ends,
        val_head_camera,
        val_state,
        val_action,
        val_episode_ends,
    )


def load_data_unified(
    data_dir,
    task_name,
    head_camera_type,
    num_episodes=100,
    train_ratio=0.9,
    batch_size_train=32,
    batch_size_val=32,
    chunk_size=100,
    history_step=0,
    predict_frame=0,
    temporal_downsample_rate=1,
    predict_only_last=False,
    distributed=False,
):
    zarr_path = os.path.join(
        data_dir, f"{task_name}_{head_camera_type}_{num_episodes}.zarr"
    )
    print(f"Loading data from {zarr_path}")
    zarr_root = zarr.open(zarr_path, mode="r")
    head_camera = zarr_root["data/head_camera"]
    state = zarr_root["data/state"]
    action = zarr_root["data/action"]
    episode_ends = zarr_root["meta/episode_ends"]
    stats = get_norm_stats(state, action)
    # split the dataset
    (
        train_head_camera,
        train_state,
        train_action,
        train_episode_ends,
        val_head_camera,
        val_state,
        val_action,
        val_episode_ends,
    ) = split_episode_indices(head_camera, state, action, episode_ends, train_ratio)
    print(
        f"Train episodes: {len(train_episode_ends)}, Validation episodes: {len(val_episode_ends)}"
    )
    print(
        f"Train samples: {train_episode_ends[-1]}, Validation samples: {val_episode_ends[-1]}"
    )
    # create dataset
    train_dataset = EpisodicDataset_Unified(
        train_head_camera,
        train_state,
        train_action,
        train_episode_ends,
        stats,
        chunk_size,
        history_step,
        predict_frame,
        temporal_downsample_rate,
        predict_only_last,
    )
    val_dataset = EpisodicDataset_Unified(
        val_head_camera,
        val_state,
        val_action,
        val_episode_ends,
        stats,
        chunk_size,
        history_step,
        predict_frame,
        temporal_downsample_rate,
        predict_only_last,
    )

    if distributed:
        print("Using distributed sampler-----------------------------")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=(train_sampler is None),
        pin_memory=True,
        num_workers=4,
        sampler=train_sampler,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        sampler=val_sampler,
    )

    return train_dataloader, val_dataloader, train_sampler, stats


### helper functions
def convert_weigt(obj):
    newmodel = OrderedDict()
    for k, v in obj.items():
        if k.startswith("module."):
            newmodel[k[7:]] = v
        else:
            newmodel[k] = v
    return newmodel


def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, last_epoch=-1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer ([~torch.optim.Optimizer]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (int):
            The number of steps for the warmup phase.
        num_training_steps (int):
            The total number of training steps.
        num_cycles (float, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (int, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        torch.optim.lr_scheduler.LambdaLR with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(
            0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule(optimizer, last_epoch: int = -1) -> LambdaLR:
    """
    Create a schedule with a constant learning rate, using the learning rate set in optimizer.

    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    return LambdaLR(optimizer, lambda _: 1, last_epoch=last_epoch)


def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)
import random


def set_seed(seed):
    random.seed(seed)  #
    np.random.seed(seed)  #
    torch.manual_seed(seed)  #
    torch.cuda.manual_seed(seed)  #
    torch.cuda.manual_seed_all(seed)  #


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


def create_multiview_video(image_list, save_path, fps=30):
    """
    Convert a list of multi-view images into a video and save it to the specified path.

    Args:
        image_list (list): A list containing T elements, where each element is a NumPy array of shape (k, c, h, w),
                           representing k views of RGB images.
        save_path (str): The path to save the video file, the filename should end with '.mp4' or other video formats.
        fps (int): Frames per second of the video, default is 30.
    """
    T = len(image_list)  # Number of frames
    k, c, h, w = image_list[
        0
    ].shape  # Get the shape of each element, k is the number of views, c is the channels, h and w are height and width

    # Ensure the input data format
    assert c == 3, "Each image must be RGB (3 channels)"

    # Output resolution, width is the sum of the widths of k views, height remains unchanged
    output_width = k * w
    output_height = h

    # Create video writer using OpenCV, specify the save path, codec (e.g., MP4V), fps and resolution
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use mp4 codec
    video_writer = cv2.VideoWriter(
        save_path, fourcc, fps, (output_width, output_height)
    )

    # Iterate over each frame
    for t in range(T):
        # Get all views for the t-th frame
        frame = image_list[t]  # shape: (k, c, h, w)

        # Horizontally concatenate k views
        views = []
        for i in range(k):
            # Extract the i-th view, ensure it's in (h, w, c) format
            img = (
                frame[i].transpose(1, 2, 0) * 255
            )  # Convert from (c, h, w) to (h, w, c)
            views.append(img)

        # Horizontally concatenate the images of k views, resulting in (h, k*w, 3)
        combined_image = np.concatenate(views, axis=1)

        # Ensure image data type is uint8 and pixel values are between 0 and 255
        combined_image = np.clip(combined_image, 0, 255).astype(np.uint8)

        # Convert RGB image to BGR (OpenCV uses BGR format)
        combined_image = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

        # Write the concatenated image to the video
        video_writer.write(combined_image)

    # Release the video writer
    video_writer.release()



