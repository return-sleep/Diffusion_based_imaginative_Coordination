import os
import h5py
import math
import torch
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops import rearrange
from collections import OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.optim.lr_scheduler import LambdaLR
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import IPython
e = IPython.embed

def get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, num_cycles= 0.5, last_epoch = -1
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
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
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


class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, samples_per_epoch=1):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.samples_per_epoch = samples_per_epoch
        self.is_sim = None
        
    def __len__(self):
        return len(self.episode_ids) * self.samples_per_epoch

    def __getitem__(self, index):
        sample_full_episode = False # hardcode
        index = index % len(self.episode_ids)
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        # action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad
  
  
class EpisodicDataset_Unified_Preload(torch.utils.data.Dataset):
    """
    Args:
        episode_ids: list of episode ids
        dataset_dir: path to the dataset
        camera_names: list of camera names
        norm_stats: normalization stats for qpos and action
        chunksize: chunk size
        history_steps: number of history steps
        predict_frame: number of future frames to predict
        samples_per_epoch: number of samples per epoch
        action_mode: 'endpose' or 'joint_action'
        mode: 'Train' or 'Val'
    Output:
        A unified dataset for all the datasets
        image_data [0~1]: history_steps+1 Num_view C H W
        qpos_data [normalized]: history_steps+1 D
        action_data [raw]: chunk_size D
        is_pad: chunk_size
        future_imgs_data [0~1]: predict_frame Num_view C H W
        is_pad_img : predict_frame
    """
    def __init__(self, image_data,qpos_data,action_data, camera_names, norm_stats, chunksize = 50 , history_steps=0, predict_frame= 0,temporal_downsample_rate=1, predict_only_last=False,samples_per_epoch=1,action_mode='endpose'):
        super(EpisodicDataset).__init__()
        self.num_episodes = len(image_data)
        self.image_data = image_data # N Num_view T+H  H W C
        self.qpos_data = qpos_data # N T H W C
        self.action_data = action_data # N T H W C
        
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.samples_per_epoch = samples_per_epoch
        self.is_sim = None
        self.chunksize = chunksize
        self.history_steps = history_steps
        self.predict_frame = predict_frame  
        self.temporal_downsample_rate = temporal_downsample_rate
        self.predict_only_last = predict_only_last
        self.action_mode = action_mode
        
        

    def __len__(self):
        return self.num_episodes * self.samples_per_epoch

    def __getitem__(self, index):
        sample_full_episode = False # hardcode
        episode_id = index % self.num_episodes
        
        original_action_shape = self.action_data[episode_id].shape
        episode_len = original_action_shape[0]
        if sample_full_episode:
            start_ts = 0
        else:
            start_ts = np.random.choice(episode_len) 

        # get observation with history_steps+1 
        past_start_ts = max(0, start_ts - self.history_steps)
        past_padding_needed = self.history_steps - (start_ts - past_start_ts) 
        qpos = self.qpos_data[episode_id][past_start_ts:start_ts+1]
        all_cam_images = []
        for cam_name_id in np.arange(len(self.camera_names)):
            all_cam_images.append(self.image_data[episode_id,cam_name_id][past_start_ts:start_ts+1]) # T' H W C    
        all_cam_images = np.stack(all_cam_images, axis=1) # N+1 4 H W C
        
        if past_padding_needed > 0: 
            padding_qpos = np.tile(self.qpos_data[episode_id,0], (past_padding_needed, 1))
            qpos = np.vstack((padding_qpos, qpos)) # B N+1 D 
            padding_all_cam_images = []
            for cam_name_id in np.arange(len(self.camera_names)):
                padding_all_cam_images.append(self.image_data[episode_id,cam_name_id][0:1]) # 1 H W C
            padding_all_cam_images = np.stack(padding_all_cam_images, axis=1) # 1 4 H W C
            padding_all_cam_images = np.tile(padding_all_cam_images, (past_padding_needed, 1, 1, 1, 1))
            all_cam_images = np.vstack((padding_all_cam_images,all_cam_images)) # N+2 4 H W C
            
        # get all actions after and including start_ts
        action = self.action_data[episode_id][start_ts:]
        action_len = episode_len - start_ts
        
        if self.predict_frame > 0:
            original_image_pad = torch.zeros(episode_len + self.chunksize).bool() 
            original_image_pad[-self.chunksize:] = 1
            future_imgs = []
            if self.predict_only_last == False:
                for cam_name_id in np.arange(len(self.camera_names)):
                    future_imgs.append(self.image_data[episode_id,cam_name_id][start_ts:start_ts+self.predict_frame+1:self.temporal_downsample_rate][1:]) # T' H W C
                future_imgs = np.stack(future_imgs, axis=1) # T' Num_view H W C
                is_pad_img = original_image_pad[start_ts:start_ts+self.predict_frame+1:self.temporal_downsample_rate][1:] # T'
            else: # just predict the corresponding result frame
                for cam_name_id in np.arange(len(self.camera_names)):
                    future_imgs.append(self.image_data[episode_id,cam_name_id][start_ts+self.predict_frame:start_ts+self.predict_frame+1]) # 1 H W C
                future_imgs = np.stack(future_imgs, axis=1) # 1 Num_view H W C  
                is_pad_img = original_image_pad[start_ts+self.predict_frame:start_ts+self.predict_frame+1] # 1
            
        padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(episode_len)
        is_pad[action_len:] = 1

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        image_data = torch.einsum('n k h w c -> n k c h w', image_data) # N K C H W
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float() # padding zero actions
        is_pad = torch.from_numpy(is_pad).bool()
        action_data = action_data[:self.chunksize]
        is_pad = is_pad[:self.chunksize]
        
        # construct future frames
        if self.predict_frame > 0:
            future_imgs = torch.from_numpy(future_imgs)
            future_imgs_data = torch.einsum('n k h w c -> n k c h w', future_imgs)
        else:
            future_imgs_data = None
            is_pad_img = None

        # normalize image and change dtype to float
        image_data = image_data / 255.0 # history_steps+1 N C H W
        future_imgs_data = future_imgs_data / 255.0 if future_imgs_data is not None else None
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        return image_data, qpos_data, action_data, is_pad, future_imgs_data, is_pad_img
    

def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
        all_qpos_data.append(torch.from_numpy(qpos))
        all_action_data.append(torch.from_numpy(action).float())

    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data) # N T DS
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping
    action_max = torch.amax(all_action_data, dim=[0, 1], keepdim=True)
    action_min = torch.amin(all_action_data, dim=[0, 1], keepdim=True)
    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping


    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(), "action_max": action_max.numpy().squeeze(), "action_min": action_min.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats

def preload_data(dataset_dir, num_episodes,camera_names, predict_frame=0,image_downsample_rate=1):
    all_qpos_data = []
    all_action_data = []
    all_image_data = [] 
    for episode_idx in tqdm(range(num_episodes), desc="Processing Episodes"):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        print('Loading dataset:', dataset_path)
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
            image_data = []
            for cam_name in camera_names:
                image = root[f'/observations/images/{cam_name}'][()] # T H W C 
                image_data.append(image[:,::image_downsample_rate,::image_downsample_rate,:])
            image_data = np.stack(image_data, axis=0) # Num_view T H W C
        all_qpos_data.append(qpos)
        all_action_data.append(action)
        all_image_data.append(image_data)
    all_qpos_data =  np.stack(all_qpos_data) # N T D
    all_action_data =  np.stack(all_action_data) # N T D
    all_image_data =  np.stack(all_image_data) # N Num_view T H W C
    if predict_frame > 0:
        last_frame = all_image_data[:, :, -1:, :, :, :]  
        # padding_frames = np.tile(last_frame, (1, 1, predict_frame, 1, 1, 1))  
        padding_frames = np.broadcast_to(last_frame, (last_frame.shape[0], last_frame.shape[1], predict_frame, *last_frame.shape[3:]))
        all_image_data = np.concatenate([all_image_data, padding_frames], axis=2)
    return all_image_data, all_qpos_data, all_action_data

def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val,samples_per_epoch, distributed=False):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats, samples_per_epoch = samples_per_epoch)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats, samples_per_epoch = samples_per_epoch)
    print('Train dataset size:', len(train_dataset))
    if distributed:
        print('Using distributed sampler-----------------------------')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None
        
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=(train_sampler is None), pin_memory=True, num_workers=8, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=8)

    return train_dataloader, val_dataloader, train_sampler, norm_stats, train_dataset.is_sim

def load_data_unified_preload(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val, chunksize= 100, history_step = 0, predict_frame=100, temporal_downsample_rate=1, image_downsample_rate= 1, predict_only_last=False, samples_per_epoch = 1 ,distributed = False):
    print(f'Data from: {dataset_dir}, num_episodes: {num_episodes}, samples_per_epoch: {samples_per_epoch}')
    print(f'chunk_size: {chunksize}, history_step: {history_step}, predict_frame: {predict_frame}')
    print(f'temporal_downsample_rate: {temporal_downsample_rate}, image_downsample_rate: {image_downsample_rate}, predict_only_last: {predict_only_last}')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]
    print('Train indices num:', len(train_indices))
    print('Val indices num:', len(val_indices))
    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)
    print('Preloading data...')
    all_image_data, all_qpos_data, all_action_data = preload_data(dataset_dir, num_episodes, camera_names, predict_frame,image_downsample_rate)
    print('Preloading done.')
    
    # construct dataset and dataloader
    train_dataset = EpisodicDataset_Unified_Preload(all_image_data[train_indices],all_qpos_data[train_indices],all_action_data[train_indices],camera_names, norm_stats, chunksize, history_step, predict_frame,temporal_downsample_rate=temporal_downsample_rate, predict_only_last=predict_only_last,samples_per_epoch=samples_per_epoch)
    val_dataset = EpisodicDataset_Unified_Preload(all_image_data[val_indices],all_qpos_data[val_indices],all_action_data[val_indices], camera_names, norm_stats, chunksize, history_step, predict_frame,temporal_downsample_rate=temporal_downsample_rate ,predict_only_last=predict_only_last, samples_per_epoch=samples_per_epoch)

    if distributed:
        print('Using distributed sampler-----------------------------')
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=(train_sampler is None), pin_memory=True, num_workers=4, sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False, pin_memory=True, num_workers=4, sampler=val_sampler)
    return train_dataloader, val_dataloader, train_sampler, norm_stats, train_dataset.is_sim

def collate_fn(batch):
    return (default_collate([b[:-1] for b in batch]),
            [b[-1] for b in batch])

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

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

def set_seed(seed):
    random.seed(seed)  # 
    np.random.seed(seed)  # 
    torch.manual_seed(seed)  # 
    torch.cuda.manual_seed(seed)  # 
    torch.cuda.manual_seed_all(seed)  # 

def get_image(ts, camera_names):
    """_summary_
    Returns:
        curr_image [0-1] torch.Tensor: (1, num_cams, C, H, W)
    """
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image

def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()
    print(f'Saved plots to {ckpt_dir}')
    
def convert_weigt(obj):
    newmodel = OrderedDict()
    for k, v in obj.items():
        if k.startswith('module.'):
            newmodel[k[7:]] = v
        else:
            newmodel[k] = v
    return newmodel 
