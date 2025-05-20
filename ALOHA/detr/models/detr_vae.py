# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import *
from einops import rearrange
import numpy as np

import IPython
e = IPython.embed

def get_spatial_temporal_positional_encoding(height, width, seq_len, dim):
    """
    Generate spatial-temporal positional encoding for video data.
    
    Args:
        height (int): Height of the video frames.
        width (int): Width of the video frames.
        seq_len (int): Number of frames in the video.
        dim (int): Embedding dimension.
        device (str): Device to store the tensor.
    
    Returns:
        torch.Tensor: Positional encoding with shape (seq_len, height, width, dim).
    """
    assert dim % 2 == 0, "Embedding dimension (dim) must be even."

    spatial_dim = dim // 2
    temporal_dim = dim - spatial_dim

    # Temporal encoding
    temporal_encoding = get_sinusoid_encoding_table(seq_len, temporal_dim)[0]  # Shape: (seq_len, temporal_dim)

    # Spatial encoding
    position_h = torch.arange(height, dtype=torch.float32).unsqueeze(1)  # (height, 1)
    position_w = torch.arange(width, dtype=torch.float32).unsqueeze(1)  # (1, width)
    div_term_h = torch.exp(-torch.arange(0, spatial_dim, 2, dtype=torch.float32) * (math.log(10000.0) / spatial_dim)).unsqueeze(0)
    div_term_w = torch.exp(-torch.arange(0, spatial_dim, 2, dtype=torch.float32) * (math.log(10000.0) / spatial_dim)).unsqueeze(0)
    
    spatial_encoding_h = torch.sin(position_h * div_term_h)  # (height, spatial_dim // 2)
    spatial_encoding_w = torch.cos(position_w * div_term_w)  # (width, spatial_dim // 2)
    
    # print(spatial_encoding_h.shape, spatial_encoding_w.shape)   
    # Combine H and W spatial encodings
    spatial_encoding_h = spatial_encoding_h.unsqueeze(1).expand(-1, width, -1)  # (height,  width£¬ spatial_dim // 2)
    spatial_encoding_w = spatial_encoding_w.unsqueeze(0).expand(height, -1, -1)  # (height, width , spatial_dim // 2)
    spatial_encoding = torch.cat([spatial_encoding_h, spatial_encoding_w], dim=-1)  # (height, width , spatial_dim)
    spatial_encoding = spatial_encoding.unsqueeze(0).repeat(seq_len, 1, 1, 1)  # (seq_len, height, width , spatial_dim)

    # Combine spatial and temporal
    temporal_encoding = temporal_encoding.unsqueeze(1).unsqueeze(1).repeat(1, height, width, 1)  # (seq_len, height, width, temporal_dim)
    # print(spatial_encoding.shape, temporal_encoding.shape)
    pos_encoding = torch.cat([spatial_encoding, temporal_encoding], dim=-1)  # Combine spatial and temporal

    return pos_encoding # (seq_len, height, width, dim)


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    # print(sinusoid_table.shape)
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, state_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(14, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state, actions=None, is_pad=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            # Image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            for cam_id, cam_name in enumerate(self.camera_names):
                features, pos = self.backbones[0](image[:, cam_id]) # HARDCODED
                features = features[0] # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]
        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1) # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]
        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]


class DETRVAE_Denoise_Token_Prediction(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, num_queries, camera_names, history_step, predict_frame, image_downsample_rate, temporal_downsample_rate,  disable_vae_latent, disable_resnet, patch_size = 5, token_pe_type = 'learned', image_height = 480, image_width = 640, tokenizer_model_temporal_rate= 8, tokenizer_model_spatial_rate = 16, resize_rate = 1):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py 
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.transformer_share_decoder= transformer.share_decoder
        self.predict_only_last = transformer.predict_only_last
        self.encoder = encoder
        self.disable_vae_latent = disable_vae_latent
        self.disable_resnet = disable_resnet
        hidden_dim = transformer.d_model # 512
        self.hidden_dim = hidden_dim
        token_dim= transformer.token_dim
        # Action head state_dim = action_dim
        self.action_head = nn.Linear(hidden_dim, state_dim) # TODO replace MLP?
        print('share decoder', self.transformer_share_decoder)
        if self.transformer_share_decoder == False:
            self.token_head = nn.Linear(hidden_dim, token_dim*patch_size*patch_size) # HardCode TODO replace MLP
        else:
            self.token_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), # Hardcode patch size * path size * patch dim
                nn.SiLU(),
                nn.Linear(hidden_dim, token_dim*patch_size*patch_size))
        
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim) # for decoder as PE
        self.diffusion_timestep_type =  transformer.diffusion_timestep_type 
        if backbones is not None and disable_resnet == False:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(14, hidden_dim) # proprioception
        elif backbones is not None and disable_resnet == True:
            self.pos_encoder = backbones[0][1] # for 2D PositionEmbedding 
            self.backbones = None # HardCDcode
            # Hardcode divide the latent feature representation into non-overlapping patches 
            self.input_proj_token = nn.Conv2d(token_dim, hidden_dim, kernel_size=5, stride=5, bias=False) # Hardcode deal with image token pa
            self.input_proj_robot_state = nn.Linear(14, hidden_dim) # proprioception
        else:
            # input_dim = 14 + 7 # robot_state + env_state
            self.input_proj_robot_state = nn.Linear(14, hidden_dim)
            self.input_proj_env_state = nn.Linear(7, hidden_dim)
            self.pos = torch.nn.Embedding(2, hidden_dim)
            self.backbones = None

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(14, hidden_dim) # project action to embedding TODO action dim = 14/16
        self.encoder_joint_proj = nn.Linear(14, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+history_step+ num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters vae latent to decoder token space
        self.history_step = history_step
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2+history_step, hidden_dim) # learned position embedding for proprio and latent and denpoe
        self.denoise_step_pos_embed = nn.Embedding(1, hidden_dim) # learned position embedding for denoise step
        # setting for token prediction
        if self.predict_only_last:
            self.num_temporal_token = 1
        else:
            # print('predict frame', predict_frame, 'temporal_downsample_rate', temporal_downsample_rate, 'tokenizer_model_temporal_rate', tokenizer_model_temporal_rate)
            self.num_temporal_token = math.ceil(predict_frame // temporal_downsample_rate / tokenizer_model_temporal_rate)
        self.patch_size = patch_size
        self.image_h = image_height // image_downsample_rate // tokenizer_model_spatial_rate // resize_rate // self.patch_size #
        self.image_w = image_width // image_downsample_rate // tokenizer_model_spatial_rate // resize_rate // self.patch_size # 
        self.num_pred_token_per_timestep = self.image_h * self.image_w * len(camera_names)
        self.token_shape = (self.num_temporal_token, self.image_h, self.image_w, self.patch_size)
        if token_pe_type == 'learned':
            self.query_embed_token = nn.Embedding(self.num_temporal_token * self.num_pred_token_per_timestep, hidden_dim) # for decoder as PE TODO replace with temporal spatial PE
        print('predict token shape', ( self.num_pred_token_per_timestep, self.num_temporal_token, self.image_h, self.image_w, self.patch_size))
        self.query_embed_token_fixed = get_spatial_temporal_positional_encoding(self.image_h, self.image_w, self.num_temporal_token, hidden_dim).view(-1,hidden_dim) # for decoder as PE  # (seq_len, height, width, dim) -> (seq_len*height*width, dim)
        self.token_pe_type = token_pe_type
        # print('predict token shape', ( self.num_pred_token_per_timestep, self.num_temporal_token, self.image_h, self.image_w, self.patch_size))
    
    def forward(self, qpos, current_image, env_state, actions, is_pad, noisy_actions, noise_tokens, is_tokens_pad, denoise_steps):
        # qpos: batch, 1+ history, qpos_dim
        # current_image: 1. batch, T', num_cam, 3, height, width - image_norm T' = 1+ history
        #                2. batch, T', num_cam*6/16, height', width' - image_token
        # env_state: None
        # actions: batch, seq, action_dim clean action for vae encoder
        # is_pad: batch, seq, for vae encoder
        # noisy_actions: batch, seq, action_dim, noisy action for denoise
        # noise_tokens: batch, seq, num_cam, 6/16, height', width', noise token for denoise
        # is_tokens_pad: batch, seq
        # denoise_steps: int, the step of denoise
        is_training = actions is not None # train or val
        bs= qpos.shape[0]
        is_actions_pad = is_pad
        # print('detr vae noise_tokens', noise_tokens.shape)
        ### Obtain latent z from action sequence
        if is_training and not self.disable_vae_latent:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1) if len(qpos_embed.shape) == 2 else qpos_embed # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2 + self.history_step), False).to(qpos.device) # False: not a padding
            
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output) # predict mu and logvar
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)
        
        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_pos = []    
        if self.backbones is not None and self.disable_resnet == False:
            image = current_image[0] # shape: batch, T', num_cam, 3, height, width
            for t in range(self.history_step+1):
                for cam_id, cam_name in enumerate(self.camera_names):
                    features, pos = self.backbones[0](image[:, t, cam_id]) #TODO use different backbone for different camera
                    features = features[0]
                    pos = pos[0]
                    all_cam_features.append(self.input_proj(features)) 
                    all_cam_pos.append(pos)    
        else: # if disable resnet, directly use token as input
            image = current_image[1]# [:,:self.num_temporal_token] # shape: batch, T', num_cam, 6, height', width' hardcode keep consistent with token prediction
            for t in range(self.history_step+1):
                for cam_id, cam_name in enumerate(self.camera_names):
                    features = image[:,t,cam_id] # B C H W
                    features = self.input_proj(features)
                    pos = self.pos_encoder(features).to(features.dtype)
                    all_cam_features.append(features)
                    all_cam_pos.append(pos)    
        # proprioception features
        proprio_input = self.input_proj_robot_state(qpos)  
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3) # B C H W*num_view*T
        pos = torch.cat(all_cam_pos, axis=3) # B C H W*num_view*T
        # deal with token patchfy
        noise_tokens = rearrange(
            noise_tokens,
            'b s n d (ph p1) (pw p2) -> b s n (d p1 p2) ph pw',
            p1=self.patch_size, p2=self.patch_size
        ) if noise_tokens is not None else None
        # print('detr vae noise_tokens', noise_tokens.shape)
        # is_tokens_pad   self.num_temporal_token B T
        if  is_tokens_pad is not None:
            is_tokens_pad = is_tokens_pad.unsqueeze(2).repeat(1,1,self.num_pred_token_per_timestep).reshape(bs,-1)
        else:
            is_tokens_pad = torch.zeros(bs, self.num_pred_token_per_timestep * self.num_temporal_token, dtype=torch.bool).to(qpos.device)
            is_pad = torch.zeros(bs, self.num_queries, dtype=torch.bool).to(qpos.device)

        if self.token_pe_type == 'learned':
            query_embed_token = self.query_embed_token.weight   
        else:
            query_embed_token = self.query_embed_token_fixed.to(qpos.device)
        # print('detr query_embed_token', query_embed_token.shape)
        hs_action, hs_token = self.transformer(src, None, self.query_embed.weight, query_embed_token, pos, latent_input, proprio_input, self.additional_pos_embed.weight, noisy_actions, noise_tokens, denoise_steps, self.denoise_step_pos_embed.weight, is_actions_pad, is_tokens_pad)
        
        a_hat = self.action_head(hs_action)
        is_pad_a_hat = self.is_pad_head(hs_action)
        pred_token = self.token_head(hs_token) if hs_token is not None else None # B T'*N*H'*W' 6*self.patch_size*self.patch_size
        is_pad_token_hat = self.is_pad_head(hs_token) if hs_token is not None else None
        is_pad_hat = torch.cat([is_pad_a_hat, is_pad_token_hat], axis=1) if is_pad_token_hat is not None else is_pad_a_hat
        
        pred_token = rearrange(
            pred_token,
            'b (t n hp wp) (c ph pw) -> b t n c (hp ph) (wp pw)',
            t=self.num_temporal_token, n=len(self.camera_names), hp= self.image_h, wp=self.image_w, ph=self.patch_size, pw=self.patch_size
        ) if pred_token is not None else None
       
        return a_hat, is_pad_hat, pred_token, [mu, logvar]


class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model


def build_diffusion_tp(args): # token prediction
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    backbone = build_backbone(args)
    if args.disable_resnet: # HARDCODE
        for param in backbone.parameters():
            param.requires_grad = False
    backbones.append(backbone) # fixed only one 
    # refer to Transformer_diffusion build_transformer_diffusion
    #  TODO unified prediction
    transformer = build_transformer_diffusion_prediction(args) # TODO design for token pred decoder input noisy input & PE & time_embedding
    encoder = build_encoder(args)
    
    model = DETRVAE_Denoise_Token_Prediction( # TODO design for token pred
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names, # add additional denoise step
        history_step = args.history_step,
        predict_frame = args.predict_frame,
        image_downsample_rate = args.image_downsample_rate,
        temporal_downsample_rate = args.temporal_downsample_rate,
        disable_vae_latent = args.disable_vae_latent,
        disable_resnet=args.disable_resnet,
        patch_size = args.patch_size,
        token_pe_type = args.token_pe_type,
        image_width=args.image_width,
        image_height=args.image_height,
        tokenizer_model_spatial_rate = args.tokenizer_model_spatial_rate,
        tokenizer_model_temporal_rate = args.tokenizer_model_temporal_rate,
        resize_rate = args.resize_rate,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model


def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

