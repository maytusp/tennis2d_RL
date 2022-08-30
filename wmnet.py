from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
import random
import sys
import pygame

import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import os

import torch
import torch.optim as optimize_model
import torch.nn as nn
import torch.nn.functional as F
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Configuring Pytorch
device = torch.device("cpu")
from collections import namedtuple, deque
from itertools import count
import math
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import ounoise
# Fix seed
np.random.seed(0)
torch.manual_seed(0)
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class Experience(object):

    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)        

    def __len__(self):
        return len(self.memory)
    
# Fully connected neural networks for dynamics prediction
class ball_net(nn.Module):

    def __init__(self, inputs, outputs, num_hidden, hidden_size):
        super(ball_net, self).__init__()
        num_hidden = len(hidden_size)
        self.input_layer = nn.Linear(inputs, hidden_size[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size[i], hidden_size[i+1]) for i in range(num_hidden-1)])
        self.output_layer = nn.Linear(hidden_size[num_hidden-1], outputs)
        self.predictions = {}
    def forward(self, x):
        x.to(device)

        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        return self.output_layer(x)
    

def get_player_state(state, n_frames):
    idx_list = []
    frame_dim = int(state.shape[1] // n_frames)

    for frame in range(n_frames):
        start_idx = int(frame * frame_dim)
        idx_list += [start_idx+0, start_idx+1, start_idx+2, start_idx+3]
        
    indices = torch.tensor(idx_list)
    player_state = torch.index_select(state, 1, indices)
    return player_state

def get_ball_state(state, n_frames):
    idx_list = []
    frame_dim = int(state.shape[1] // n_frames)

    for frame in range(n_frames):
        start_idx = int(frame * frame_dim)
        idx_list += [start_idx+4, start_idx+5, start_idx+6, start_idx+7, start_idx+8]
        
    indices = torch.tensor(idx_list)
    ball_state = torch.index_select(state, 1, indices)
    return ball_state

def get_ball_state_array(state_array, train_wm=False):
    idx_list = []
    if train_wm:
        idx_list = [4,5,6,7,8]
    else:
        idx_list = [6,7,8,9,10]
    ball_state_array = state_array[idx_list]
    return ball_state_array

# Ball player interaction
def get_ball_player_int_state(state, n_frames):
    idx_list = []
    frame_dim = int(state.shape[1] // n_frames)

    for frame in range(n_frames):
        start_idx = int(frame * frame_dim)
        idx_list += [start_idx+10]
        
    indices = torch.tensor(idx_list)
    ball_player_int = torch.index_select(state, 1, indices)
    return ball_player_int

# Ball map interaction
def get_ball_map_int_state(state, n_frames):
    idx_list = []
    frame_dim = int(state.shape[1] // n_frames)

    for frame in range(n_frames):
        start_idx = int(frame * frame_dim)
        idx_list += [start_idx+11, start_idx+12, start_idx+13, start_idx+14, start_idx+15]
    
    
    indices = torch.tensor(idx_list)
    ball_map_int = torch.index_select(state, 1, indices)

    return ball_map_int

# Combine four states
def get_separated_state(state, n_frames):
    player_state = get_player_state(state, n_frames)
    ball_state = get_ball_state(state, n_frames)
    ball_player_int_state = get_ball_player_int_state(state, n_frames)
    ball_map_int_state = get_ball_map_int_state(state, n_frames)

    return player_state, ball_state, ball_player_int_state, ball_map_int_state


# The Info NCE loss is modified from https://github.com/sthalles/SimCLR
def info_nce_loss(features, n_param, n_sample_per_param):

    labels = torch.cat([torch.arange(n_param) for i in range(n_sample_per_param)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(self.args.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

    logits = logits / self.args.temperature
    return logits, labels
    
def optimize_model(memory, env, wm_nets, wm_optimizer, wm_scheduler, 
                   batch_size, state_dim, n_frames, next_frame_idx,
                   running_dynamics_loss, grad_clip, step_range):
    is_train = True
    running_dynamics_loss, total_loss = compute_loss(memory, env, wm_nets, wm_optimizer, wm_scheduler, 
                       batch_size, state_dim, n_frames, next_frame_idx,
                       running_dynamics_loss, grad_clip, step_range, is_train)
    

    ball_optimizer = wm_optimizer['ball_net']
    ball_scheduler = wm_scheduler['ball_net']

    
    # Update player loss
    ball_optimizer.zero_grad()
    
    # Backward
    total_loss.backward()
    
    if grad_clip:
        for param in wm_nets['ball_net'].parameters():
            param.grad.data.clamp_(-1, 1)

    ball_optimizer.step()
    ball_scheduler.step()

    
    
    
    return running_dynamics_loss
    
def compute_loss(memory, env, wm_nets, wm_optimizer, wm_scheduler, 
                   batch_size, state_dim, n_frames, next_frame_idx,
                   running_dynamics_loss, grad_clip, step_range, is_train):
    k_ce = 0.1
    ball_net = wm_nets['ball_net']
    
    if is_train:
        ball_net.train()
    else:
        ball_net.eval()
    
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    

    # Can safely omit the condition below to check that not all states in the
    # sampled batch are terminal whenever the batch size is reasonable and
    # there is virtually no chance that all states in the sampled batch are 
    # terminal
    if sum(non_final_mask) > 0:
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
    else:
        non_final_next_states = torch.empty(0,state_dim).to(device)
        
    # TODO Deal with batch size
    state_batch = torch.cat(batch.state)[non_final_mask]
    action_batch = torch.cat(batch.action)[non_final_mask]
    reward_batch = torch.cat(batch.reward)[non_final_mask]
    

    initial_frames = state_batch[:, next_frame_idx:]
    non_final_next_frames = non_final_next_states[:, next_frame_idx:]

    initial_player_frames, initial_ball_frames, \
    initial_ball_player_int_frames, initial_ball_map_int_frames \
    = get_separated_state(initial_frames, 1)

    next_player_frames, next_ball_frames, \
    next_ball_player_int_frames, next_ball_map_int_frames \
    = get_separated_state(non_final_next_frames, 1)

    # State input of dynamics model
    player_state, ball_state, ball_player_int_state, ball_map_int_state = get_separated_state(state_batch, n_frames)
    
    
    # Make Predictions
    ball_predictions = ball_net(ball_state)
    # Ball loss
    ball_loss = ((ball_predictions - (next_ball_frames - initial_ball_frames).unsqueeze(1))**2).mean()
    

            
            
    running_dynamics_loss['ball_net'] += ball_loss.item() / step_range
    
    total_loss = ball_loss

    
    return running_dynamics_loss, total_loss