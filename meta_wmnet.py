from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
import random
import sys
import numpy as np
import os

import torch
import torch.optim as optimize_model
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cpu")
from collections import namedtuple, deque
from itertools import count
import math
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import ounoise

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
only_position = False
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
class prediction_net(nn.Module):

    def __init__(self, inputs, outputs, num_hidden, hidden_size):
        super(prediction_net, self).__init__()
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
    
class knowledge_net(nn.Module):

    def __init__(self, inputs, outputs, num_hidden, hidden_size):
        super(knowledge_net, self).__init__()
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

        return F.relu(self.output_layer(x))
    
# Fully connected neural networks for dynamics prediction

class param_net(nn.Module):

    def __init__(self, inputs, outputs, num_hidden, hidden_size):
        super(param_net, self).__init__()
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

        return F.relu(self.output_layer(x))
    
class projection_net(nn.Module):

    def __init__(self, inputs, outputs, num_hidden, hidden_size):
        super(projection_net, self).__init__()
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

def get_ball_state(state, n_frames, only_position=only_position):
    idx_list = []
    frame_dim = int(state.shape[1] // n_frames)
    if only_position:
        for frame in range(n_frames):
            start_idx = int(frame * frame_dim)
            idx_list += [start_idx+4, start_idx+5]
    else:
        for frame in range(n_frames):
            start_idx = int(frame * frame_dim)
            idx_list += [start_idx+4, start_idx+5, start_idx+6, start_idx+7]
        
    indices = torch.tensor(idx_list)
    ball_state = torch.index_select(state, 1, indices)
    return ball_state

def get_ball_state_array(state_array, train_wm=False, only_position=only_position):
    idx_list = []
    if only_position:
        if train_wm:
            idx_list = [4,5]
        else:
            idx_list = [6,7]
    else:
        if train_wm:
            idx_list = [4,5,6,7]
        else:
            idx_list = [6,7,8,9]
            
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
def info_nce_loss(features, n_dataset, n_sample_per_dataset, temperature=0.8, sim_func='bilinear'):
    # Original
    # labels = torch.cat([torch.arange(n_dataset) for i in range(n_sample_per_dataset)], dim=0)
    labels = torch.cat([torch.tensor([i]*n_sample_per_dataset) for i in range(n_dataset)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)
    
    features = F.normalize(features, dim=1)
    
    if sim_func == 'bilinear':
        similarity_matrix = torch.matmul(features, features.T)
    elif sim_func == 'l2':
        similarity_matrix = -torch.cdist(features, features, p=2)
    else:
        print("ERROR: Sim Func is not determined")
        
    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    if sim_func == 'bilinear':
        logits = logits / temperature
    return logits, labels

# Modified from https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html
def triplet_loss(features, n_dataset, n_sample_per_dataset):
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)
    anchor = torch.randn(100, 128, requires_grad=True)
    positive = torch.randn(100, 128, requires_grad=True)
    negative = torch.randn(100, 128, requires_grad=True)
    output = triplet_loss(anchor, positive, negative)
    return output

def meta_sample(meta_memory, n_dataset, n_sample_per_dataset):
    key_list = list(meta_memory.keys())
    for dataset_idx in range(n_dataset):
        # Sampling 1 tuple from memory
        transitions = meta_memory[key_list[dataset_idx]].sample(n_sample_per_dataset)
        batch = Transition(*zip(*transitions))


        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        next_state_batch = torch.cat(batch.next_state)

        if dataset_idx == 0:
            meta_state_batch = torch.clone(state_batch)
            meta_action_batch = torch.clone(action_batch)
            meta_reward_batch = torch.clone(reward_batch)
            meta_next_state_batch = torch.clone(next_state_batch)
        else:
            meta_state_batch = torch.cat([meta_state_batch, state_batch], dim=0)
            meta_action_batch = torch.cat([meta_action_batch, action_batch], dim=0)
            meta_reward_batch = torch.cat([meta_reward_batch, reward_batch], dim=0)
            meta_next_state_batch = torch.cat([meta_next_state_batch, next_state_batch], dim=0)
    return meta_state_batch, meta_action_batch, meta_reward_batch, meta_next_state_batch
    
    
def optimize_model(meta_memory, wm_nets, wm_optimizer, wm_scheduler, 
                   batch_size, state_dim, n_frames, next_frame_idx,
                   running_dynamics_loss, grad_clip, step_range, temp, k_ce,
                   n_dataset, n_sample_per_dataset, 
                   contrastive=False, adaptation=False, sim_func=''):
    is_train = True
    running_dynamics_loss, total_loss= compute_loss(meta_memory, wm_nets, wm_optimizer, wm_scheduler, 
                                                    batch_size, state_dim, n_frames, next_frame_idx,
                                                    running_dynamics_loss, grad_clip, step_range, temp, k_ce,
                                                    is_train, n_dataset, n_sample_per_dataset, 
                                                     contrastive, adaptation, sim_func=sim_func)
    

    prediction_optimizer = wm_optimizer['prediction_net']
    prediction_scheduler = wm_scheduler['prediction_net']
    
    knowledge_optimizer = wm_optimizer['knowledge_net']
    knowledge_scheduler = wm_scheduler['knowledge_net']
    
    param_optimizer = wm_optimizer['param_net']
    param_scheduler = wm_scheduler['param_net']

    projection_optimizer = wm_optimizer['projection_net']
    projection_scheduler = wm_scheduler['projection_net']
    
    
    
    # Update player loss
    prediction_optimizer.zero_grad()
    param_optimizer.zero_grad()
    knowledge_optimizer.zero_grad()
    projection_optimizer.zero_grad()
    
    # Backward
    total_loss.backward()
    
    param_optimizer.step()
    param_scheduler.step()
    
    prediction_optimizer.step()
    prediction_scheduler.step()
    
    projection_optimizer.step()
    projection_scheduler.step()
    
    if not(adaptation):
        
        knowledge_optimizer.step()
        knowledge_scheduler.step()
        
        
    
    return running_dynamics_loss

def meta_wm_prediction(knowledge_net, param_net, prediction_net, ball_state, contrastive):
    latent_predictions = knowledge_net(ball_state)
    param_predictions = param_net(ball_state)
    

    prediction_net_input = torch.cat([latent_predictions, param_predictions], dim=1)

    ball_predictions = prediction_net(prediction_net_input)
    return ball_predictions, param_predictions

def compute_loss(meta_memory, wm_nets, wm_optimizer, wm_scheduler, 
                   batch_size, state_dim, n_frames, next_frame_idx,
                   running_dynamics_loss, grad_clip, step_range, temp, k_ce,
                 is_train, n_dataset, n_sample_per_dataset, contrastive=False, 
                 adaptation=False, sim_func=''):
    knowledge_net = wm_nets['knowledge_net']
    param_net = wm_nets['param_net']
    prediction_net = wm_nets['prediction_net']
    projection_net = wm_nets['projection_net']
    
    
    total_loss = {}
    
    if is_train:
        if not(adaptation):
            knowledge_net.train()
        else:
            knowledge_net.eval()
        param_net.train()
        prediction_net.train()
        projection_net.train()
    else:
        knowledge_net.eval()
        param_net.eval()
        prediction_net.eval()
        projection_net.eval()
    
    
    state_batch, action_batch, reward_batch, non_final_next_states = meta_sample(meta_memory,
                                                                                 n_dataset,
                                                                                n_sample_per_dataset)

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
    ball_predictions, param_predictions = meta_wm_prediction(knowledge_net, param_net, prediction_net, ball_state, contrastive)
    # Ball loss
    ball_loss = ((ball_predictions - (next_ball_frames - initial_ball_frames).unsqueeze(1))**2).mean()




    running_dynamics_loss['prediction'] += ball_loss.item() / step_range

    if contrastive:
        # Info NCE Loss
        CE_loss = nn.CrossEntropyLoss()
        projections = projection_net(param_predictions)
        logits, labels = info_nce_loss(projections, n_dataset, n_sample_per_dataset, temp, sim_func)
        param_loss = k_ce*CE_loss(logits, labels)
        running_dynamics_loss['contrastive'] += param_loss.item() / step_range
        total_loss = ball_loss + param_loss
    else:
        total_loss = ball_loss
        
    return running_dynamics_loss, total_loss