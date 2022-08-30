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
import torch.optim as optim
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
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
class ReplayBuffer(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class Actor(nn.Module):

    def __init__(self, inputs, outputs, num_hidden, hidden_size):
        super(Actor, self).__init__()
        num_hidden = len(hidden_size)
        self.input_layer = nn.Linear(inputs, hidden_size[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size[i], hidden_size[i+1]) for i in range(num_hidden-1)])
        self.output_layer = nn.Linear(hidden_size[num_hidden-1], outputs)
    
    def forward(self, x):
        # x.to(device)

        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return torch.sigmoid(self.output_layer(x))

class Critic(nn.Module):

    def __init__(self, inputs, outputs, num_hidden, hidden_size):
        super(Critic, self).__init__()
        num_hidden = len(hidden_size)
        self.input_layer = nn.Linear(inputs, hidden_size[0])
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_size[i], hidden_size[i+1]) for i in range(num_hidden-1)])
        self.output_layer = nn.Linear(hidden_size[num_hidden-1], outputs)
    
    def forward(self, x):
        # x.to(device)

        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        
        return self.output_layer(x)
    
def optimize_model(actor_policy_net, actor_target_net, actor_optimizer, actor_scheduler, 
                   critic_policy_net, critic_target_net, critic_optimizer, critic_scheduler,
                    memory, BATCH_SIZE, state_dim, tau, GAMMA, running_actor_loss, running_critic_loss, grad_clip):
    if len(memory) < BATCH_SIZE:
        return running_actor_loss, running_critic_loss
    transitions = memory.sample(BATCH_SIZE)
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

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute V(s_{t+1}) for all next states.
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE)
    
    # Input for critic target network
    with torch.no_grad():
        state_action_batch = torch.cat((state_batch, action_batch), 1).float()
        next_actions = actor_target_net(non_final_next_states)
        next_states_actions = torch.cat((non_final_next_states, next_actions), 1)    
        # Once again can omit the condition if batch size is large enough
        if sum(non_final_mask) > 0:
            next_state_values[non_final_mask] = critic_target_net(next_states_actions).squeeze(1)
            # next_state_values[non_final_mask] = critic_target_net(non_final_next_states, next_actions).squeeze(1)
        else:
            next_state_values = torch.zeros_like(next_state_values)

    state_action_values = critic_policy_net(state_action_batch)

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Update Critic
    criterion = nn.SmoothL1Loss(reduction='sum')
    critic_loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    # critic_loss = ((state_action_values - expected_state_action_values.unsqueeze(1))**2).sum()
    critic_optimizer.zero_grad()
    critic_loss.backward()
    if grad_clip:
        for critic_param in critic_policy_net.parameters():
            critic_param.grad.data.clamp_(-1, 1)   
    critic_optimizer.step()
    critic_scheduler.step()
    
    
    # Update Actor
    actor_loss = -critic_policy_net(torch.cat((state_batch, actor_policy_net(state_batch)), 1)).sum()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    if grad_clip:
        # Limit magnitude of gradient for update step
        for actor_param in actor_policy_net.parameters():
            actor_param.grad.data.clamp_(-1, 1)       
    actor_optimizer.step()
    actor_scheduler.step()
    

    # update target networks 
    for target_param, param in zip(actor_target_net.parameters(), actor_policy_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    for target_param, param in zip(critic_target_net.parameters(), critic_policy_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    running_actor_loss += actor_loss.item()
    running_critic_loss += critic_loss.item()
    return running_actor_loss, running_critic_loss


class get_agent:
    def __init__(self, learning_rate, batch_size, grad_clip, buffer_size, tau, gamma,
             frame_dim, n_actions, n_frames, size_hidden_layers, device, agent_idx, param_idx, pred_state_dim):
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.buffer_size = buffer_size
        self.n_frames = n_frames
        self.size_hidden_layers = size_hidden_layers
        self.num_hidden_layers = len(size_hidden_layers)
        self.device = device
        self.agent_idx = agent_idx
        self.param_idx = param_idx
        
        self.num_hidden_layers = len(self.size_hidden_layers)
        self.state_dim  = (frame_dim * self.n_frames) + pred_state_dim
        self.ball_state_dim = 4
            
        self.memory = ReplayBuffer(buffer_size)
        
        self.model_prefix = "param_"+str(param_idx)+"_agent_"+ str(agent_idx)
        
        # Create Actor
        self.actor_policy_net = Actor(self.state_dim, n_actions, self.num_hidden_layers, 
                                      size_hidden_layers).to(self.device)
        # Create Critic
        self.critic_policy_net = Critic(self.state_dim+n_actions, 1, self.num_hidden_layers, 
                                        size_hidden_layers).to(self.device)
        
        # Create target nets as copies of policy nets
        self.actor_target_net = Actor(self.state_dim, n_actions, self.num_hidden_layers, 
                                      size_hidden_layers).to(self.device)
        self.critic_target_net = Critic(self.state_dim+n_actions, 1, self.num_hidden_layers, 
                                        size_hidden_layers).to(self.device)
        self.actor_target_net.load_state_dict(self.actor_policy_net.state_dict())
        self.critic_target_net.load_state_dict(self.critic_policy_net.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor_policy_net.parameters(), lr=learning_rate)
        self.actor_scheduler = torch.optim.lr_scheduler.ConstantLR(self.actor_optimizer, 
                                                                   factor=1, total_iters=1e3)                
        self.critic_optimizer = torch.optim.Adam(self.critic_policy_net.parameters(), lr=learning_rate)
        self.critic_scheduler = torch.optim.lr_scheduler.ConstantLR(self.critic_optimizer, 
                                                                    factor=1, total_iters=1e3)

        self.actor_target_net.eval()
        self.critic_target_net.eval()
        
        
    def learn(self, running_actor_loss, running_critic_loss): 
        optimize_model(self.actor_policy_net, self.actor_target_net, self.actor_optimizer, self.actor_scheduler, 
                           self.critic_policy_net, self.critic_target_net, self.critic_optimizer, self.critic_scheduler,
                            self.memory, self.batch_size, self.state_dim, self.tau, self.gamma, 
                       running_actor_loss, running_critic_loss, self.grad_clip)
        return running_actor_loss, running_critic_loss
        
    def load(self, prefix):
        self.actor_policy_net.load_state_dict(torch.load(prefix+self.model_prefix+"_actor_model.pth"))
        self.critic_policy_net.load_state_dict(torch.load(prefix+self.model_prefix+"_critic_model.pth"))
        self.actor_policy_net.eval()
        self.critic_policy_net.eval()
    
    def save(self, prefix):
        torch.save(self.actor_policy_net.state_dict(), prefix+self.model_prefix+"_actor_model.pth")
        torch.save(self.critic_policy_net.state_dict(), prefix+self.model_prefix+"_critic_model.pth")
    def get_action(self, state):
        return self.actor_policy_net(state)
