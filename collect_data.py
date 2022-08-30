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
import pickle
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
import uninoise
import tennis2d
import tennis2d_multi as tennis2d_mrl
import agent
import wmnet
import uninoise
matplotlib.rcParams.update({'font.size': 22})
# World Model parameters
EXP_SIZES = [int(1e3)]
WM_EPOCHS = int(1e2)
WM_BATCH_SIZES = [32,64,128,256,512]
WM_LRS = [1e-4, 1e-5]
WM_HIDDEN_SIZES = [[256, 128, 64]]
WM_N_FRAMES = [4]


# Deep RL parameters
validation_episodes = 30
continuous_control = True
n_actions = 2
grad_clip = True
num_skip = 4 # 1 means no frame skipping
max_timestep = int(360 // num_skip) # 6 seconds per episode
task = 'counter'
folder_name = "results_meta_wm/"
if task == 'smash':
    folder_name = 'results_meta_wm_smash/'
# prefix = folder_name + "wm_adapt_"
prefix = folder_name + "wm_"
prefix_orig = prefix
# This part is for adaptation to new physical parameters
ball_mass_list = [9,9.5,10,10.5,11]
wind_list = [-7.5, -5, 0, 5, 7.5]


dmc_params_list = list(product(ball_mass_list, wind_list))
# dmc_params_list = list(product(ball_mass_list_adapt, wind_list_adapt))

# Run mode
render_mode = False
shorter_eps = False
train_wm = True
inject_obs_noise = False
use_ignore_state = False


def ignore_states_before_net(env):
    done = False
    while not(env.pass_net) and not(done):
        action = torch.add(torch.zeros((1, n_actions), device=device), 0.5).float()
        next_state, reward, done = env.step(action.cpu().detach().numpy())
    return next_state, reward, done

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def collect_data(exp_size, dmc_idx, ball_mass, wind_magnitude, n_frames):
    memory = wmnet.Experience()
    train = True
    env = tennis2d.create_tennis2D_env(train, max_timestep, num_skip, continuous_control, 
                                   render_mode, shorter_eps, train_wm, noise=inject_obs_noise, 
                                   ball_mass=ball_mass, wind_magnitude=wind_magnitude,
                                   task=task)
    exp_filename = prefix + 'dmc'+str(dmc_idx)+'_dataset.pkl'
    while len(memory) < exp_size:
        noise = uninoise.UniNoise(n_actions)
        env.reset()
        multiple_frames = deque([],maxlen=n_frames)
        next_multiple_frames = deque([],maxlen=n_frames)
        done = False

        
        while not(done):
            while len(multiple_frames) < n_frames-1:
                current_frame = env.get_env_state()
                multiple_frames.append(current_frame)
                action = torch.add(torch.zeros((1, n_actions), device=device), 0.5).float()
                next_state, reward, done = env.step(action.cpu().detach().numpy())
                reward = torch.tensor([reward], device=device)
                next_frame = env.get_env_state()
                next_multiple_frames.append(next_frame)


            current_frame = env.get_env_state()
            multiple_frames.append(current_frame)

            state = torch.tensor(np.array(multiple_frames).flatten()).float().unsqueeze(0).to(device)
            # No Action
            # action = torch.add(torch.zeros((1, n_actions), device=device), 0.5).float()
            action = noise.get_action(torch.add(torch.zeros((1, n_actions)), 0.5).float()).to(device)
            next_state, reward, done = env.step(action.cpu().detach().numpy())
            reward = torch.tensor([reward], device=device)
            next_frame = env.get_env_state()
            next_multiple_frames.append(next_frame)

            # Observe new state
            if not done:
                next_state = torch.tensor(np.array(next_multiple_frames).flatten()).float().unsqueeze(0).to(device)
                # For physics prediction
                memory.push(state, action, next_state, reward)                
            else:
                next_state = None
    # For RL training need to store the final state but for physics prediction we do not do it 
    # because we need next state to compute prediction error
    # memory.push(state, action, next_state, reward)       

    file = open(exp_filename, 'wb+')
    pickle.dump(memory, file)
    
for dmc_idx, dmc_params in enumerate(dmc_params_list):
    ball_mass, wind_magnitude = dmc_params
    collect_data(EXP_SIZES[0], dmc_idx, ball_mass, wind_magnitude, n_frames=WM_N_FRAMES[0])
    print('Ball mass:', ball_mass)
    print('Wind Mag:', wind_magnitude)