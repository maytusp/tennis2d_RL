from joblib import Parallel, delayed
import multiprocessing
# num_cores = multiprocessing.cpu_count()
num_cores = 1
import random
import sys
import pygame

import pymunk
import pymunk.pygame_util
from pymunk import Vec2d
import numpy as np
import os
import argparse
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
import tennis2d_multi as multi_tn
import agent as ag
import wmnet
import use_wm as uwm
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--env_mode', type=str, nargs='?', default='original')
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--render', default=False, action='store_true')
parser.add_argument('--use_wm',  default=False, action='store_true')
parser.add_argument('--pred_steps', type=int, nargs='?', default=4)
parser.add_argument('--wm_type', type=str, nargs='?', default='learned_dynamics')
parser.add_argument('--wm_loss', type=str, nargs='?', default='mse')
parser.add_argument('--reward_func', type=str, nargs='?', default='rbf')
parser.add_argument('--best_models', type=int, nargs='?', default=0)
parser.add_argument('--inject_obs_noise', default=False, action='store_true')
parser.add_argument('--multi_params_train', default=False, action='store_true')
args = parser.parse_args()

matplotlib.rcParams.update({'font.size': 22})

# Deep RL parameters
validation_episodes = 30
CV = 1
continuous_control = True
n_actions = 2 if continuous_control else 5
# HYPERPARAMETERS  TO TUNE
NUM_EPISODES = int(2e3)
GAMMAS = [1]
# Deep learning Hyperparameters
LRS = [1e-4] # Learning rate
TAUS = [1e-3]
OUNOISE_PARAMS = [#{'theta': 0.2, 'max_sigma':1, 'min_sigma':0.2},
                  {'theta': 0.15, 'max_sigma':0.6, 'min_sigma':0.2}]
                  # {'theta': 0.2, 'max_sigma':1, 'min_sigma':0.6},
                  # {'theta': 0.4, 'max_sigma':1, 'min_sigma':0.4},
                  #                  {'theta': 0.4, 'max_sigma':1, 'min_sigma':0.6}]
BATCH_SIZES = [128]
BUFFER_SIZES = [int(1e4)]
EPSILONS = [1]
N_FRAMES = [4] # k
EXPLS = [0.6]
HIDDEN_SIZES = [[512, 256, 128]]
SEEDS = [i for i in range(10)]
grad_clip = False
swing_delay = 0 # 0 sec
num_skip = 4 # 1 means no frame skipping
max_timestep = int(6000 // num_skip) # For training
folder_name = "results_multi_agent_ddpg/"
wm_folder_name = "results_meta_wm_mrl/"
prefix = folder_name + "DDPG_"
prefix_orig = prefix

# Run mode
env_mode = args.env_mode
train_mode = args.train
render_mode = args.render
wm_future_state = args.use_wm
wm_pred_steps = args.pred_steps
wm_type = args.wm_type
wm_loss = args.wm_loss
reward_func = args.reward_func
inject_obs_noise = args.inject_obs_noise
multi_params_train = args.multi_params_train


adaptation = False
short_eps = True
relative_vec = False
validation_every = 10
baseline_best_idx = 8
bilinear_best_idx = 0
l2_best_idx = 0

num_agents = 2
num_models = 10

wm_best_idx_dict = {'mse':baseline_best_idx, 'bilinear':bilinear_best_idx, 
                    'l2':l2_best_idx}
if multi_params_train:
    prefix += 'multi_params_'
# For naming files saved from different setting
if wm_future_state:
    if wm_type == 'true_dynamics':
        prefix += 'rm_' + 'steps_' + str(wm_pred_steps) + "_"
    elif wm_type == 'learned_dynamics':
        prefix += 'wm_' + 'steps_' + str(wm_pred_steps) + "_"
        
        if wm_loss == 'bilinear':
            prefix += 'bilinear_'
        elif wm_loss == 'l2':
            prefix += 'l2_'
            
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
if not(train_mode):
    best_scores, best_models = find_top5_model(prefix, [i for i in range(num_models)])
    best_params = 0.0001, 0.01, 1, 0.6, {'theta': 0, 'max_sigma': 4, 'min_sigma': 0.2}, 64, 100000, 1, 4, [512, 256, 128], 0
    
PARAMS = list(product(LRS, TAUS, GAMMAS, EXPLS, OUNOISE_PARAMS, BATCH_SIZES, BUFFER_SIZES, EPSILONS, N_FRAMES, HIDDEN_SIZES, SEEDS))
PARAMS_loss = []


if env_mode == 'original' and multi_params_train:
    ball_mass_list = [10]
    wind_list = [-2.5,-1.5,0,1.5,2.5]
    dmc_params_list = list(product(ball_mass_list, wind_list))

    num_dmc_params = len(dmc_params_list)
    print('Num phy params:', num_dmc_params)
    
# Environment Physical Parameters
if env_mode == 'adjusted':
    ball_mass_list = [10]
    wind_list = [-2.5,-1.5,0,1.5,2.5]
    dmc_params_list = list(product(ball_mass_list, wind_list))

    num_dmc_params = len(dmc_params_list)
    print('Num phy params:', num_dmc_params)

if not(train_mode):
    max_timestep = int(600000 // num_skip) # For evaluation

def train(agents, env, expl, ounoise_params, param_idx, wm_nets):
    total_returns = []
    eval_return_mean_list = []
    eval_return_std_list = []
    eval_score_mean_list = []
    eval_score_std_list = []
    actor_loss_list = [[] for i in range(num_agents)] 
    critic_loss_list = [[] for i in range(num_agents)]
    
    best_return = -1e6
    running_actor_losses = [0.0 for i in range(num_agents)] 
    running_critic_losses = [0.0 for i in range(num_agents)] 
    n_frames = agents[0].n_frames
    

    steps = 0
    vary_eps_steps = int(expl * NUM_EPISODES * max_timestep)
    for i_episode in range(NUM_EPISODES):
    
        noises = [ounoise.OUNoise(action_dim=n_actions,
                                theta=ounoise_params['theta'],
                                max_sigma=ounoise_params['max_sigma'],
                                min_sigma=ounoise_params['min_sigma'], 
                                decay_period=vary_eps_steps) for agent in agents]

        # Initialize the environment and state
        env.reset()
        multiple_frames = deque([],maxlen=n_frames)
        next_multiple_frames = deque([],maxlen=n_frames)
        
        ball_multiple_frames = deque([],maxlen=n_frames)
        ball_next_multiple_frames = deque([],maxlen=n_frames)        
        rewards_running = {'0':0.0,'1':0.0}
        done = False
        
            
        while not(done):
            reward_buffer = {'0':0.0,'1':0.0}
            while len(multiple_frames) < n_frames-1:
                current_frame = env.get_env_state()
                multiple_frames.append(current_frame)
                ball_multiple_frames.append(env.get_ball_state(current_frame))
                actions = [torch.add(torch.zeros((1, n_actions), device=device), 0.5).float() for i, agent in enumerate(agents)]
                next_state, rewards, done = env.step(actions)
                
                for i, agent in enumerate(agents):
                    rewards_running[str(i)] += rewards['player'+str(i+1)]
                    reward_buffer[str(i)] = torch.tensor([rewards['player'+str(i+1)]], device=device)

                next_frame = env.get_env_state()
                next_multiple_frames.append(next_frame)
                ball_next_multiple_frames.append(env.get_ball_state(next_frame))
                

            current_frame = env.get_env_state()
            multiple_frames.append(current_frame)
            ball_multiple_frames.append(env.get_ball_state(current_frame))
            state = torch.tensor(np.array(multiple_frames).flatten()).float().unsqueeze(0).to(device)
            # Add future states predictions as additional inputs
            if wm_future_state:
                input_frames = ball_multiple_frames
                pred_next_frames = predict_future_states_mrl(input_frames, wm_pred_steps, wm_type, wm_nets)
                future_input = torch.tensor(pred_next_frames.flatten()).float().unsqueeze(0).to(device)
                state = torch.cat((state, future_input), 1)            
            actions = [noises[i].get_action(agent.get_action(state), steps).to(device) for i, agent in enumerate(agents)]
            next_state, rewards, done = env.step(actions)
            
            
            for i, agent in enumerate(agents):
                rewards_running[str(i)] += rewards['player'+str(i+1)]
                reward_buffer[str(i)] = torch.tensor([rewards['player'+str(i+1)]], device=device)
                
            next_frame = env.get_env_state()
            next_multiple_frames.append(next_frame)
            ball_next_multiple_frames.append(env.get_ball_state(next_frame))
            
            
            # Observe new state
            if not done:
                next_state = torch.tensor(np.array(next_multiple_frames).flatten()).float().unsqueeze(0).to(device)
                # Add future states predictions as additional inputs
                if wm_future_state:
                    input_frames = ball_next_multiple_frames
                    pred_next_frames = predict_future_states_mrl(input_frames, wm_pred_steps, wm_type, wm_nets)
                    future_input = torch.tensor(pred_next_frames.flatten()).float().unsqueeze(0).to(device)
                    next_state = torch.cat((next_state, future_input), 1)                
            else:
                next_state = None
            
            # Store the transition in memory
            for i, agent in enumerate(agents):
                # if env.execute['player'+str(i+1)]:
                # if env.ball_body.velocity[0]>0 and
                agent.memory.push(state, actions[i].to(device), next_state, reward_buffer[str(i)])
            
            
            # Training agents
            for i, agent in enumerate(agents):
                running_actor_losses[i], running_critic_losses[i] = agent.learn(running_actor_losses[i], 
                                                                     running_critic_losses[i])
            if steps % 100 == 99:    
                if (env_mode == 'original' and multi_params_train) or env_mode == 'adjusted':
                    rand_param_idx = np.random.choice(num_dmc_params)
                    rand_ball_mass, rand_wind_magnitude = dmc_params_list[rand_param_idx]
                    env.ball_mass =  rand_ball_mass
                    env.wind_magnitude = rand_wind_magnitude
    
            if steps % 100 == 99:    # print every 2000 mini-batches
                [actor_loss_list[i].append(running_actor_losses[i] / 100) for i in range(num_agents)]
                [critic_loss_list[i].append(running_critic_losses[i] / 100) for i in range(num_agents)]
                running_actor_losses = [0.0 for i in range(num_agents)]
                running_critic_losses = [0.0 for i in range(num_agents)]
            if done:
                total_returns.append(rewards_running['0']+rewards_running['1'])

            steps += 1

        # Save model every "validation_every" episodes                
        if i_episode % validation_every == validation_every-1:
            print("Evaluuate at episode:", i_episode)
            eval_returns, eval_scores = evaluate(agents, env, validation_episodes, wm_nets)
            eval_return_array = np.array(eval_returns)
            eval_return_mean = np.mean(eval_return_array, axis=0)
            eval_return_std = np.std(eval_return_array, axis=0)
            eval_return_mean_list.append(eval_return_mean)
            eval_return_std_list.append(eval_return_std)
            
            eval_score_array = np.array(eval_scores)
            eval_score_mean = np.mean(eval_score_array, axis=0)
            eval_score_std = np.std(eval_score_array, axis=0)            
            eval_score_mean_list.append(eval_score_mean)
            eval_score_std_list.append(eval_score_std)
            if eval_return_mean > best_return:
                best_return = eval_return_mean
                [agent.save(prefix) for agent in agents]
                print("Model Saved")
                print("Current Return:", eval_return_mean)
    
    return total_returns, eval_return_mean_list, eval_return_std_list, \
            eval_score_mean_list, eval_score_std_list, actor_loss_list, critic_loss_list

def evaluate(agents, env, validation_episodes, wm_nets):
    total_returns = []
    total_scores = []
    steps = 0
    n_frames = agents[0].n_frames
    
    for i_episode in range(validation_episodes):
        # Randomly selected ball masss and wind magnitude for nonstationary training

        # Initialize the environment and state
        env.reset()
        multiple_frames = deque([],maxlen=n_frames)
        next_multiple_frames = deque([],maxlen=n_frames)
        
        ball_multiple_frames = deque([],maxlen=n_frames)
        ball_next_multiple_frames = deque([],maxlen=n_frames)        
        rewards_running = {'0':0.0,'1':0.0}
        done = False

        while not(done):
            while len(multiple_frames) < n_frames-1:
                current_frame = env.get_env_state()
                multiple_frames.append(current_frame)
                ball_multiple_frames.append(env.get_ball_state(current_frame))
                actions = [torch.add(torch.zeros((1, n_actions), device=device), 0.5).float() for i, agent in enumerate(agents)]
                next_state, rewards, done = env.step(actions)
                
                for i, agent in enumerate(agents):
                    rewards_running[str(i)] += rewards['player'+str(i+1)]
                    # reward_buffer[str(i)] = torch.tensor([rewards['player'+str(i+1)]], device=device)

                next_frame = env.get_env_state()
                next_multiple_frames.append(next_frame)
                ball_next_multiple_frames.append(env.get_ball_state(next_frame))
                
            if steps % 100 == 99:    
                if (env_mode == 'original' and multi_params_train) or env_mode == 'adjusted':
                    rand_param_idx = np.random.choice(num_dmc_params)
                    rand_ball_mass, rand_wind_magnitude = dmc_params_list[rand_param_idx]
                    env.ball_mass =  rand_ball_mass
                    env.wind_magnitude = rand_wind_magnitude                          

            current_frame = env.get_env_state()
            multiple_frames.append(current_frame)
            ball_multiple_frames.append(env.get_ball_state(current_frame))
            
            state = torch.tensor(np.array(multiple_frames).flatten()).float().unsqueeze(0).to(device)
            # Add future states predictions as additional inputs
            if wm_future_state:
                input_frames = ball_multiple_frames
                pred_next_frames = predict_future_states_mrl(input_frames, wm_pred_steps, wm_type, wm_nets)
                future_input = torch.tensor(pred_next_frames.flatten()).float().unsqueeze(0).to(device)
                state = torch.cat((state, future_input), 1)
                
            actions = [agent.get_action(state).to(device) for i, agent in enumerate(agents)]
            next_state, rewards, done = env.step(actions)
            
            
            for i, agent in enumerate(agents):
                rewards_running[str(i)] += rewards['player'+str(i+1)]
                # reward_buffer[str(i)] = torch.tensor([rewards['player'+str(i+1)]], device=device)
                
            next_frame = env.get_env_state()
            next_multiple_frames.append(next_frame)
            ball_next_multiple_frames.append(env.get_ball_state(next_frame))
            
            # Observe new state
            if not done:
                next_state = torch.tensor(np.array(next_multiple_frames).flatten()).float().unsqueeze(0).to(device)
                # Add future states predictions as additional inputs
                if wm_future_state:
                    input_frames = ball_next_multiple_frames
                    pred_next_frames = predict_future_states_mrl(input_frames, wm_pred_steps, wm_type, wm_nets)
                    future_input = torch.tensor(pred_next_frames.flatten()).float().unsqueeze(0).to(device)
                    next_state = torch.cat((next_state, future_input), 1)                
            else:
                next_state = None
    
            if done:
                total_returns.append(rewards_running['0']+rewards_running['1'])
                total_scores.append(env.cum_rewards['player1']+env.cum_rewards['player2'])

            steps += 1
    
    return total_returns, total_scores

def process(param_idx, params, pretrain=False):
    PARAMS_train_returns = []
    print('Start training params:', params)
    learning_rate, tau, gamma, expl, ounoise_params, batch_size, buffer_size, epsilon, \
    n_frames, size_hidden_layers, num_seed = params
    np.random.seed(num_seed)
    torch.manual_seed(num_seed)
    is_train = True
    env = multi_tn.create_tennis2D_env(num_agents, is_train, max_timestep, num_skip,
                                       continuous_control, render_mode, short_eps,
                                       noise=inject_obs_noise)    
    
    ball_frame_dim = 4
    frame_dim  = env.get_env_state_dim()
    if wm_future_state and wm_type == 'learned_dynamics':
        wm_nets = uwm.get_wm(wm_folder_name, wm_best_idx_dict[wm_loss], wm_loss, n_frames)
    else:
        wm_nets = None
        
    if wm_future_state:
        pred_state_dim = (ball_frame_dim*wm_pred_steps)
    else:
        pred_state_dim = 0
        


    

    agents = [ag.get_agent(learning_rate, batch_size, grad_clip, buffer_size, tau, gamma,
             frame_dim, n_actions, n_frames, size_hidden_layers, device, agent_idx, param_idx, pred_state_dim) 
              for agent_idx in range(num_agents)]

    if train_mode:

        train_returns, eval_return_mean_list, eval_return_std_list, \
        eval_score_mean_list, eval_score_std_list, \
        actor_loss_list, critic_loss_list \
        = train(agents, env, expl, ounoise_params, param_idx, wm_nets)
        
        train_returns = np.array(train_returns)
        eval_returns_mean = np.array(eval_return_mean_list)
        eval_returns_std = np.array(eval_return_std_list)
        eval_scores_mean = np.array(eval_score_mean_list)
        eval_scores_std = np.array(eval_score_std_list)
        

        np.save(prefix+str(param_idx)+'_train_returns.npy', train_returns)
        np.save(prefix+str(param_idx)+'_eval_returns_mean.npy', eval_returns_mean)
        np.save(prefix+str(param_idx)+'_eval_returns_std.npy', eval_returns_std)
        np.save(prefix+str(param_idx)+'_eval_scores_mean.npy', eval_scores_mean)
        np.save(prefix+str(param_idx)+'_eval_scores_std.npy', eval_scores_std)        
        
        # TODO Save losses of two agents
        np.save(prefix+str(param_idx)+'_actor_loss.npy', np.array(actor_loss_list))
        np.save(prefix+str(param_idx)+'_critic_loss.npy', np.array(critic_loss_list))
        

        print('Training Log Saved')
    # Watching mode
    else:
        print('Watching mode')
        for agent_idx, agent in enumerate(agents):
            agent.load(prefix)
            
        # TODO load actor net
        num_eval_episodes = 100
        eval_returns, eval_scores = evaluate(agents, env, num_eval_episodes, wm_nets)
        
        eval_returns_mean = np.mean(np.array(eval_returns))
        eval_returns_std = np.std(np.array(eval_returns))
        
        eval_scores_mean = np.mean(np.array(eval_scores))
        eval_scores_std = np.std(np.array(eval_scores))
        
        print('Mean return:', eval_returns_mean)
        print('Std return:', eval_returns_std)
        print('Mean score:', eval_scores_mean)
        print('Std score:', eval_scores_std)
        return eval_returns_mean, eval_returns
        

if train_mode:
    results = Parallel(n_jobs=min(num_cores, len(PARAMS)))(delayed(process)(i, params) for i, params in enumerate(PARAMS))                
    
else:
    num_models = len(best_models)
    avg_returns_list = []
    eval_returns_list = []
    for best_model in best_models:
        print('evaluate model:', best_model)
        avg_return, eval_returns = process(best_model, best_params)
        avg_returns_list.append(avg_return)
        eval_returns_list.append(eval_returns)


    avg_returns_array = np.array(avg_returns_list)
    eval_returns_array = np.array(eval_returns_list)
    np.save(prefix+'avg_returns.npy', avg_returns_array)
    np.save(prefix+'eval_returns.npy', eval_returns_array)
    
    print('Avg Hits:', np.mean(avg_returns_array), np.std(avg_returns_array))
    print('Max Hits:', np.max(eval_returns_array))
    print('Min Hits:', np.min(eval_returns_array))
    

