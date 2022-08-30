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
import tennis2d
import agent
from utils import *
from meta_wmnet import get_ball_state_array
import wmnet
import use_wm as uwm

matplotlib.rcParams.update({'font.size': 22})

parser = argparse.ArgumentParser()
parser.add_argument('--env_mode', type=str, nargs='?', default='original')
# parser.add_argument('--ball_mass', type=str, nargs='?', default=None)
# parser.add_argument('--wind_mag', type=str, nargs='?', default=None)
parser.add_argument('--train', default=False, action='store_true')
parser.add_argument('--render', default=False, action='store_true')
parser.add_argument('--use_wm',  default=False, action='store_true')
parser.add_argument('--pred_steps', type=int, nargs='?', default=4)
parser.add_argument('--wm_type', type=str, nargs='?', default='learned_dynamics')
parser.add_argument('--wm_loss', type=str, nargs='?', default='mse')
parser.add_argument('--ignore_before_net', default=False, action='store_true')
parser.add_argument('--single_step', default=False, action='store_true')
parser.add_argument('--task', type=str, nargs='?', default='counter')
parser.add_argument('--reward_func', type=str, nargs='?', default='rbf')
parser.add_argument('--best_models', type=int, nargs='?', default=0)
parser.add_argument('--inject_obs_noise', default=False, action='store_true')
parser.add_argument('--multi_params_train', default=False, action='store_true')
args = parser.parse_args()




# WM Parameters
wm_size_hidden_layers = [256, 128, 64]
wm_best_model = 0
# Deep RL parameters
validation_episodes = 30
CV = 1
continuous_control = True
n_actions = 2 if continuous_control else 5
# HYPERPARAMETERS  TO TUNE
NUM_EPISODES = int(1e3)
GAMMAS = [1]
# Deep learning Hyperparameters
LRS = [1e-4] # Learning rate
TAUS = [1e-2]
OUNOISE_PARAMS = [#{'theta': 0.2, 'max_sigma':1, 'min_sigma':0.2},
                  {'theta': 0.3, 'max_sigma':0.8, 'min_sigma':0.2}]
                  # {'theta': 0.2, 'max_sigma':1, 'min_sigma':0.6},
                  # {'theta': 0.4, 'max_sigma':1, 'min_sigma':0.4},
                  #                  {'theta': 0.4, 'max_sigma':1, 'min_sigma':0.6}]
BATCH_SIZES = [128]
BUFFER_SIZES = [int(1e4)]
EPSILONS = [1]
N_FRAMES = [4] # k
EXPLS = [0.6]
HIDDEN_SIZES = [[512, 256, 128]]
SEEDS = [i for i in range(30)]
grad_clip = False
swing_delay = 0 # 0 sec
num_skip = 4 # 1 means no frame skipping
max_timestep = int(600 // num_skip)
folder_name = "results_ddpg_exp4/"
wm_folder_name = "results_meta_wm/"
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
ignore_before_net = args.ignore_before_net
single_step = args.single_step
task = args.task
reward_func = args.reward_func
inject_obs_noise = args.inject_obs_noise
multi_params_train = args.multi_params_train


adaptation = False
pretrain = False
short_eps = True
relative_vec = False
validation_every = 10
baseline_best_idx = 5
bilinear_best_idx = 5
l2_best_idx = 5

if task == 'smash':
    print('smash')
    wm_folder_name = "results_meta_wm_smash/"
    print(wm_folder_name)
    baseline_best_idx = 0
    bilinear_best_idx = 0
    l2_best_idx = 0
else:
    print('counter')
    
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

prefix += 'task_' + str(task) + '_'
if not(wm_future_state):
    prefix += str(reward_func) + '_'
if ignore_before_net:
    prefix += 'after_net_'
if single_step:
    prefix += 'single_step_'
if inject_obs_noise:
    prefix += 'noisy_'

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

PARAMS = list(product(LRS, TAUS, GAMMAS, EXPLS, OUNOISE_PARAMS, BATCH_SIZES, BUFFER_SIZES, EPSILONS, N_FRAMES, HIDDEN_SIZES, SEEDS))
PARAMS_loss = []
num_models = len(PARAMS)

print('Num models:', num_models)

if not(train_mode):
    _ ,best_models = find_top5_model(prefix, [i for i in range(num_models)])
    best_params = 0.0001, 0.01, 1, 0.6, {'theta': 0, 'max_sigma': 4, 'min_sigma': 0.2}, 64, 100000, 1, 4, [512, 256, 128], 0

    
if env_mode == 'original' and multi_params_train:
    # Unused models
    # if task == 'counter':
    #     ball_mass_list = [11,13,15]
    #     wind_list = [-27.5,-22.5,-17.5-12.5,-7.5,7.5,12.5,17.5,22.5,27.5]   
    #     dmc_params_list = list(product(ball_mass_list, wind_list))
    # elif task == 'smash':
    #     ball_mass_list = [10.5, 11.5]
    #     wind_list = [-27.5,-17.5,-7.5,7.5,17.5, 22.5, 27.5]
    #     dmc_params_list = list(product(ball_mass_list, wind_list))
    
    # Upper-bound models
    if task == 'counter':
        ball_mass_list = [12, 14, 16]
        wind_list = [-25,-20,-15-10,-5,5,10,15,20,25]
        dmc_params_list = list(product(ball_mass_list, wind_list))
    elif task == 'smash':
        ball_mass_list = [11, 12]
        wind_list = [-30,-20,-10,10,15, 20, 25]    
        
    num_dmc_params = len(dmc_params_list)
    print('Num phy params:', num_dmc_params)   
    
# Environment Physical Parameters
if env_mode == 'adjusted':
    if task == 'counter':
        ball_mass_list = [12, 14, 16]
        wind_list = [-25,-20,-15-10,-5,5,10,15,20,25]
        dmc_params_list = list(product(ball_mass_list, wind_list))
    elif task == 'smash':
        ball_mass_list = [11, 12]
        wind_list = [-30,-20,-10,10,15, 20, 25]

        dmc_params_list = list(product(ball_mass_list, wind_list))

def train(i,  actor_policy_net, actor_target_net, actor_optimizer, actor_scheduler, 
            critic_policy_net, critic_target_net, critic_optimizer, critic_scheduler,
            epsilon, memory, n_frames, BATCH_SIZE, state_dim, tau, GAMMA, expl, ounoise_params, wm_nets):
    total_returns = []
    eval_return_mean_list = []
    eval_return_std_list = []
    actor_loss_list = []
    critic_loss_list = []
    steps = 0
    
    best_return = -1e6
    vary_eps_steps = int(expl * NUM_EPISODES * max_timestep)
    running_actor_loss = 0.0 
    running_critic_loss = 0.0
    train = True
    
    env = tennis2d.create_tennis2D_env(train, max_timestep, num_skip, continuous_control, 
                                       render_mode, short_eps, noise=inject_obs_noise, task=task,
                                       reward_func=reward_func)
    for i_episode in range(NUM_EPISODES):
        
        # Randomly selected ball masss and wind magnitude for nonstationary training
        if env_mode == 'original' and multi_params_train:
            rand_param_idx = np.random.choice(num_dmc_params)
            rand_ball_mass, rand_wind_magnitude = dmc_params_list[rand_param_idx]
            env.ball_mass =  rand_ball_mass
            env.wind_magnitude = rand_wind_magnitude
            
        noise = ounoise.OUNoise(action_dim=n_actions, 
                                theta=ounoise_params['theta'],
                                max_sigma=ounoise_params['max_sigma'], 
                                min_sigma=ounoise_params['min_sigma'], 
                                decay_period=vary_eps_steps)
        # if i_episode % 200 == 0:
        #     print("episode ", i_episode, "/", NUM_EPISODES)

        # Initialize the environment and state
        env.reset()
        multiple_frames = deque([],maxlen=n_frames)
        next_multiple_frames = deque([],maxlen=n_frames)
        
        ball_multiple_frames = deque([],maxlen=n_frames)
        ball_next_multiple_frames = deque([],maxlen=n_frames)
        rewards_running = 0
        done = False
        
        # Ignore ball before passing net
        if ignore_before_net:
            next_state, reward, done = ignore_states_before_net(env)
            
        while not(done):
            while len(multiple_frames) < n_frames-1:
                current_frame = env.get_env_state()
                multiple_frames.append(current_frame)
                ball_multiple_frames.append(get_ball_state_array(current_frame))
                
                action = torch.add(torch.zeros((1, n_actions), device=device), 0.5).float()
                next_state, reward, done = env.step(action.cpu().detach().numpy())
                
                rewards_running += reward
                reward = torch.tensor([reward], device=device)

                next_frame = env.get_env_state()
                next_multiple_frames.append(next_frame)
                ball_next_multiple_frames.append(get_ball_state_array(next_frame))
                

            current_frame = env.get_env_state()
            multiple_frames.append(current_frame)
            ball_multiple_frames.append(get_ball_state_array(current_frame))
            state = torch.tensor(np.array(multiple_frames).flatten()).float().unsqueeze(0).to(device)
            
            # Add future states predictions as additional inputs
            if wm_future_state:
                if wm_type == 'true_dynamics':
                    input_frames = multiple_frames
                elif wm_type == 'learned_dynamics':
                    input_frames = ball_multiple_frames
                    
                pred_next_frames = predict_future_states(input_frames, wm_pred_steps, wm_type, wm_nets)
                future_input = torch.tensor(pred_next_frames.flatten()).float().unsqueeze(0).to(device)
                state = torch.cat((state, future_input), 1)
            
            if pretrain:
                # Minimal exploration when adaptation
                action = noise.get_action(actor_policy_net(state), vary_eps_steps).to(device)
            else:
                action = noise.get_action(actor_policy_net(state), steps).to(device)
            
            next_state, reward, done = env.step(action.cpu().detach().numpy())
            if single_step:
                next_state, reward, done = ignore_states_before_done(env, action)            
            
            rewards_running += reward
            reward = torch.tensor([reward], device=device)
            next_frame = env.get_env_state()
            next_multiple_frames.append(next_frame)
            ball_next_multiple_frames.append(get_ball_state_array(next_frame))
            
            # Observe new state
            if not done:
                next_state = torch.tensor(np.array(next_multiple_frames).flatten()).float().unsqueeze(0).to(device)
                # Add future states predictions as additional inputs
                if wm_future_state:
                    if wm_type == 'true_dynamics':
                        input_frames = next_multiple_frames
                    elif wm_type == 'learned_dynamics':
                        input_frames = ball_next_multiple_frames
          
                    pred_next_frames = predict_future_states(input_frames, wm_pred_steps, wm_type, wm_nets)
                    future_input = torch.tensor(pred_next_frames.flatten()).float().unsqueeze(0).to(device)
                    next_state = torch.cat((next_state, future_input), 1)
            else:
                next_state = None
            
            # Store the transition in memory
            memory.push(state, action, next_state, reward)
            
            running_actor_loss, running_critic_loss = agent.optimize_model(actor_policy_net, actor_target_net, actor_optimizer, actor_scheduler, 
                                       critic_policy_net, critic_target_net, critic_optimizer, critic_scheduler,
                                       memory, BATCH_SIZE, state_dim, tau, GAMMA, running_actor_loss, running_critic_loss, grad_clip)

            

            if steps % 100 == 99:    # print every 2000 mini-batches
                actor_loss_list.append(running_actor_loss / 100)
                critic_loss_list.append(running_critic_loss / 100)
                running_actor_loss = 0.0 
                running_critic_loss = 0.0
            if done:
                total_returns.append(rewards_running)

            steps += 1
        if not(pretrain):
            # Save model every "validation_every" episodes                
            if i_episode % validation_every == validation_every-1:
                print("Evaluuate at episode:", i_episode)
                eval_returns, _ = evaluate(validation_episodes, actor_policy_net, n_frames, state_dim, wm_nets=wm_nets)
                eval_return_array = np.array(eval_returns)
                eval_return_mean = np.mean(eval_return_array, axis=0)
                eval_return_std = np.std(eval_return_array, axis=0)
                eval_return_mean_list.append(eval_return_mean)
                eval_return_std_list.append(eval_return_std)

                if eval_return_mean > best_return:
                    best_return = eval_return_mean
                    torch.save(actor_policy_net.state_dict(), prefix+str(i)+"_actor_model.pth")
                    torch.save(critic_policy_net.state_dict(), prefix+str(i)+"_critic_model.pth")
                    print("Model Saved")
                    print("Current Return:", eval_return_mean)
        else:
            print("Evaluuate at episode:", i_episode)
            eval_returns, _ = evaluate(validation_episodes, actor_policy_net, n_frames, state_dim, wm_nets=wm_nets)
            eval_return_array = np.array(eval_returns)
            eval_return_mean = np.mean(eval_return_array, axis=0)
            eval_return_std = np.std(eval_return_array, axis=0)
            eval_return_mean_list.append(eval_return_mean)
            eval_return_std_list.append(eval_return_std)

            if eval_return_mean > best_return:
                best_return = eval_return_mean
                torch.save(actor_policy_net.state_dict(), prefix+str(i)+"_actor_model.pth")
                torch.save(critic_policy_net.state_dict(), prefix+str(i)+"_critic_model.pth")
                print("Model Saved")
                print("Current Return:", eval_return_mean)        
    
    return total_returns, eval_return_mean_list, eval_return_std_list, actor_loss_list, critic_loss_list

def evaluate(num_eval_episodes, actor_policy_net, n_frames, state_dim, wm_nets=None,
            ball_mass=None, wind_magnitude=None):
    total_returns = []
    train = False
    if env_mode == 'original':
        env = tennis2d.create_tennis2D_env(train, max_timestep, num_skip, continuous_control, 
                                           render_mode, short_eps, noise=inject_obs_noise, task=task,
                                           reward_func=reward_func)
    elif env_mode =='adjusted':
        env = tennis2d.create_tennis2D_env(train, max_timestep, num_skip, continuous_control, 
                                           render_mode, short_eps,
                                           noise=inject_obs_noise, task=task, 
                                           reward_func=reward_func,
                                          ball_mass=ball_mass, wind_magnitude=wind_magnitude)
    for ep in range(num_eval_episodes):
        
        # Randomly selected ball masss and wind magnitude for nonstationary training
        if env_mode == 'original' and multi_params_train:
            rand_param_idx = np.random.choice(num_dmc_params)
            rand_ball_mass, rand_wind_magnitude = dmc_params_list[rand_param_idx]
            env.ball_mass =  rand_ball_mass
            env.wind_magnitude = rand_wind_magnitude  
            
        env.reset()
        multiple_frames = deque([],maxlen=n_frames)
        next_multiple_frames = deque([],maxlen=n_frames)
        
        ball_multiple_frames = deque([],maxlen=n_frames)
        ball_next_multiple_frames = deque([],maxlen=n_frames)
        
        rewards_running = 0
        done = False
        t = 0
        
        # Ignore ball before passing net
        if ignore_before_net:
            next_state, reward, done = ignore_states_before_net(env)
            
        while not(done):
            while len(multiple_frames) < n_frames-1:
                current_frame = env.get_env_state()
                multiple_frames.append(current_frame)
                ball_multiple_frames.append(get_ball_state_array(current_frame))
                
                action = torch.add(torch.zeros((1, n_actions), device=device), 0.5).float()
                next_state, reward, done = env.step(action.cpu().detach().numpy())
                
                rewards_running += reward
                reward = torch.tensor([reward], device=device)

                next_frame = env.get_env_state()
                next_multiple_frames.append(next_frame)
                ball_next_multiple_frames.append(get_ball_state_array(next_frame))
                

            current_frame = env.get_env_state()
            multiple_frames.append(current_frame)
            ball_multiple_frames.append(get_ball_state_array(current_frame))
            state = torch.tensor(np.array(multiple_frames).flatten()).float().unsqueeze(0).to(device)
            
            # Add future states predictions as additional inputs
            if wm_future_state:
                if wm_type == 'true_dynamics':
                    input_frames = multiple_frames
                elif wm_type == 'learned_dynamics':
                    input_frames = ball_multiple_frames
                    
                pred_next_frames = predict_future_states(input_frames, wm_pred_steps, wm_type, wm_nets)
                future_input = torch.tensor(pred_next_frames.flatten()).float().unsqueeze(0).to(device)
                state = torch.cat((state, future_input), 1)
            
            action = actor_policy_net(state).to(device)
            next_state, reward, done = env.step(action.cpu().detach().numpy())
            if single_step:
                next_state, reward, done = ignore_states_before_done(env, action)
                
            rewards_running += reward
            reward = torch.tensor([reward], device=device)
            next_frame = env.get_env_state()
            next_multiple_frames.append(next_frame)
            ball_next_multiple_frames.append(get_ball_state_array(next_frame))
            
            
            # Observe new state
            if not done:
                next_state = torch.tensor(np.array(next_multiple_frames).flatten()).float().unsqueeze(0).to(device)
                # Add future states predictions as additional inputs
                if wm_future_state:
                    if wm_type == 'true_dynamics':
                        input_frames = next_multiple_frames
                    elif wm_type == 'learned_dynamics':
                        input_frames = ball_next_multiple_frames
          
                    pred_next_frames = predict_future_states(input_frames, wm_pred_steps, wm_type, wm_nets)
                    future_input = torch.tensor(pred_next_frames.flatten()).float().unsqueeze(0).to(device)
                    next_state = torch.cat((next_state, future_input), 1)
            else:
                next_state = None
                
            if done:
                # print('agent timestep:', t)
                total_returns.append(rewards_running)
            t=t+1
    
    return total_returns, env

def process(i, params, pretrain=False, ball_mass=None, wind_magnitude=None):
    
    PARAMS_train_returns = []
    print('Start training params:', params)
    learning_rate, tau, GAMMA, expl, ounoise_params, BATCH_SIZE, buffer_size, epsilon, \
    n_frames, size_hidden_layers, num_seed = params
    # Fix seed
    np.random.seed(num_seed)
    torch.manual_seed(num_seed)    
    num_hidden_layers = len(size_hidden_layers)
    state_dim = 13 * n_frames    #x, x_dot, theta, theta_dot
    ball_state_dim = 4
    
    if wm_future_state and wm_type == 'learned_dynamics':
        wm_nets = uwm.get_wm(wm_folder_name, wm_best_idx_dict[wm_loss], wm_loss, n_frames)
    else:
        wm_nets = None
    if wm_future_state:
        state_dim += (ball_state_dim*wm_pred_steps)
    # Create Actor
    actor_policy_net = agent.Actor(state_dim, n_actions, num_hidden_layers, size_hidden_layers).to(device)
    
    if train_mode:
        # Create Critic
        critic_policy_net = agent.Critic(state_dim+n_actions, 1, num_hidden_layers, size_hidden_layers).to(device)
        
        # Load pretrained model
        if pretrain:
            actor_policy_net.load_state_dict(torch.load(prefix_orig+str(i)+"_actor_model.pth"))
            critic_policy_net.load_state_dict(torch.load(prefix_orig+str(i)+"_critic_model.pth"))
        
        # Create target nets as copies of policy nets
        actor_target_net = agent.Actor(state_dim, n_actions, num_hidden_layers, size_hidden_layers).to(device)
        critic_target_net = agent.Critic(state_dim+n_actions, 1, num_hidden_layers, size_hidden_layers).to(device)
        actor_target_net.load_state_dict(actor_policy_net.state_dict())
        critic_target_net.load_state_dict(critic_policy_net.state_dict())
        
        actor_optimizer = torch.optim.Adam(actor_policy_net.parameters(), lr=learning_rate)
        actor_scheduler = torch.optim.lr_scheduler.ConstantLR(actor_optimizer, factor=0.9, total_iters=1e3)                
        critic_optimizer = torch.optim.Adam(critic_policy_net.parameters(), lr=learning_rate)
        critic_scheduler = torch.optim.lr_scheduler.ConstantLR(critic_optimizer, factor=0.9, total_iters=1e3)
        
        
        actor_target_net.eval()
        critic_target_net.eval()
        memory = agent.ReplayBuffer(buffer_size)

        train_returns, eval_return_mean_list, eval_return_std_list,  \
        actor_loss_list, critic_loss_list \
        = train(i,  actor_policy_net,
        actor_target_net, actor_optimizer, actor_scheduler, 
        critic_policy_net, critic_target_net, critic_optimizer, critic_scheduler,
        epsilon, memory, n_frames, BATCH_SIZE, state_dim, tau, GAMMA, expl, ounoise_params, wm_nets=wm_nets)


        train_returns = np.array(train_returns)
        eval_returns_mean = np.array(eval_return_mean_list)
        eval_returns_std = np.array(eval_return_std_list)
        

        np.save(prefix+str(i)+'_train_returns.npy', train_returns)
        np.save(prefix+str(i)+'_eval_returns_mean.npy', eval_returns_mean)
        np.save(prefix+str(i)+'_eval_returns_std.npy', eval_returns_std)
        np.save(prefix+str(i)+'_actor_loss.npy', np.array(actor_loss_list))
        np.save(prefix+str(i)+'_critic_loss.npy', np.array(critic_loss_list))
        

        print('Training Log Saved')
    # Watching mode
    else: 
        print('Watching mode')
        # TODO load actor net
        actor_policy_net.load_state_dict(torch.load(prefix+str(i)+"_actor_model.pth"))
        actor_policy_net.eval()
        if env_mode == 'original':
            num_eval_episodes = 1000
        else:
            num_eval_episodes = 100
        eval_returns = 0
    
            
        eval_returns, env = evaluate(num_eval_episodes, actor_policy_net, n_frames, state_dim,
                                          ball_mass=ball_mass, wind_magnitude=wind_magnitude, wm_nets=wm_nets)
        
        eval_returns_mean = np.mean(np.array(eval_returns))
        eval_returns_std = np.std(np.array(eval_returns))
        
        # print('Mean return:', eval_returns_mean)
        # print('Std return:', eval_returns_std)
        # print('Hit Rate:', env.total_hits['player'] / num_eval_episodes)
        # print('Net Pass Rate:', (env.total_hits['target']+env.total_hits['right']) /
        #                           num_eval_episodes)
        # print('Success Rate:', env.total_hits['target'] / num_eval_episodes)
        hr = env.total_hits['player'] / num_eval_episodes
        npr = (env.total_hits['target']+env.total_hits['right']) / num_eval_episodes
        sr =  env.total_hits['target'] / num_eval_episodes
        ar = eval_returns_mean
        return hr, npr, sr, ar
#         plt.figure(figsize=(20,10))
#         plt.title('Angular Velocity - Target distance')
#         plt.scatter(env.dist_list, env.hit_velocity_list)
#         plt.show()

#         np.save(prefix+str(i)+'_hit_velocity_list.npy', np.array(env.hit_velocity_list))     
#         np.save(prefix+str(i)+'_dist_list.npy', np.array(env.dist_list))
        

if env_mode == 'original':
    if train_mode:
        results = Parallel(n_jobs=min(num_cores, len(PARAMS)))(delayed(process)(i, params) for i, params in enumerate(PARAMS))                
        # for i, params in enumerate(PARAMS):
        #     process(i, params, False)
    else:
        model_hr = []
        model_npr = []
        model_sr = []
        model_ar = []        
        for best_model in best_models:
            hr, npr, sr, ar = process(best_model, best_params)
            model_hr.append(hr)
            model_npr.append(npr)
            model_sr.append(sr)
            model_ar.append(ar)

        model_hr_array = np.array(model_hr)
        model_npr_array = np.array(model_npr)
        model_sr_array = np.array(model_sr)
        model_ar_array = np.array(model_ar)

        np.save(prefix+'orig_model_hr.npy', model_hr_array)
        np.save(prefix+'orig_model_npr.npy', model_npr_array)
        np.save(prefix+'orig_model_sr.npy', model_sr_array)
        np.save(prefix+'orig_model_ar.npy', model_ar_array)

        print('Avg Hit Rate:', np.mean(model_hr_array), np.std(model_hr_array))
        print('Avg Net Pass Rate:', np.mean(model_npr_array), np.std(model_npr_array))
        print('Avg Success Rate:', np.mean(model_sr_array), np.std(model_sr_array))
        print('Avg Return:', np.mean(model_ar_array), np.std(model_ar_array))
                
else:
    num_models = len(best_models)
    model_hr = []
    model_npr = []
    model_sr = []
    model_ar = []
    for best_model in best_models: # model idx does not need to be in order like 0,1,2 (can be 1,3,6)
        print('evaluate model:', best_model)
        running_hr = 0.0
        running_npr = 0.0
        running_sr = 0.0
        running_ar = 0.0
        num_phy_params = len(dmc_params_list)
        for i, env_param in enumerate(dmc_params_list):
            # change value of an adjusted param
            ball_mass, wind_magnitude = env_param
            print('Ball Mass:', ball_mass)
            print('Wind Magnitude:', wind_magnitude)

            hr, npr, sr, ar = process(best_model, best_params, ball_mass=ball_mass, wind_magnitude=wind_magnitude)

            running_hr += hr
            running_npr += npr
            running_sr += sr
            running_ar += ar
        # print('Avg Hit Rate:', running_hr / num_phy_params)
        # print('Avg Net Pass Rate:', running_npr / num_phy_params)
        # print('Avg  Success Rate:', running_sr / num_phy_params)
        # print('Avg Return:', running_ar / num_phy_params)
        
        model_hr.append(running_hr / num_phy_params)
        model_npr.append(running_npr / num_phy_params)
        model_sr.append(running_sr / num_phy_params)
        model_ar.append(running_ar / num_phy_params)

    model_hr_array = np.array(model_hr)
    model_npr_array = np.array(model_npr)
    model_sr_array = np.array(model_sr)
    model_ar_array = np.array(model_ar)
    
    np.save(prefix+'adj_model_hr.npy', model_hr_array)
    np.save(prefix+'adj_model_npr.npy', model_npr_array)
    np.save(prefix+'adj_model_sr.npy', model_sr_array)
    np.save(prefix+'adj_model_ar.npy', model_ar_array)
    
    print('Avg Hit Rate:', np.mean(model_hr_array), np.std(model_hr_array))
    print('Avg Net Pass Rate:', np.mean(model_npr_array), np.std(model_npr_array))
    print('Avg Success Rate:', np.mean(model_sr_array), np.std(model_sr_array))
    print('Avg Return:', np.mean(model_ar_array), np.std(model_ar_array))
    
        
        

