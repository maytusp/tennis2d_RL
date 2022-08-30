from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
import random
import sys
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
import agent
import meta_wmnet as wmnet
import uninoise
import tennis2d
matplotlib.rcParams.update({'font.size': 22})
# World Model parameters
EXP_SIZES = [int(1e5)]
WM_EPOCHS = int(2e2)
WM_ADAPT_EPOCHS = 100
WM_BATCH_SIZES = [128]
WM_ADAPT_BATCH_SIZE = 32
WM_LRS = [1e-4]
KN_HIDDEN_SIZES = [[256,128]]
PARAM_HIDDEN_SIZES = [[256,128]]
PRED_HIDDEN_SIZES = [[128,64]]
PROJ_HIDDEN_SIZES = [[64,32]]
model_name_list = ['knowledge_net', 'param_net', 'prediction_net', 'projection_net']
knowledge_latent_dim = 64
param_latent_dim = 32
projection_latent_dim = 16
WM_N_FRAMES = [4]
TEMPERATURES = [0.8]
CON_LOSS_WEIGHTS = [1]
NUM_SEEDS = [0,1,2,3,4,5,6,7,8,9]




# Deep RL parameters
validation_episodes = 30
continuous_control = True
n_actions = 2
grad_clip = True
num_skip = 4 # 1 means no frame skipping
max_timestep = int(360 // num_skip) # 6 seconds per episode
task  = 'mrl' # counter (None), smash, multi-agent
folder_name = "results_meta_wm/"
if task == 'smash':
    folder_name = 'results_meta_wm_smash/'
elif task == 'mrl':
    folder_name = 'results_meta_wm_mrl/'
n_frames = 4

ball_mass_list_train = [9,9.5,10,10.5,11]
wind_list_train = [-7.5, -5, 0, 5, 7.5]

dmc_params_list_train = list(product(ball_mass_list_train, wind_list_train))

ball_mass_list_inter = [9.75, 10.25]
wind_list_inter = [-2.5, 2.5]
dmc_params_list_inter = list(product(ball_mass_list_inter, wind_list_inter))

ball_mass_list_extra = [8.5, 11.5]
wind_list_extra = [-10, 10]
dmc_params_list_extra = list(product(ball_mass_list_extra, wind_list_extra))

dmc_params_list_eval =  dmc_params_list_train + dmc_params_list_inter + dmc_params_list_extra
n_dataset = len(dmc_params_list_train)

# Run mode
run_modes = ['train', 'adapt', 'evaluate']
evaluate_mode = 'original'
run_mode = run_modes[0]
evaluate_path = True
pretrain = False
render_mode = False
shorter_eps = False
train_wm = True
inject_obs_noise = False
contrastive = False
sim_func = ''
use_ignore_state = False
best_param_idx = 17
best_params = 0.0001, 32, 100000, 4, [512, 256, 256, 128, 128, 64], [128, 32], 1, 0.01

if contrastive:
    prefix = folder_name + "wm_con_"
    loss_name_list = ['prediction', 'contrastive']
    if sim_func == 'l2':
        prefix += "l2_"
else:
    prefix = folder_name + "wm_"
    loss_name_list = ['prediction']
if n_dataset == 1:
    prefix += 'single_param_'
    
    
def ignore_states_before_net(env):
    done = False
    while not(env.pass_net) and not(done):
        action = torch.add(torch.zeros((1, n_actions), device=device), 0.5).float()
        next_state, reward, done = env.step(action.cpu().detach().numpy())
    return next_state, reward, done

if not os.path.exists(folder_name):
    os.makedirs(folder_name)

WM_PARAMS = list(product(WM_LRS, WM_BATCH_SIZES, EXP_SIZES, WM_N_FRAMES, KN_HIDDEN_SIZES, PARAM_HIDDEN_SIZES, 
                         PRED_HIDDEN_SIZES, PROJ_HIDDEN_SIZES, TEMPERATURES, CON_LOSS_WEIGHTS, NUM_SEEDS))
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def train_wm(wm_nets, wm_optimizer, wm_scheduler, state_dim, n_frames, 
            batch_size, exp_size, temp, k_ce, collect_data=False, policy_net=None, policy='random', i=0):
    zero_value_list = len(loss_name_list) * [0.0]
    best_value_list = len(loss_name_list) * [1e6]
    
    dynamics_loss_list = []
    val_dynamics_loss_list = []
    best_dynamics_loss = {loss_name_list[i]: best_value_list[i] for i in range(len(loss_name_list))}
    steps = 0
    meta_train_memory = {}
    meta_val_test_memory = {}
    train_val_test_ratio = 0.8
    min_num_train = int(1e7)
    min_num_val_test = int(1e7)
    # Load N datasets
    for dmc_idx, dmc_params in enumerate(dmc_params_list_train):
        ball_mass, wind_magnitude = dmc_params
        
        # Load train memory
        exp_filename = folder_name + 'wm_dmc'+str(dmc_idx)+'_dataset.pkl'
        loaded_train_memory = pickle.load(open(exp_filename, 'rb+'))
        loaded_train_memory = loaded_train_memory.memory
        num_train = len(loaded_train_memory)
        random.shuffle(loaded_train_memory)
        
        # Load val memory
        val_test_exp_filename = folder_name + 'wm_val_dmc'+str(dmc_idx)+'_dataset.pkl'
        loaded_val_test_memory = pickle.load(open(val_test_exp_filename, 'rb+'))
        loaded_val_test_memory = loaded_val_test_memory.memory
        num_val_test = len(loaded_val_test_memory)
        random.shuffle(loaded_val_test_memory)        
        
        
        
        train_memory = wmnet.Experience()
        val_test_memory = wmnet.Experience()
        train_memory.memory = loaded_train_memory.copy()
        val_test_memory.memory = loaded_val_test_memory.copy()
        
        # Pack dataset from each physcial parameter to meta dataset
        meta_train_memory[str(dmc_idx)] = train_memory
        meta_val_test_memory[str(dmc_idx)] = val_test_memory
        
        if num_train < min_num_train:
            min_num_train = num_train
            print('min num train:', num_train)
            
        if num_val_test < min_num_val_test:
            min_num_val_test = num_val_test
            print('min num val test:', num_val_test)
        
    step_range = min_num_train // n_sample_per_dataset
    steps = 0
    
    # Ball State predictions
    next_frame_idx = (n_frames-1) * (state_dim // n_frames)
    
    for epoch in range(WM_EPOCHS):
        running_dynamics_loss = {loss_name_list[i]: zero_value_list[i] for i in range(len(loss_name_list))}
        print('epoch:', epoch)
        for batch_num in range(step_range):
            running_dynamics_loss = \
            wmnet.optimize_model(meta_train_memory, wm_nets, wm_optimizer, 
                                 wm_scheduler, batch_size, state_dim, n_frames, 
                                 next_frame_idx, running_dynamics_loss, grad_clip, step_range, temp, k_ce,
                                 n_dataset=n_dataset, n_sample_per_dataset=n_sample_per_dataset, 
                                 contrastive=contrastive, sim_func=sim_func)
            steps += 1
            
        dynamics_loss_list.append(running_dynamics_loss)
        val_dynamics_loss = compute_validation_loss(meta_val_test_memory, min_num_val_test, 
                                                    wm_nets, wm_optimizer, 
                                                    wm_scheduler, batch_size, state_dim, temp, k_ce,
                                                    n_frames, next_frame_idx, 
                                                    loss_name_list, zero_value_list)
        val_dynamics_loss_list.append(val_dynamics_loss)
        
        if val_dynamics_loss['prediction'] <  best_dynamics_loss['prediction']:
            
            best_dynamics_loss['prediction'] = val_dynamics_loss['prediction']
            saved_knowledge_net = wm_nets['knowledge_net']
            saved_param_net = wm_nets['param_net']
            saved_prediction_net = wm_nets['prediction_net']
            saved_projection_net = wm_nets['projection_net']
            
            torch.save(saved_knowledge_net.state_dict(), prefix+str(i)+"_knowledge_net.pth")
            torch.save(saved_param_net.state_dict(), prefix+str(i)+"_param_net.pth")
            torch.save(saved_prediction_net.state_dict(), prefix+str(i)+"_prediction_net.pth")
            torch.save(saved_projection_net.state_dict(), prefix+str(i)+"_projection_net.pth")
            print('Saved')
            
            
    return dynamics_loss_list, val_dynamics_loss_list

def adapt_wm(wm_nets, wm_optimizer, wm_scheduler, state_dim, n_frames, 
            batch_size, exp_size, temp, k_ce, i=0):
    zero_value_list = len(loss_name_list) * [0.0]
    best_value_list = len(loss_name_list) * [1e6]
    
    dynamics_loss_list = []
    val_dynamics_loss_list = []
    best_dynamics_loss = {loss_name_list[i]: best_value_list[i] for i in range(len(loss_name_list))}
    steps = 0
    meta_train_memory = {}
    meta_val_test_memory = {}
    train_val_test_ratio = 0.8
    min_num_train = int(1e7)
    min_num_val_test = int(1e7)
    # Load N datasets
    for dmc_idx, dmc_params in enumerate(dmc_params_list_extra):
        ball_mass, wind_magnitude = dmc_params
        exp_filename = folder_name + 'adapt_wm_dmc'+str(dmc_idx)+'_dataset.pkl'
        memory = pickle.load(open(exp_filename, 'rb+'))
   
        memory = memory.memory
        num_data = len(memory)
        num_train = int(train_val_test_ratio * num_data)
        random.shuffle(memory)
        

        train_memory = wmnet.Experience()
        val_test_memory = wmnet.Experience()
        train_memory.memory = memory.copy()
        meta_train_memory[str(dmc_idx)] = train_memory
        meta_val_test_memory[str(dmc_idx)] = val_test_memory
        
        if num_train < min_num_train:
            min_num_train = num_train
            print('min num train:', num_train)
        
    step_range = min_num_train // n_sample_per_dataset
    steps = 0
    
    # Ball State predictions
    next_frame_idx = (n_frames-1) * (state_dim // n_frames)
    
    for epoch in range(WM_ADAPT_EPOCHS):
        running_dynamics_loss = {loss_name_list[i]: zero_value_list[i] for i in range(len(loss_name_list))}
        print('epoch:', epoch)
        for batch_num in range(step_range):
            running_dynamics_loss = \
            wmnet.optimize_model(meta_train_memory, wm_nets, wm_optimizer, 
                                 wm_scheduler, batch_size, state_dim, n_frames, 
                                 next_frame_idx, running_dynamics_loss, grad_clip, step_range, temp, k_ce,
                                 n_dataset=n_dataset, n_sample_per_dataset=n_sample_per_dataset, 
                                 contrastive=contrastive, sim_func=sim_func, adaptation=True)
            steps += 1
        dynamics_loss_list.append(running_dynamics_loss)


    saved_knowledge_net = wm_nets['knowledge_net']
    saved_param_net = wm_nets['param_net']
    saved_prediction_net = wm_nets['prediction_net']
    saved_projection_net = wm_nets['projection_net']
    torch.save(saved_knowledge_net.state_dict(), prefix+str(i)+"_adapt_knowledge_net.pth")
    torch.save(saved_param_net.state_dict(), prefix+str(i)+"_adapt_param_net.pth")
    torch.save(saved_prediction_net.state_dict(), prefix+str(i)+"_adapt_prediction_net.pth")
    torch.save(saved_projection_net.state_dict(), prefix+str(i)+"_adapt_projection_net.pth")

    return dynamics_loss_list


def compute_validation_loss(meta_val_test_memory, min_num_val_test, wm_nets, wm_optimizer, 
                            wm_scheduler, batch_size, state_dim, temp, k_ce,
                            n_frames, next_frame_idx, loss_name_list, zero_value_list):
    running_dynamics_loss = {loss_name_list[i]: zero_value_list[i] for i in range(len(loss_name_list))}
    step_range = min_num_val_test // n_sample_per_dataset
    for batch_num in range(step_range):
        running_dynamics_loss, _ = wmnet.compute_loss(meta_val_test_memory, wm_nets, wm_optimizer, 
                                                      wm_scheduler, batch_size, state_dim, 
                                                      n_frames, next_frame_idx, 
                                                      running_dynamics_loss, grad_clip, 
                                                      step_range, temp, k_ce, is_train=False,
                                                      n_dataset=n_dataset, 
                                                      n_sample_per_dataset=n_sample_per_dataset, 
                                                      contrastive=contrastive, sim_func=sim_func)
    return running_dynamics_loss
    
def visualise_predicted_path(wm_nets, ball_mass, wind_magnitude, dmc_idx, frame_dim=4, num_visualise_episodes=100, i=0):
    train = True
    env = tennis2d.create_tennis2D_env(train, max_timestep, num_skip, continuous_control, 
                                       render_mode, shorter_eps, train_wm, 
                                       noise=inject_obs_noise, 
                                       ball_mass=ball_mass,
                                      wind_magnitude=wind_magnitude)
    true_frames = np.zeros((num_visualise_episodes, max_timestep, frame_dim))
    pred_frames = np.zeros((num_visualise_episodes, max_timestep, frame_dim))
    multi_pred_frames = np.zeros((num_visualise_episodes, max_timestep, frame_dim))
    true_current_frames = np.zeros((num_visualise_episodes, max_timestep, frame_dim))
    total_returns = []
    knowledge_net = wm_nets['knowledge_net']
    param_net = wm_nets['param_net']
    prediction_net = wm_nets['prediction_net']
    for ep in range(num_visualise_episodes):
        env.reset()
        noise = uninoise.UniNoise(n_actions)
        multiple_frames = deque([],maxlen=n_frames)
        multi_multiple_frames = deque([],maxlen=n_frames)
        rewards_running = 0
        time_step = 0
        done = False
        if use_ignore_state:
            next_state, reward, done = ignore_states_before_net(env)
        while not(done):
            while len(multiple_frames) < n_frames-1:
                
                current_frame = wmnet.get_ball_state_array(env.get_env_state(), train_wm)
                multiple_frames.append(current_frame)
                multi_multiple_frames.append(current_frame)

                action = torch.add(torch.zeros((1, n_actions), device=device), 0.5).float()
                next_state, reward, done = env.step(action.cpu().detach().numpy())
                
                rewards_running += reward
                reward = torch.tensor([reward], device=device)

            # For Multi-steps prediction
            if time_step == 0:
                multi_current_frame = wmnet.get_ball_state_array(env.get_env_state(), train_wm)
            else:
                multi_current_frame = np.squeeze(multi_pred_next_frame)
            
            multi_multiple_frames.append(multi_current_frame)
            multi_state = torch.tensor(np.array(multi_multiple_frames).flatten()).float().unsqueeze(0).to(device)
            multi_pred_next_frame, _ = wmnet.meta_wm_prediction(knowledge_net, param_net, prediction_net, multi_state, contrastive)
            multi_pred_next_frame = multi_pred_next_frame.cpu().detach().numpy() + multi_current_frame
            # For signle-step prediction
            current_frame = wmnet.get_ball_state_array(env.get_env_state(), train_wm)
            multiple_frames.append(current_frame)
            
            state = torch.tensor(np.array(multiple_frames).flatten()).float().unsqueeze(0).to(device)
            action = torch.add(torch.zeros((1, n_actions), device=device), 0.5).float()

            # Single step prediction
            pred_next_frame, _ = wmnet.meta_wm_prediction(knowledge_net, param_net, prediction_net, state, contrastive)
            pred_next_frame = pred_next_frame.cpu().detach().numpy()
            # TODO Edit output of pred_next_frame
            pred_next_frame = np.squeeze(pred_next_frame, 0)
            
            _, reward, done = env.step(action.cpu().detach().numpy())
            rewards_running += reward

            
            reward = torch.tensor([reward], device=device)
            next_frame = wmnet.get_ball_state_array(env.get_env_state(), train_wm)
            next_frame_clean = wmnet.get_ball_state_array(env.get_env_state(clean_obs=True), train_wm)

            true_frames[ep, time_step] = next_frame_clean
            pred_frames[ep, time_step] = pred_next_frame
            multi_pred_frames[ep, time_step] = multi_pred_next_frame
            true_current_frames[ep, time_step] = current_frame
            time_step += 1

    saved_path = prefix+str(i)+"_dmc_"+str(dmc_idx)+"_"
    if inject_obs_noise:
        saved_path += 'noisy_'    
    np.save(saved_path+"true_frames.npy", true_frames)
    np.save(saved_path+"pred_frames.npy", pred_frames)
    np.save(saved_path+"multi_pred_frames.npy", multi_pred_frames)
    np.save(saved_path+"true_current_frames.npy", true_current_frames)

def get_prediction_loss_from_pkl(loss):
    ball_loss = []
    for idx in range(len(loss)):
        ball_loss.append(loss[idx]['prediction'])
    return ball_loss

def get_contrastive_loss_from_pkl(loss):
    con_loss = []
    for idx in range(len(loss)):
         con_loss.append(loss[idx]['contrastive'])
    return con_loss


for i, params in enumerate(WM_PARAMS):
    PARAMS_train_returns = []
    print('Start training params:', i, params)
    learning_rate, batch_size, exp_size, n_frames, knowledge_size_hidden_layers, param_size_hidden_layers, \
                                     prediction_size_hidden_layers, projection_size_hidden_layers, temp, k_ce, num_seed = params
    
    # Fix seed
    np.random.seed(num_seed)
    torch.manual_seed(num_seed)
    
    
    knowledge_num_hidden_layers = len(knowledge_size_hidden_layers)
    param_num_hidden_layers = len(param_size_hidden_layers)
    prediction_num_hidden_layers = len(prediction_size_hidden_layers)
    projection_num_hidden_layers = len(projection_size_hidden_layers)
    
    n_sample_per_dataset = int(batch_size // n_dataset)
    state_dim = 16 * n_frames
    
    # Each frame contain
    player_state_dim = 4
    ball_state_dim = 4
    ball_player_int_dim = 1
    ball_map_int_dim = 5
    

    knowledge_net_input_dim =  ball_state_dim * n_frames
    knowledge_net_output_dim = knowledge_latent_dim
    
    param_net_input_dim = ball_state_dim * n_frames
    param_net_output_dim = param_latent_dim
    
    prediction_net_input_dim = knowledge_net_output_dim + param_net_output_dim
    prediction_net_output_dim = ball_state_dim
    
    projection_net_input_dim = param_net_output_dim
    projection_net_output_dim = projection_latent_dim
    
    knowledge_net = wmnet.knowledge_net(knowledge_net_input_dim, knowledge_net_output_dim, 
                                  knowledge_num_hidden_layers, knowledge_size_hidden_layers).to(device)

    
    # NN that trying to extract physical parameters
    param_net = wmnet.param_net(param_net_input_dim, param_net_output_dim, 
                              param_num_hidden_layers, param_size_hidden_layers).to(device)
    
    prediction_net = wmnet.prediction_net(prediction_net_input_dim, prediction_net_output_dim, 
                              prediction_num_hidden_layers, prediction_size_hidden_layers).to(device)
    
    projection_net = wmnet.projection_net(projection_net_input_dim, projection_net_output_dim, 
                              projection_num_hidden_layers, projection_size_hidden_layers).to(device)    
    
    wm_nets = {'knowledge_net':knowledge_net, 'param_net': param_net, 
               'prediction_net':prediction_net, 'projection_net':projection_net}
    
    if run_mode=='train':

        knowledge_optimizer = torch.optim.Adam(knowledge_net.parameters(), lr=learning_rate)
        knowledge_scheduler = torch.optim.lr_scheduler.ConstantLR(knowledge_optimizer, factor=0.1, total_iters=2e5)        
        
        param_optimizer = torch.optim.Adam(param_net.parameters(), lr=learning_rate)
        param_scheduler = torch.optim.lr_scheduler.ConstantLR(param_optimizer, factor=0.1, total_iters=2e5)  
        
        prediction_optimizer = torch.optim.Adam(prediction_net.parameters(), lr=learning_rate)
        prediction_scheduler = torch.optim.lr_scheduler.ConstantLR(prediction_optimizer, factor=0.1, total_iters=2e5)                

        projection_optimizer = torch.optim.Adam(projection_net.parameters(), lr=learning_rate)
        projection_scheduler = torch.optim.lr_scheduler.ConstantLR(projection_optimizer, factor=0.1, total_iters=2e5) 
        
        memory = wmnet.Experience()
        
        
    
        wm_optimizer = {'knowledge_net':knowledge_optimizer, 'param_net':param_optimizer, 
                        'prediction_net':prediction_optimizer, 'projection_net':projection_optimizer}
                  
        wm_scheduler = {'knowledge_net':knowledge_scheduler, 'param_net':param_scheduler, 
                        'prediction_net':prediction_scheduler, 'projection_net':projection_scheduler}

        
        dynamics_loss_list, val_dynamics_loss_list = train_wm(wm_nets, wm_optimizer, 
                                                            wm_scheduler, 
                                                            state_dim, n_frames, 
                                                            batch_size, exp_size, temp, k_ce, 
                                                            collect_data=False,
                                                            policy_net=None, 
                                                            policy='random', i=i)
        dynamic_loss_file = open(prefix + str(i) + "_"+ 'dynamics_loss.pkl', 'wb+')
        val_dynamic_loss_file = open(prefix + str(i) + "_"+ 'val_dynamics_loss.pkl', 'wb+')
        pickle.dump(dynamics_loss_list, dynamic_loss_file)
        pickle.dump(val_dynamics_loss_list, val_dynamic_loss_file)
        
        print('Training Log Saved')
        
    elif run_mode == 'adapt':
        
        prefix2 = prefix + str(i) + "_"
        knowledge_net.load_state_dict(torch.load(prefix2+ "knowledge_net.pth"))
        param_net.load_state_dict(torch.load(prefix2+ "param_net.pth"))
        prediction_net.load_state_dict(torch.load(prefix2+ "prediction_net.pth"))
        projection_net.load_state_dict(torch.load(prefix2 + "projection_net.pth"))
        
        knowledge_optimizer = torch.optim.Adam(knowledge_net.parameters(), lr=learning_rate*0.1)
        knowledge_scheduler = torch.optim.lr_scheduler.ConstantLR(knowledge_optimizer, factor=0.1, total_iters=2e5)        
        
        param_optimizer = torch.optim.Adam(param_net.parameters(), lr=learning_rate*0.1)
        param_scheduler = torch.optim.lr_scheduler.ConstantLR(param_optimizer, factor=0.1, total_iters=2e5)  
        
        prediction_optimizer = torch.optim.Adam(prediction_net.parameters(), lr=learning_rate*0.1)
        prediction_scheduler = torch.optim.lr_scheduler.ConstantLR(prediction_optimizer, factor=0.1, total_iters=2e5)                

        projection_optimizer = torch.optim.Adam(projection_net.parameters(), lr=learning_rate*0.1)
        projection_scheduler = torch.optim.lr_scheduler.ConstantLR(projection_optimizer, factor=0.1, total_iters=2e5) 
        
        memory = wmnet.Experience()
        
        
    
        wm_optimizer = {'knowledge_net':knowledge_optimizer, 'param_net':param_optimizer, 
                        'prediction_net':prediction_optimizer, 'projection_net':projection_optimizer}
                  
        wm_scheduler = {'knowledge_net':knowledge_scheduler, 'param_net':param_scheduler, 
                        'prediction_net':prediction_scheduler, 'projection_net':projection_scheduler}

        
        dynamics_loss_list = adapt_wm(wm_nets, wm_optimizer, 
                                        wm_scheduler, 
                                        state_dim, n_frames, 
                                        batch_size, exp_size, temp, k_ce, 
                                       i=i)
        
        dynamic_loss_file = open(prefix + str(i) + "_"+ 'adapt_dynamics_loss.pkl', 'wb+')
        pickle.dump(dynamics_loss_list, dynamic_loss_file)
        
        print('Adaptation Log Saved')

    elif run_mode=='evaluate':
        if evaluate_mode == 'original':
            print("Evaluate non-adapted models")
            prefix2 = prefix + str(i) + "_"
            knowledge_net.load_state_dict(torch.load(prefix2+ "knowledge_net.pth"))
            param_net.load_state_dict(torch.load(prefix2+ "param_net.pth"))
            prediction_net.load_state_dict(torch.load(prefix2+ "prediction_net.pth"))
            dynamic_loss_file = pickle.load(open(prefix2 + 'dynamics_loss.pkl', 'rb+'))
            val_dynamic_loss_file = pickle.load(open(prefix2 + 'val_dynamics_loss.pkl', 'rb+'))            
        elif evaluate_mode =='adapt':
            print("Evaluate adapted models")
            prefix2 = prefix + str(i) + "_"
            knowledge_net.load_state_dict(torch.load(prefix2+ "adapt_knowledge_net.pth"))
            param_net.load_state_dict(torch.load(prefix2+ "adapt_param_net.pth"))
            prediction_net.load_state_dict(torch.load(prefix2+ "adapt_prediction_net.pth"))
            dynamic_loss_file = pickle.load(open(prefix2 + 'adapt_dynamics_loss.pkl', 'rb+'))
            
            
            
        wm_nets = {'knowledge_net':knowledge_net, 'param_net': param_net, 'prediction_net':prediction_net}
        
        
        prediction_loss_list = get_prediction_loss_from_pkl(dynamic_loss_file)
        if evaluate_mode == 'original':
            val_prediction_loss_list = get_prediction_loss_from_pkl(val_dynamic_loss_file)
            print("Best Epoch:", np.argmin(val_prediction_loss_list))
            print("Best MSE:", np.min(val_prediction_loss_list))     
            
        x1 = [i for i in range(len(prediction_loss_list))]
        
        plt.figure(figsize=(20,10))
        plt.title('Prediction Loss')
        # plt.ylim(0.205,0.23)
        plt.plot(x1, prediction_loss_list, label='Training')
        
        if evaluate_mode == 'original':
            plt.plot(x1, val_prediction_loss_list, label='Validation')
            
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.show()
        
        if contrastive:
            contrastive_loss_list = get_contrastive_loss_from_pkl(dynamic_loss_file)
            if evaluate_mode == 'original':
                val_contrastive_loss_list = get_contrastive_loss_from_pkl(val_dynamic_loss_file)
                print("Worst Contrastive Epoch:", np.argmin(val_contrastive_loss_list))
                print("Worst Contrastive Loss:", np.min(val_contrastive_loss_list))                  
            x2 = [i for i in range(len(contrastive_loss_list))]
            plt.figure(figsize=(20,10))
            plt.title('Contrastive Loss')
            # plt.ylim(0.205,0.23)
            plt.plot(x1, contrastive_loss_list, label='Training')
            if evaluate_mode == 'original':
                plt.plot(x1, val_contrastive_loss_list, label='Validation')
            plt.xlabel("Epoch")
            plt.ylabel("Contrastive Loss")
            plt.legend()
            plt.show()        
            
        if evaluate_path:
            for dmc_idx, dmc_params in enumerate(dmc_params_list_eval):
                ball_mass, wind_magnitude = dmc_params
                visualise_predicted_path(wm_nets, ball_mass, wind_magnitude, dmc_idx, frame_dim=ball_state_dim, i=i)
            print('Evaluate Path Done')


