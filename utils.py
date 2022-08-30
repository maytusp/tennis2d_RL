import random
import sys
import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cpu")
import ounoise
import tennis2d
import tennis2d_multi as tennis2d_mrl
import meta_wmnet as wmnet

def ignore_states_before_done(env, action):
    done = False
    while not(done):
        next_state, reward, done = env.step(action.cpu().detach().numpy())
    return next_state, reward, done

def ignore_states_before_net(env, n_actions=2):
    done = False
    while not(env.pass_net) and not(done):
        action = torch.add(torch.zeros((1, n_actions), device=device), 0.5).float()
        next_state, reward, done = env.step(action.cpu().detach().numpy())
    return next_state, reward, done
        
def predict_future_states(multiple_frames, wm_pred_steps=4, wm_type='true_dynamics', 
                          wm_nets=None, n_actions=2): # wm = ['true_dynamics', 'learned_dynamics']
    if wm_type == 'true_dynamics':
        pred_next_frames = []
        init_frame = multiple_frames[-1]
        wm_env = tennis2d.create_tennis2D_env(train=False, max_timestep=150, num_skip=4, 
                                              continuous_control=True, render_mode=False, 
                                              noise=False)
        wm_env.reset(init_frame)
        for step in range(wm_pred_steps):
            pred_next_frame, _, _ = wm_env.step(np.zeros((1, n_actions))+0.5)
            pred_next_frame = pred_next_frame[6:10]
            pred_next_frames.append(pred_next_frame)
            
        pred_next_frames = np.array(pred_next_frames)
        return pred_next_frames
        
    elif wm_type == 'learned_dynamics':
        pred_next_frames = np.zeros((wm_pred_steps, 4))
        # temp_multiple_frames is used for multi-step prediction
        # It keeps current prediction to be used as input for further predictions
        temp_multiple_frames = multiple_frames.copy()
        for time_step in range(wm_pred_steps):
            # For Multi-steps prediction
            if time_step == 0:
                current_frame = temp_multiple_frames[-1]
                current_frame = current_frame[:4]
            else:
                current_frame = np.squeeze(pred_next_frame)
                temp_multiple_frames.append(current_frame)
                
            temp_multiple_frames_array = np.array(temp_multiple_frames)
            state = torch.tensor(temp_multiple_frames_array.flatten()).float().unsqueeze(0).to(device)
            knowledge_net = wm_nets['knowledge_net']
            param_net = wm_nets['param_net']
            prediction_net = wm_nets['prediction_net']
            
            pred_next_frame, _ = wmnet.meta_wm_prediction(knowledge_net, param_net, 
                                                          prediction_net, state, contrastive=True)
            pred_next_frame = pred_next_frame.cpu().detach().numpy() + current_frame
            
            pred_next_frames[time_step] = pred_next_frame
        
        return pred_next_frames
    
def predict_future_states_mrl(multiple_frames, wm_pred_steps=4, wm_type='true_dynamics', 
                          wm_nets=None, n_actions=2, num_agent=2): # wm = ['true_dynamics', 'learned_dynamics']
    if wm_type == 'true_dynamics':
        pred_next_frames = []
        init_frame = multiple_frames[-1]
        wm_env = tennis2d_mrl.create_tennis2D_env(train=False, max_timestep=200, num_skip=4, 
                                              continuous_control=True, render_mode=False, 
                                              noise=False, num_agents=num_agent)
        wm_env.reset(init_frame)
        actions = [torch.add(torch.zeros((1, n_actions), device=device), 0.5).float() for i in range(num_agent)]        
        for step in range(wm_pred_steps):
            
            pred_next_frame, _, _ = wm_env.step(actions)
            pred_next_frame = wm_env.get_ball_state(pred_next_frame)
            pred_next_frames.append(pred_next_frame)
            
        pred_next_frames = np.array(pred_next_frames)
        return pred_next_frames
        
    elif wm_type == 'learned_dynamics':
        pred_next_frames = np.zeros((wm_pred_steps, 4))
        # temp_multiple_frames is used for multi-step prediction
        # It keeps current prediction to be used as input for further predictions
        temp_multiple_frames = multiple_frames.copy()
        for time_step in range(wm_pred_steps):
            # For Multi-steps prediction
            if time_step == 0:
                current_frame = temp_multiple_frames[-1]
                current_frame = current_frame[:4]
            else:
                current_frame = np.squeeze(pred_next_frame)
                temp_multiple_frames.append(current_frame)
                
            temp_multiple_frames_array = np.array(temp_multiple_frames)
            state = torch.tensor(temp_multiple_frames_array.flatten()).float().unsqueeze(0).to(device)
            knowledge_net = wm_nets['knowledge_net']
            param_net = wm_nets['param_net']
            prediction_net = wm_nets['prediction_net']
            
            pred_next_frame, _ = wmnet.meta_wm_prediction(knowledge_net, param_net, 
                                                          prediction_net, state, contrastive=True)
            pred_next_frame = pred_next_frame.cpu().detach().numpy() + current_frame
            
            pred_next_frames[time_step] = pred_next_frame
        
        return pred_next_frames
    
def find_top5_model(prefix, model_idx_list):
    model_scores = []
    for i in model_idx_list:
        eval_returns_mean = np.load(prefix+str(i)+'_eval_returns_mean.npy')
        eval_returns_std = np.load(prefix+str(i)+'_eval_returns_std.npy')
        best_ep = np.argmax(eval_returns_mean)
        model_scores.append(eval_returns_mean[best_ep])
    
    # sort model from best to worst
    model_scores = np.array(model_scores)
    sorted_model_idx = np.argsort(model_scores)[::-1]
    best_5_idx = sorted_model_idx[:5]
    print('Best scores:', model_scores[best_5_idx])
    print('Best idx:', best_5_idx)
    return model_scores[best_5_idx], best_5_idx