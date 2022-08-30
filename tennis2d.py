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


# Constant
pi = 3.141592653
width, height = 800, 600
norm_dist = np.sqrt(width**2 + height**2)
fps = 60
collision_types = {
    "ball": 1,
    "player": 2,
    "target": 3
}

# Parameters
# Player
init_player_position = 120, 150
bat_mass = 1e4
bat_length = 80
bat_thick = 10
bat_params = [(0, 0), (bat_length, 0), bat_thick]
bat_moment = pymunk.moment_for_segment(bat_mass, bat_params[0], bat_params[1], bat_params[2])
# player_body_initial_positions_original = [150, 160,170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 
#                                         280, 290, 300, 310, 320, 330, 340, 350, 360]
player_body_initial_positions = [115,120,125]
# Ball
# spawn_ball_params = (width-80, 80), [(-1e4, -1e4), (1, 2000)]
force_const = 1e4
# Original
# spawn_ball_params= [[[(260,500), (261,500)], [(110/180)*pi, (120/180)*pi], [0,0.1]]]
spawn_ball_params = {}
spawn_ball_params['serve'] = [[[(init_player_position[0]+15, 200), (init_player_position[0]+20, 200)], [(90/180)*pi, (91/180)*pi], [1,1.2]]]
spawn_ball_params['counter'] = [[[(600, 260), (650, 260)], [(190/180)*pi, (195/180)*pi], [3,4]]]
spawn_ball_params['smash'] = [[[(600, 100), (650, 100)], [(130/180)*pi, (140/180)*pi], [5,6]]]
spawn_ball_params['wm'] = [[[(600, 260), (650, 260)], [(185/180)*pi, (195/180)*pi], [2,4]]]
# spawn_ball_params['mrl'] = [[[(680, 160), (700, 160)], [(120/180)*pi, (140/180)*pi], [1.5,2.2]],
#                            [[(100, 160), (120, 160)], [(30/180)*pi, (50/180)*pi], [1.5,2.2]]]                   
ball_radius = 10
ball_mass = 10
ball_moment = pymunk.moment_for_circle(mass=ball_mass, inner_radius=0, outer_radius=ball_radius)

# Target
target_speed = 0

# Gravity
gravity_const = 1000

# Drag
drag_constant = 5e-4

# Wind
wind_magnitude = 0

player_force = 1e7
player_torque = 2.5e9

max_speed = 400
max_ang_speed = 12
render_speed_up = 2


class create_tennis2D_env:
    def __init__(self, train, max_timestep, num_skip, continuous_control, render_mode, short_eps=False, train_wm=False, noise=False, relative_vec=False, ball_mass=ball_mass,
                 drag_constant=drag_constant,
                 gravity_const=gravity_const,
                 spawn_ball_params=spawn_ball_params,
                 wind_magnitude=wind_magnitude, task='counter', reward_func='rbf'):
        self.ball_shape = None
        self.ball_body = None
        self.player_shape = None
        self.player_body = None
        self.player_joint_body = None
        self.target_shape = None
        self.target_body = None
        self.move_joint = None
        self.space = None
        self.player = None
        self.static_lines = None
        self.target_line = None
        self.ball_player_hit = None
        self.ball_left_hit = None
        self.ball_ceiling_hit = None
        self.ball_right_hit = None
        self.ball_ground_hit = None
        self.ball_net_hit = None
        self.after_hit = None
        self.state_dim = None
        self.target_angle = None
        self.control_angular_velocity = None
        self.hits = None
        self.start_time = None
        self.clock = None
        self.draw_options = None
        self.font = None
        self.screen = None
        self.t = None
        self.pass_net = None
        self.save_speed_dist_corr = True
        self.noise = noise
        self.short_eps = short_eps
        self.relative_vec = relative_vec
        self.train_wm = train_wm
        self.total_hits = {'player':0, 'right':0, 'target': 0}
        
        self.max_speed = max_speed
        self.max_ang_speed = max_ang_speed
        self.norm_dist = norm_dist

        self.force_list = []
        self.torque_list = []
        self.dist_list = []
        self.hit_velocity_list = []
        self.hit_angle_list = []
        self.achievement_list = []
        
        self.render_mode = render_mode
        self.max_timestep = max_timestep
        self.train = train
        self.continuous_control = continuous_control
        self.num_skip = num_skip
        
        # physical parameters to be adjusted
        self.ball_mass = ball_mass
        self.ball_moment = pymunk.moment_for_circle(mass=self.ball_mass, inner_radius=0, outer_radius=ball_radius)
        self.drag_constant = drag_constant
        self.gravity_const = gravity_const
        self.wind_magnitude = wind_magnitude
        self.spawn_ball_params = spawn_ball_params[task]
        self.task = task
        self.reward_func = reward_func
        self.angles = {}
        self.angles['serve'] = [(30/180) * pi, (185/180) * pi]
        self.angles['counter'] = [(60/180) * pi, (370/180) * pi]
        self.angles['smash'] = [(30/180) * pi, (185/180) * pi]
        # self.angles['mrl'] = [(60/180) * pi, (370/180) * pi]

        self.min_angle = self.angles[task][0]
        self.max_angle = self.angles[task][1]
        if not(train_wm):
            if task == 'smash' and wind_magnitude < 0:
                self.spawn_ball_params = [[[(600, 100), (650, 100)], [(120/180)*pi,
                                                                          (130/180)*pi], [3,4]]]
            elif task == 'smash' and 20 > wind_magnitude >= 10:
                self.spawn_ball_params = [[[(600, 100), (650, 100)], [(130/180)*pi,
                                                                          (140/180)*pi], [5,6]]]
            elif task == 'smash' and 30 >= wind_magnitude >= 20:
                self.spawn_ball_params = [[[(600, 100), (650, 100)], [(130/180)*pi,
                                                                          (140/180)*pi], [7,8]]]            

        # if not(train_wm) and task=='counter':
        #     if 8 <= ball_mass < 10:
        #         self.spawn_ball_params = [[[(550, 260), (600, 260)], [(190/180)*pi, (195/180)*pi]
        #                                    , [3,4]]]
        #     elif 10 < ball_mass <= 12:
        #         self.spawn_ball_params = [[[(600, 260), (650, 260)], [(190/180)*pi, (195/180)*pi]
        #                                    , [2,3]]]
        #     elif 12 < ball_mass <= 14:
        #         self.spawn_ball_params = [[[(650, 260), (700, 260)], [(190/180)*pi, (195/180)*pi]
        #                                    , [2,3]]]               
        #     elif 14 < ball_mass <= 16:
        #         self.spawn_ball_params = [[[(650, 260), (700, 260)], [(185/180)*pi, (190/180)*pi]
        #                                    , [2,3]]]

        if not(self.render_mode):
            os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    def seed(self, seed):
        np.random.seed(seed)
        
    def spawn_ball(self, spawn_ball_params):
        params_choice = spawn_ball_params[np.random.choice(len(spawn_ball_params))]
        position_x = np.random.uniform(low=params_choice[0][0][0], high=params_choice[0][1][0])
        position_y = np.random.uniform(low=params_choice[0][0][1], high=params_choice[0][1][1])
        position = position_x, position_y
        self.ball_body = pymunk.Body(self.ball_mass, ball_moment)
        self.ball_body.position = position

        # ball_shape = pymunk.Poly(ball_body, fp)
        self.ball_shape = pymunk.Circle(self.ball_body, ball_radius)
        self.ball_shape.friction = 0.2
        self.ball_shape.color = pygame.Color("purple")
        self.ball_shape.elasticity = 1.0
        self.ball_shape.collision_type = collision_types["ball"]

        # Random shooting
        sample_mag = np.random.uniform(low=params_choice[2][0], high=params_choice[2][1]) * force_const
        sample_ang = np.random.uniform(low=params_choice[1][0], high=params_choice[1][1])

        direction = Vec2d(1,0).rotated(sample_ang)
        force = sample_mag * direction


        self.ball_body.apply_impulse_at_local_point(Vec2d(force[0], force[1]))

        self.space.add(self.ball_body, self.ball_shape)
        return self.ball_body

    # Limit velocity and angular velocity
    def limit_velocity(self, body, gravity, damping, dt):
        max_velocity = max_speed
        max_ang_velocity = max_ang_speed
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        l = body.velocity.length
        ang_l = np.abs(body.angular_velocity)
        ang = body.angle
        if l > max_velocity:
            scale = max_velocity / l
            body.velocity = body.velocity * scale
        if ang_l > max_ang_velocity:
            ang_scale = max_ang_velocity / ang_l
            body.angular_velocity = body.angular_velocity * ang_scale
        if body.angular_velocity < 0 and ang < self.min_angle:
            body.angle = self.min_angle
            body.torque = 0
            body.angular_velocity = 0
        if body.angular_velocity > 0 and ang > self.max_angle:
            body.angle = self.max_angle
            body.torque = 0
            body.angular_velocity = 0
        
            

    def setup_level(self):
        # Remove balls and bricks
        for s in self.space.shapes[:]:
            if s.body.body_type == pymunk.Body.DYNAMIC and s.body not in [self.player_body]:
                self.space.remove(s.body, s)

        # Spawn a ball for the player to have something to play with
        self.spawn_ball(self.spawn_ball_params)
        
        # Create target
        var_x, var_y = np.random.uniform(low=0.0, high=1.0, size=2)
        target_x = 600 + (var_x - 0.5)*40
        target_y = 60
        if self.train_wm:
            target_y = -100
        self.target_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        if self.init_state != []:
            self.target_body.position = self.init_state[11] * norm_dist, self.init_state[12] * norm_dist
        else:
            self.target_body.position = target_x, target_y
        self.target_shape = pymunk.Poly.create_box(self.target_body, (40, 10))
        self.target_shape.elasticity = 1
        self.target_shape.color = pygame.Color("red")
        self.target_shape.collision_type = collision_types["target"]
        self.space.add(self.target_body, self.target_shape)


    def get_env_state(self, normalised=True, clean_obs=False):
        # Get Vec2D values from the pymunk objects
        player_pos = self.player_body.position
        player_ang_pos = self.player_body.angle
        player_vel = self.player_body.velocity
        player_ang_vel = self.player_body.angular_velocity
        
        ball_pos = self.ball_body.position
        ball_vel = self.ball_body.velocity
        ball_ang_vel = self.ball_body.angular_velocity
        target_pos = self.target_body.position
        # Clean observation is for visualising purpose
        if self.noise and not(clean_obs):
            # player_pos_noise = norm_dist*0.01*np.random.normal(size=2)
            # player_ang_pos_noise = np.random.normal(size=1)
            # player_vel_noise = np.random.normal(size=2)
            # player_ang_vel_noise = np.random.normal(size=1)
            ball_pos_noise = norm_dist*0.0025*np.random.normal(size=2)
            ball_vel_noise = max_speed*0.0025*np.random.normal(size=2)
            ball_ang_vel_noise = max_ang_speed*0.0025*np.random.normal(size=1)
            target_pos_noise = norm_dist*0.0025*np.random.normal(size=2)
            
            ball_pos_noise_vec = Vec2d(ball_pos_noise[0], ball_pos_noise[1])
            ball_vel_noise_vec = Vec2d(ball_vel_noise[0], ball_vel_noise[1])
            target_pos_noise_vec = Vec2d(target_pos_noise[0], target_pos_noise[1])
            
            ball_pos += ball_pos_noise_vec
            ball_vel += ball_vel_noise_vec
            ball_ang_vel += ball_ang_vel_noise[0]
            target_pos += target_pos_noise_vec            
            

        
        if self.relative_vec:

            horizontal_vec = Vec2d(1,0)
            player_ball_vec = ball_pos - player_pos + Vec2d(0.0001, 0.0001)
            ball_target_vec = target_pos - ball_pos + Vec2d(0.0001, 0.0001)

            _, player_ball_dist = player_ball_vec.normalized_and_length()
            _, ball_target_dist = ball_target_vec.normalized_and_length()

            player_ball_rel_angle = player_ball_vec.get_angle_between(horizontal_vec)
            ball_target_rel_angle = ball_target_vec.get_angle_between(horizontal_vec)

            player_ball_vel = player_vel - ball_vel + Vec2d(0.0001, 0.0001)

            _, player_ball_speed = player_ball_vel.normalized_and_length()

            player_ball_rel_vel_angle = player_ball_vel.get_angle_between(horizontal_vec)

            state_list = [player_ball_dist / norm_dist, player_ball_rel_angle / (2*pi), player_ang_pos / (2*pi), player_ball_speed / max_speed, player_ball_rel_vel_angle / (2*pi), player_ang_vel /  max_ang_speed, 
                          ball_pos[0] / norm_dist, ball_pos[1] / norm_dist,
ball_target_dist / norm_dist, ball_target_rel_angle / (2*pi), ball_vel[0] / max_speed, ball_vel[1] / max_speed, ball_ang_vel / max_ang_speed, target_pos[0] / norm_dist]
       
        if self.train_wm:
            state_list = [player_pos[0] / norm_dist,       # [0] Player Position x
                          player_ang_pos / (2*pi),         # [1] Player angle
                          player_vel[0] / max_speed,       # [2] Player velocity x
                          player_ang_vel /  max_ang_speed, # [3] Player angular velocity 
                          ball_pos[0] / norm_dist,         # [4] ball position x
                          ball_pos[1] / norm_dist,         # [5] ball position y
                          ball_vel[0] / max_speed,         # [6] ball velocity x
                          ball_vel[1] / max_speed,         # [7] ball velocity y
                          ball_ang_vel /  max_ang_speed,   # [8] ball angular velocity
                          target_pos[0] / norm_dist,       # [9] target position x
                          self.ball_player_hit,            # [10] ball-player hit
                          self.ball_left_hit,              # [11] ball-left-wall hit
                          self.ball_ceiling_hit,           # [12] ball-ceiling hit
                          self.ball_right_hit,             # [13] ball-right-wall hit
                          self.ball_ground_hit,            # [14] ball-ground hit
                          self.ball_net_hit]               # [15] ball-net hit
        else:
            state_list = state_list = [player_pos[0] / norm_dist,
                                       player_pos[1] / norm_dist, 
                                       player_ang_pos / (2*pi),      
                                       player_vel[0] / max_speed,
                                       player_vel[1] / max_speed,
                                       player_ang_vel /  max_ang_speed, 
                                       ball_pos[0] / norm_dist, 
                                       ball_pos[1] / norm_dist,
                                       ball_vel[0] / max_speed, 
                                       ball_vel[1] / max_speed,
                                       ball_ang_vel /  max_ang_speed, 
                                       target_pos[0] / norm_dist,
                                       target_pos[1] / norm_dist
                                      ]
         
        state_array = np.array(state_list)
        self.state_dim = state_array.shape
            
        return state_array
    
    def get_state_dict(self, state_list):
        state_dict = {}
        state_dict['player_position'] = [state_list[0]]
        state_dict['player_angle'] = [state_list[1]]
        state_dict['player_velocity'] = [state_list[2]]
        state_dict['player_angular_velocity'] = [state_list[3]]
        state_dict['ball_position'] = [state_list[4], state_list[5]]
        state_dict['ball_velocity'] = [state_list[6], state_list[7]]
        state_dict['ball_angular_velocity'] = [state_list[8]]
        state_dict['target_position'] = [state_list[9]]
        state_dict['ball_player_hit'] = [state_list[10]]
        state_dict['ball_left_hit'] = [state_list[11]]
        state_dict['ball_ceiling_hit'] = [state_list[12]]
        state_dict['ball_right_hit'] = [state_list[13]]
        state_dict['ball_ground_hit'] = [state_list[14]]
        state_dict['ball_net_hit'] = [state_list[15]]
        return state_dict
    
    # TODO fix idx
    def unnormalise_state(self, state, ball_pred=False):
        new_state = np.zeros_like(state)
        if ball_pred:
            new_state[0] = state[0] * norm_dist
            new_state[1] = state[1] * norm_dist
            new_state[2] = state[2] * max_speed
            new_state[3] = state[3] * max_speed
            state_dict = {}
            state_dict['ball_position'] = (new_state[0], new_state[1])
            state_dict['ball_velocity'] = (new_state[2], new_state[3])

        else:
            new_state[0] = state[0] * norm_dist
            new_state[1] = state[1] * norm_dist
            new_state[2] = state[2] * (2*pi)
            new_state[3] = state[3] * max_speed
            new_state[4] = state[4] * max_speed
            new_state[5] = state[5] * max_ang_speed
            new_state[6] = state[6] * norm_dist
            new_state[7] = state[7] * norm_dist
            new_state[8] = state[8] * max_speed
            new_state[9] = state[9] * max_speed
            new_state[10] = state[10] * max_ang_speed
            new_state[11] = state[11] * norm_dist 
            new_state[12] = state[12] * norm_dist
            new_state[13] = state[13]

            state_dict = {}
            state_dict['player_position'] = (new_state[0], new_state[1])
            state_dict['player_angle'] = new_state[2]
            state_dict['player_velocity'] = (new_state[3], new_state[4])
            state_dict['player_angular_velocity'] = new_state[5]
            state_dict['ball_position'] = (new_state[6], new_state[7])
            state_dict['ball_velocity'] = (new_state[8], new_state[9])
            state_dict['ball_angular_position'] = new_state[10]
            state_dict['target_position'] = (new_state[11], new_state[12])
            state_dict['hit'] = new_state[13]
            
        return  new_state, state_dict
        
        
    def initialise_state(self, state):
        # Initialise state
        self.player_body.position = state[0] * norm_dist, state[1] * norm_dist
        self.player_body.angle = state[2] * (2*pi)
        self.player_body.velocity = state[3] * max_speed, state[4] * max_speed
        self.player_body.angular_velocity = state[5] * max_ang_speed
        self.ball_body.position = state[6] * norm_dist, state[7] * norm_dist
        self.ball_body.velocity = state[8] * max_speed, state[9] * max_speed
        self.ball_body.angular_velocity = state[10] * max_ang_speed
        self.target_body.position = state[11] * norm_dist, state[12] * norm_dist
        
        self.player_shape.position = self.player_body.position
        self.player_shape.angle = self.player_body.angle
        self.player_shape.velocity = self.player_body.velocity
        self.player_shape.angular_velocity = self.player_body.angular_velocity
        self.ball_shape.position = self.ball_body.position
        self.ball_shape.velocity = self.ball_body.velocity
        self.ball_shape.angular_velocity = self.ball_body.angular_velocity
        self.target_shape.position = self.target_body.position
                

        

    def get_distance(self, pos1, pos2):
        dist = abs(pos1 - pos2)
        return dist / norm_dist

    def step(self, action):
        reward = 0
        running = True
        prev_action = np.copy(action)
        
        if self.continuous_control:
            # action = action.cpu().detach().numpy()
            # Transform action range from [0,1] to [-1,1]
            action = (2*action) - 1
        for _ in range(self.num_skip):
            # Agent Control
            if not(self.player):
                if not(self.continuous_control):
                    event_list = pygame.event.get()
                    # Input (move, swing, target_angle, gauge)
                    # Moving left or right or not moving (Need to transform one-hot output of the network to the number
                    # Not move
                    if action == 0:
                        self.player_body.velocity = (0, 0)
                    # Move left 
                    elif action == 1:
                        self.player_body.velocity = (-400, 0)
                    # Move right
                    elif action == 2:
                        self.player_body.velocity = (400, 0)

                    # bottom-to-top swing
                    if action == 3:
                        if self.target_line.reach and target_line.delay_time <= 0:
                            self.target_line.swing(control_angular_velocity=self.control_angular_velocity,
                                              control_angle=-self.target_angle, player=self.player)
                        self.player_body.velocity = (0, 0)
                        action = 0

                    # top-to-bottom swing
                    if action == 4:
                        if self.target_line.reach and self.target_line.delay_time <= 0:
                            self.target_line.swing(control_angular_velocity=-self.control_angular_velocity,
                                              control_angle=self.target_angle, player=self.player)
                        self.player_body.velocity = (0, 0)
                        action = 0

                    if not(self.target_line.reach):
                        self.player_body.velocity = (0, 0)
                        self.player_body.angular_velocity = self.target_line.control_angular_velocity

                elif self.continuous_control:
                    self.player_body.force =  (player_force*action[0, 0], 0)
                    self.player_body.torque =  player_torque*action[0, 1]


            # Human Control
            elif self.player:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN and (
                        event.key in [pygame.K_ESCAPE, pygame.K_q]
                    ):
                        running = False
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                        pygame.image.save(screen, "breakout.png")

                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT:
                        self.player_body.velocity = (-600, 0)
                    elif event.type == pygame.KEYUP and event.key == pygame.K_LEFT:
                        self.player_body.velocity = 0, 0

                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT:
                        self.player_body.velocity = (600, 0)
                    elif event.type == pygame.KEYUP and event.key == pygame.K_RIGHT:
                        self.player_body.velocity = 0, 0

                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        # setup_level(space, player_body)
                        # Spawn a ball for the player to have something to play with
                        running = False
                        self.space.remove(self.ball_shape, self.ball_shape.body)
                        self.spawn_ball(self.spawn_ball_params)
                    elif event.type == pygame.MOUSEBUTTONDOWN and (event.button == 1 or event.button == 3):
                        start_time = pygame.time.get_ticks()
                        self.target_line.swing()


                    elif event.type == pygame.MOUSEBUTTONUP: # and event.button == 1:
                        end_time = pygame.time.get_ticks()

                        diff = end_time - start_time
                        power = max(min(diff, 1000), 10) * 6

                        # If the bat doesn't reach the target angle do
                        if not(target_line.reach):
                            if event.button == 1:
                                self.player_body.angular_velocity = power * 0.0025
                            elif event.button == 3:
                                self.player_body.angular_velocity = - power * 0.0025
                                
            # reward-=1
            # Compute ball to target distance
            player_ball_dist = self.get_distance(self.player_body.position, self.ball_body.position)
            ball_target_dist = self.get_distance(self.ball_body.position, self.target_body.position)
            self.ball_player_hit = self.ball_shape.shapes_collide(self.player_shape).points != []
            self.ball_left_hit = self.ball_shape.shapes_collide(self.static_lines[0]).points != []
            self.ball_ceiling_hit = self.ball_shape.shapes_collide(self.static_lines[1]).points != []
            self.ball_right_hit = self.ball_shape.shapes_collide(self.static_lines[2]).points != []
            self.ball_ground_hit = self.ball_shape.shapes_collide(self.static_lines[3]).points != []
            self.ball_net_hit = self.ball_shape.shapes_collide(self.static_lines[4]).points != []
            
            if self.hits['player'] == 1 and self.save_speed_dist_corr:
                self.step_after_hit += 1
                if self.step_after_hit == 2:
                    ball_velocity = Vec2d(*self.ball_body.velocity)
                    ball_direction, ball_speed = ball_velocity.normalized_and_length()
                    self.hit_velocity_list.append(ball_speed)
                    self.hit_angle_list.append(ball_direction.angle * (180 / pi))
                    self.dist_list.append(norm_dist * ball_target_dist)
                    self.save_speed_dist_corr = False
                
                
            
            # if player swing and hit the ball
            if self.ball_player_hit:
                if self.hits['left'] == 0 and self.task == 'counter':
                    running = False
                else:
                    self.hits['player'] += 1
                    self.after_hit = 1
                    self.step_after_hit += 1
            

            # Check if the ball pass the net (middle)
            self.pass_net = (self.ball_body.position[0] <= (width // 2))
            
            # If the ball hit the ground
            if self.static_lines[3].shapes_collide(self.ball_shape).points != []:
                # We only need the paths before the ball bounces from the floor for WM
                if self.train_wm:
                    running=False
                # Left hit
                if self.ball_body.position[0] <= width // 2:
                    self.hits['left'] += 1
                else:
                    self.hits['right'] += 1

            if self.hits['left'] >= 2:
                running=False
                
            if self.hits['right'] >= 1:
                running=False
                
            if self.target_shape.shapes_collide(self.ball_shape).points != [] and self.hits['target'] == 0:
                self.hits['target'] += 1
                running=False
                
            if self.ball_body.position[0] < 0 or self.ball_body.position[0] > width:
                running=False
            if self.ball_body.position[1] < 0 or self.ball_body.position[1] > height:
                running=False                

            self.player_joint_body.position = self.player_body.position

            # Apply Drag force
            pointing_direction = Vec2d(1, 0).rotated(self.ball_body.angle)
            ball_velocity = Vec2d(*self.ball_body.velocity)
            ball_direction, ball_speed = ball_velocity.normalized_and_length()
            drag_force_magnitude = (ball_speed ** 2) * (self.drag_constant)
            self.ball_body.apply_impulse_at_world_point(
                drag_force_magnitude * -ball_direction, self.ball_body.position
            )

            # Apply Wind when the ball is moving
            if np.abs(self.ball_body.velocity[0]) > 0 or np.abs(self.ball_body.velocity[1]) > 0:
                wind_force = Vec2d(self.wind_magnitude, 0)
                self.ball_body.apply_impulse_at_world_point(
                    wind_force, self.ball_body.position
                )

            if not(self.continuous_control):
                self.target_line.update()

            ### Update physics
            dt = 0.6 / fps
            self.space.step(dt)

            if self.render_mode:
            # Power meter
                if pygame.mouse.get_pressed()[0] or pygame.mouse.get_pressed()[2]:
                    current_time = pygame.time.get_ticks()
                    diff = current_time - start_time
                    power = max(min(diff, 1000), 10)
                    h = power // 2
                    pygame.draw.line(self.screen, pygame.Color("red"), (30, 550), (30, 550 - h), 10)                
                ### Clear screen
                self.screen.fill(pygame.Color("black"))

                ### Draw stuff
                self.space.debug_draw(self.draw_options)                

                pygame.display.flip()

                self.clock.tick(fps * render_speed_up)
            # next_state = get_state(space)

        next_state = self.get_env_state()

        # Negative distance between player and ball when the ball is on player side
#         if self.train and self.ball_body.position[0] < width / 2 and player_ball_dist >= bat_length / norm_dist:
#                 reward -= 5e-3 * player_ball_dist
        
        
        # Check if last step
        if self.t == self.max_timestep - 1:
            running = False
        
        if not(running):
            # if self.train:
            if self.ball_body.position[0] > width / 2:
                if self.reward_func == 'linear':
                    reward += (1-ball_target_dist)
                elif self.reward_func == 'rbf':
                    reward += np.exp(-0.5*(ball_target_dist/0.06)**2)
                elif self.reward_func == 'poly':
                    reward += (1 - 2.2*(ball_target_dist)**(3/4))          
                
            
            # Compute total hit rates
            if self.hits['player'] >= 1:
                self.total_hits['player'] += 1
                
                if self.hits['right'] == 0 and self.hits['target'] == 0:
                    self.achievement_list.append(0)
    
            if self.hits['right'] >= 1 and self.hits['target'] == 0:
                self.total_hits['right'] += 1
                self.achievement_list.append(1)
                
            if self.hits['target'] >= 1:
                self.total_hits['target'] += 1
                self.achievement_list.append(2)
            
                
                

            # else:
            #     # reward -= ball_target_dist
            #     # reward -= 0.5*np.exp(-0.5*((1-ball_target_dist)/0.2)**2)
            #     reward -= 1
                
            # if self.hits['target'] > 0:
            #     reward += 1
                
            # else:
            #     if self.ball_body.position[0] > width / 2:
            #         reward += (1-ball_target_dist)
            #     else:
            #         reward -= ball_target_dist
                
#             if self.train and self.hits['target'] > 0:
#                 reward += 2
            
#             if self.train and self.hits['player'] <= 2:
#                 reward += 1

        self.t += 1
        
        if self.short_eps and not(self.player) and running and self.hits['player'] > 0:
            next_state, reward, done = self.step(np.zeros((1, 2))+0.5)
            running = not(done)
        
        # Not use
        # elif not(self.short_eps) and self.two_steps_eps and not(self.player) and running:
        #     # Receives the first state of the environment and immediately perform action
        #     if self.two_steps_mode == 'start':
        #         next_state, reward, done = self.step(prev_action)
        #         running = not(done)
        #     elif self.two_steps_mode == 'net' and self.pass_net:
        #         next_state, reward, done = self.step(prev_action)
        #         running = not(done)                
            # Receives the state of the environment such that the ball pass the net position
            # elif self.two_steps_mode == 'net':
            #     pass
                
        # print(self.t, self.get_env_state()[6:8], reward)
        return next_state, reward, not(running)

    def reset(self, init_state=[]):
        ### PyGame init
        self.step_count = 0
        self.state = None
        self.hit_angle = 0
        self.target_angle = 0
        self.reward = 0
        self.delta_ang = 0.05 * pi
        self.start_time = 0
        self.running = True
        self.ball_player_hit = 0
        self.ball_left_hit = 0
        self.ball_ceiling_hit = 0
        self.ball_right_hit = 0
        self.ball_ground_hit = 0
        self.ball_net_hit = 0
        self.after_hit = 0
        self.space = pymunk.Space()
        self.space.gravity = 0, -self.gravity_const
        self.t = 0
        self.init_state = init_state
        self.pass_net = False
        self.save_speed_dist_corr = True
        self.step_after_hit = 0
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 16)
            pymunk.pygame_util.positive_y_is_up = True
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        
        action, target_angle, control_angular_velocity = 0, 0, 12

        # hit counter
        self.hits = {}
        self.hits['left'] = 0
        self.hits['right'] = 0
        self.hits['player'] = 0
        self.hits['target'] = 0
        
        ### Game area
        # walls - the left-top-right walls
        if not(self.train_wm):
            self.static_lines = [
                # For frames
                pymunk.Segment(self.space.static_body, (-100, 50), (-100, height-50), 4),
                pymunk.Segment(self.space.static_body, (0, height-50), (width, height-50), 4),
                pymunk.Segment(self.space.static_body, (width+100, height-50), (width+100, 50), 4),
                pymunk.Segment(self.space.static_body, (0, 50), (width, 50), 4),

                # Net
                pymunk.Segment(self.space.static_body, (400, 50), (400, 130), 4)
            ]
        else:
            self.static_lines = [
                # For frames
                pymunk.Segment(self.space.static_body, (-200, 50), (-200, height-50), 4),
                pymunk.Segment(self.space.static_body, (0, height+200), (width, height+200), 4),
                pymunk.Segment(self.space.static_body, (width+200, height-50), (width+200, 50), 4),
                pymunk.Segment(self.space.static_body, (0, 50), (width, 50), 4),

                # Net
                pymunk.Segment(self.space.static_body, (-1000, 50), (-1000, 130), 4)
            ]            
        for line in self.static_lines:

            line.color = pygame.Color("lightgray")
            line.elasticity = 1.0
            line.friction = 0.2

        self.space.add(*self.static_lines)
        self.init_player_position = init_player_position


        ### Player
        self.player_body = pymunk.Body(bat_mass, bat_moment)
        # Unifrom Continuous Random Position
        # player_var_x = np.random.uniform(low=-50, high=70, size=1)
        # player_body.position = init_player_position[0] + player_var_x,  init_player_position[1]

        # For training world model of the ball, we need to remove the bouncing effect due to the racket
        if self.train_wm:
            self.init_player_position = -200, 600
            player_random_x = self.init_player_position[0]
        else:
            player_random_x = np.random.choice(player_body_initial_positions)

        
        self.player_body.position = player_random_x,  self.init_player_position[1]
        self.player_body.angle = pi
        self.player_body.velocity_func = self.limit_velocity
        self.player_shape = pymunk.Segment(self.player_body, bat_params[0], bat_params[1], bat_params[2])
        self.player_shape.color = pygame.Color("red")
        self.player_shape.elasticity = 1.0
        self.player_shape.friction = 0.5
        self.player_shape.collision_type = collision_types["player"]
        self.player_joint_body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
        self.player_joint_body.position = self.player_body.position

        self.target_line = create_target_line(self.player_body, self.screen, self.delta_ang)


        # restrict movement of player to a straigt line
        self.move_joint = pymunk.GrooveJoint(self.space.static_body, self.player_body, 
                                        (self.init_player_position[0]-40, self.init_player_position[1]), 
                                        (self.init_player_position[0]+40, self.init_player_position[1]), 
                                        (0, 0))
        self.move_joint.color = pygame.Color("black")
        self.space.add(self.player_body, self.player_shape, self.move_joint)

        self.setup_level()
        
        if len(init_state) != 0:
            self.initialise_state(init_state)
        self.state = self.get_env_state()
    
class create_target_line:
    def __init__(self, player_body, screen, delta_angle):
        self.reach = True
        self.target_vec =  Vec2d(1,0)
        self.target_ang = 0
        self.player_vec = Vec2d(1,0)
        self.player_body = player_body
        self.screen = screen
        self.delta_angle = delta_angle
        self.delay_time = 0
        self.control_angular_velocity = 0

        self.target_end_point = Vec2d(0,0)
        self.player_end_point = Vec2d(0,0)
        self.max_frame = 30
        self.current_frame = 0

    def swing(self, control_angular_velocity, control_angle=0, player=True):
        self.delay_time = fps * swing_delay # 2 seconds delay
        self.reach = False
        if player:
            mouse_position_vec = pymunk.pygame_util.from_pygame(
                Vec2d(*pygame.mouse.get_pos()), self.screen
            )        
            self.target_ang = (mouse_position_vec - self.player_body.position).angle
        # For agent
        else:
            self.target_ang = control_angle
            self.control_angular_velocity = control_angular_velocity
        self.target_vec = Vec2d(1,0).rotated(self.target_ang)

    def update(self):
        self.player_vec = Vec2d(1,0).rotated(self.player_body.angle)
        self.target_end_point = self.player_body.position + 200 * self.target_vec
        self.player_end_point = self.player_body.position + 200 * self.player_vec        
        if self.reach:
            None
            # pygame.draw.line(self.screen, pygame.Color("lightgray"), 
            #                  pygame.mouse.get_pos(), 
            #                  pymunk.pygame_util.to_pygame(self.player_body.position, self.screen), 1)
        else:
            self.delay_time -= 1
            pygame.draw.line(self.screen, pygame.Color("green"), 
                             pymunk.pygame_util.to_pygame(self.target_end_point, self.screen),
                             pymunk.pygame_util.to_pygame(self.player_body.position, self.screen), 1)

        angular_dist = np.abs(self.target_vec.get_angle_between(self.player_vec))


        if not(self.reach):
            self.font = pygame.font.SysFont("Arial", 16)
            self.screen.blit(
                self.font.render("Ang Dist: " + str(angular_dist), 1, pygame.Color("white")),
                (500, 0),
            )
        if angular_dist < self.delta_angle:
            self.reach = True
            self.player_body.angular_velocity = 0
            self.player_body.angle = pi    