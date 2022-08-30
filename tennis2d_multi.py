
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
from collections import namedtuple, deque
from itertools import count
import math
from itertools import product
import matplotlib
import matplotlib.pyplot as plt
import ounoise
device = torch.device("cpu")
# Fix seed
# Constant
pi = 3.141592653
width, height = 800, 600
norm_dist = np.sqrt(width**2 + height**2)
fps = 60
collision_types = {
    "ball": 1,
    "player1": 2,
    "player2":3,
}
# Parameters
# Player 1
init_player_position = {'1':(120, 150), '2':(680, 150)}
bat_mass = 1e4
bat_length = 80
bat_thick = 10
bat_params = [(0, 0), (bat_length, 0), bat_thick]
bat_moment = pymunk.moment_for_segment(bat_mass, bat_params[0], bat_params[1], bat_params[2])
force_const = 1e4
# spawn_ball_params= [[[(init_player_position['1'][0]+20,300), (init_player_position['1'][0]+21,300)], [(110/180)*pi, (120/180)*pi], [0,0.1]]]
spawn_ball_params_player  = {}
spawn_ball_params_player['1'] = [[[(550, 160), (700, 160)], [(120/180)*pi, (140/180)*pi], [1.5,2.2]]]
spawn_ball_params_player['2'] = [[[(100, 160), (250, 160)], [(30/180)*pi, (50/180)*pi], [1.5,2.2]]]
ball_radius = 10
ball_mass = 10
ball_moment = pymunk.moment_for_circle(mass=ball_mass, inner_radius=0, outer_radius=ball_radius)

# Gravity
gravity_const = 500
# Drag
drag_constant = 0
# Wind
wind_magnitude = 0
player_force = 1e7
player_torque = 2.5e9
max_speed = 200
max_ball_speed = 400
max_ang_speed = 12
render_speed_up = 2
horizontal_vec = Vec2d(1,0)
# Constrains on agents
min_angle1 = (240/180) * pi
max_angle1 = (330/180) * pi

min_angle2 = -(150/180) * pi
max_angle2 = (-60/180) * pi

class create_tennis2D_env:
    def __init__(self, num_agents, train, max_timestep, num_skip, continuous_control, render_mode, short_eps=False, 
                train_wm=False, noise=False, ball_mass=ball_mass, drag_constant=drag_constant, gravity_const=gravity_const,
                 spawn_ball_params=spawn_ball_params_player, wind_magnitude=wind_magnitude):
        self.num_agents = num_agents
        self.ball_shape = None
        self.ball_body = None
        self.player_shape = {}
        self.player_body = {}
        self.player_joint_body = {}

        if train_wm:
            self.init_player_position = {'1':(-1000, 150), '2':(1000, 150)}
        else:
            self.init_player_position = init_player_position
        self.move_joint = {}
        self.player_colors = {'1':'red', '2':'blue'}
        self.player_init_angles = {'1':min_angle1, '2':max_angle2}
        
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
        self.noise = noise
        self.train_wm = train_wm
        self.total_hits = {'player1':0, 'player2':0, 'right':0, 'target': 0}
        
        self.execute = {'player1':0, 'player2':0}
        
        self.max_speed = max_speed
        self.max_ang_speed = max_ang_speed
        self.norm_dist = norm_dist

        self.force_list = []
        self.torque_list = []
        self.dist_list = []
        self.hit_velocity_list = []
        
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
        self.spawn_ball_params_player = spawn_ball_params
        self.who_play = 2
        if not(self.render_mode):
            os.environ["SDL_VIDEODRIVER"] = "dummy"
    
    def seed(self, seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def limit_ball_velocity(self, body, gravity, damping, dt):
        max_velocity = max_ball_speed
        pymunk.Body.update_velocity(body, gravity, damping, dt)
        l = body.velocity.length
        if l > max_velocity:
            scale = max_velocity / l
            body.velocity = body.velocity * scale

    def limit_velocity1(self, body, gravity, damping, dt):
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
        if body.angular_velocity < 0 and ang < min_angle1:
            body.angle = min_angle1
            body.torque = 0
            body.angular_velocity = 0
        if body.angular_velocity > 0 and ang > max_angle1:
            body.angle = max_angle1
            body.torque = 0
            body.angular_velocity = 0

    def limit_velocity2(self, body, gravity, damping, dt):
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
            
        if body.angular_velocity > 0 and ang > max_angle2:
            body.angle = max_angle2
            body.torque = 0
            body.angular_velocity = 0
        if body.angular_velocity < 0 and ang < min_angle2:
            body.angle =  min_angle2
            body.torque = 0
            body.angular_velocity = 0
            
    def spawn_ball(self, spawn_ball_params):
        params_choice = spawn_ball_params[np.random.choice(len(spawn_ball_params))]
        position_x = np.random.uniform(low=params_choice[0][0][0], high=params_choice[0][1][0])
        position_y = np.random.uniform(low=params_choice[0][0][1], high=params_choice[0][1][1])
        position = position_x, position_y
        self.ball_body = pymunk.Body(self.ball_mass, ball_moment)
        self.ball_body.position = position
        self.ball_body.velocity_func = self.limit_ball_velocity

        # ball_shape = pymunk.Poly(ball_body, fp)
        self.ball_shape = pymunk.Circle(self.ball_body, ball_radius)
        self.ball_shape.friction = 0.2
        self.ball_shape.color = pygame.Color("purple")
        self.ball_shape.elasticity = 0.7
        self.ball_shape.collision_type = collision_types["ball"]

        # Random shooting
        sample_mag = np.random.uniform(low=params_choice[2][0], high=params_choice[2][1]) * force_const
        sample_ang = np.random.uniform(low=params_choice[1][0], high=params_choice[1][1])

        direction = Vec2d(1,0).rotated(sample_ang)
        force = sample_mag * direction


        self.ball_body.apply_impulse_at_local_point(Vec2d(force[0], force[1]))

        self.space.add(self.ball_body, self.ball_shape)
        return self.ball_body
    
    
    def add_players(self):
        
        for i in range(1,self.num_agents+1):
            self.limit_velocity_functions = {'1': self.limit_velocity1, '2':self.limit_velocity2}
            ### Player
            self.player_body[str(i)] = pymunk.Body(bat_mass, bat_moment)

            self.player_body[str(i)].position = self.init_player_position[str(i)][0],  self.init_player_position[str(i)][1]
            self.player_body[str(i)].angle = self.player_init_angles[str(i)]
            self.player_body[str(i)].velocity_func = self.limit_velocity_functions[str(i)]
            self.player_shape[str(i)] = pymunk.Segment(self.player_body[str(i)], 
                                                       bat_params[0], bat_params[1], bat_params[2])
            self.player_shape[str(i)].color = pygame.Color(self.player_colors[str(i)])
            self.player_shape[str(i)].elasticity = 1.0
            self.player_shape[str(i)].friction = 0.5
            self.player_shape[str(i)].collision_type = collision_types["player"+str(i)]
            self.player_joint_body[str(i)] = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
            self.player_joint_body[str(i)].position = self.player_body[str(i)].position
            self.player_joint_body[str(i)].color = pygame.Color("green")

            # restrict movement of player to a straigt line
            
            if i == 1:
                self.move_joint[str(i)] = pymunk.GrooveJoint(self.space.static_body, self.player_body[str(i)], 
                                            (self.init_player_position[str(i)][0]-30, self.init_player_position[str(i)][1]), 
                                            (self.init_player_position[str(i)][0]+70, self.init_player_position[str(i)][1]), 
                                                (0, 0))
            else:
                self.move_joint[str(i)] = pymunk.GrooveJoint(self.space.static_body, self.player_body[str(i)], 
                                            (self.init_player_position[str(i)][0]-70, self.init_player_position[str(i)][1]), 
                                            (self.init_player_position[str(i)][0]+30, self.init_player_position[str(i)][1]), 
                                                (0, 0))                
            self.move_joint[str(i)].color = pygame.Color("green")
            self.space.add(self.player_body[str(i)], self.player_shape[str(i)], self.move_joint[str(i)])

    def setup_level(self):
        # Remove balls and bricks
        for s in self.space.shapes[:]:
            # if s.body.body_type == pymunk.Body.DYNAMIC and s.body not in [self.player_body]:
            self.space.remove(s.body, s)

        ### Game area
        # walls - the left-top-right walls
        if not(self.train_wm):
            self.static_lines = [
                # For frames
                pymunk.Segment(self.space.static_body, (-100, 50), (-100, height-50), 4),
                pymunk.Segment(self.space.static_body, (0, height-50), (width, height-50), 4),
                pymunk.Segment(self.space.static_body, (width+100, height-50), (width+100, 50), 4),
                pymunk.Segment(self.space.static_body, (-1000, 50), (width+1000, 50), 4),

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
        # Spawn a ball for the player to have something to play with
        # Random who start playing
        self.who_play = int(np.random.binomial(1, 0.5) + 1)
        self.spawn_ball(self.spawn_ball_params_player[str(self.who_play)])
        self.execute['player'+str(self.who_play)] = True
        self.add_players()
        

    def get_env_state(self, normalised=True, clean_obs=False):
        state_list = []
        if self.train_wm:
            ball_pos = self.ball_body.position
            ball_vel = self.ball_body.velocity            
            # Need to make output of wm to be similar to single-agent tennis2d
            # to reuse meta_wmnet.py code
            state_list = [0,       # [0] Player Position x
                          0,         # [1] Player angle
                          0,       # [2] Player velocity x
                          0, # [3] Player angular velocity 
                          ball_pos[0] / norm_dist,         # [4] ball position x
                          ball_pos[1] / norm_dist,         # [5] ball position y
                          ball_vel[0] / max_speed,         # [6] ball velocity x
                          ball_vel[1] / max_speed,         # [7] ball velocity y
                          0,   # [8] ball angular velocity
                          0,       # [9] target position x
                          0,            # [10] ball-player hit
                          0,              # [11] ball-left-wall hit
                          0,           # [12] ball-ceiling hit
                          0,             # [13] ball-right-wall hit
                          0,            # [14] ball-ground hit
                          0]               # [15] ball-net hit
        else:
            for i in range(1, self.num_agents+1):
                # Get Vec2D values from the pymunk objects
                player_pos = self.player_body[str(i)].position
                player_ang_pos = self.player_body[str(i)].angle
                player_vel = self.player_body[str(i)].velocity
                player_ang_vel = self.player_body[str(i)].angular_velocity

                state_list +=  [player_pos[0] / norm_dist,
                               player_pos[1] / norm_dist, 
                               player_ang_pos / (2*pi),      
                               player_vel[0] / max_speed,
                               player_vel[1] / max_speed,
                               player_ang_vel /  max_ang_speed]


            ball_pos = self.ball_body.position
            ball_vel = self.ball_body.velocity

            # Clean observation is for visualising purpose
            if self.noise and not(clean_obs):
                ball_pos_noise = norm_dist*0.0025*np.random.normal(size=2)
                ball_vel_noise = max_speed*0.0025*np.random.normal(size=2)

                ball_pos_noise_vec = Vec2d(ball_pos_noise[0], ball_pos_noise[1])
                ball_vel_noise_vec = Vec2d(ball_vel_noise[0], ball_vel_noise[1])
                ball_pos += ball_pos_noise_vec
                ball_vel += ball_vel_noise_vec

            state_list += [ball_pos[0] / norm_dist, 
                           ball_pos[1] / norm_dist,
                           ball_vel[0] / max_speed, 
                           ball_vel[1] / max_speed]
         
        state_array = np.array(state_list)
        self.state_dim = state_array.shape
        return state_array
    
    def get_ball_state(self, state_array):
        return state_array[-4:]
    
    def get_env_state_dim(self):
        self.reset()
        return self.state_dim[0]

    def get_distance(self, pos1, pos2):
        dist = abs(pos1 - pos2)
        return dist / norm_dist

    def step(self, input_actions):
        self.reward_after_hit = {'player1':False, 'player2':False}
        reward = 0
        rewards = {'player1':0, 'player2':0}
        running = True
        actions = []
        
        # Transform action range from [0,1] to [-1,1]
        for i in range(len(input_actions)):
            action = input_actions[i].cpu().detach().numpy()
            action = (2*action) - 1
            actions.append(action)
            
        for _ in range(self.num_skip):
            for i in range(1, self.num_agents+1):
                if self.execute['player'+str(i)]:
                    action = actions[i-1]
                    self.player_body[str(i)].force =  (player_force*action[0, 0], 0)
                    self.player_body[str(i)].torque =  player_torque*action[0, 1]

                    self.ball_player_hit[str(i)] = self.ball_shape.shapes_collide(self.player_shape[str(i)]).points != []

                    # if player swing and hit the ball
                    if self.ball_player_hit[str(i)] and self.execute['player'+str(i)]:
                        if self.execute['player1']:
                            self.execute['player1'] = False
                            self.execute['player2'] = True
                            self.hit_by_player1 = True
                            self.hit_by_player2 = False
                            self.hits['player1'] += 1
                            self.reward_after_hit['player1'] = True
                            # if self.hits['left'] == 0:
                            #     running = False
                            # else:


                            self.hits['left'] = 0
                            self.hits['right'] = 0
                            self.hit_by_floor = False                            

                        elif self.execute['player2']:
                            self.execute['player2'] = False
                            self.execute['player1'] = True
                            self.hit_by_player1 = False
                            self.hit_by_player2 = True
                            self.hits['player2'] += 1
                            self.reward_after_hit['player2'] = True
                            # if self.hits['right'] == 0:
                            #     running = False
                            # else:


                            self.hits['left'] = 0
                            self.hits['right'] = 0
                            self.hit_by_floor = False
                
                self.player_joint_body[str(i)].position = self.player_body[str(i)].position
                               
            self.ball_left_hit = self.ball_shape.shapes_collide(self.static_lines[0]).points != []
            self.ball_ceiling_hit = self.ball_shape.shapes_collide(self.static_lines[1]).points != []
            self.ball_right_hit = self.ball_shape.shapes_collide(self.static_lines[2]).points != []
            self.ball_ground_hit = self.ball_shape.shapes_collide(self.static_lines[3]).points != []
            self.ball_net_hit = self.ball_shape.shapes_collide(self.static_lines[4]).points != []
            
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
                
            if self.ball_body.position[0] < 0 or self.ball_body.position[0] > width:
                running=False
            if self.ball_body.position[1] < 0 or self.ball_body.position[1] > height:
                running=False
            if self.train_wm:
                if self.hits['left'] > 0 or self.hits['right'] >0:
                    running = False

            ### Update physics
            dt = 0.6 / fps
            self.space.step(dt)
            
            # If the ball hit the ground
            if self.static_lines[3].shapes_collide(self.ball_shape).points != []:
                self.hit_by_floor = True
                if self.ball_body.position[0] <= width // 2:
                    self.hits['left'] += 1
                    if self.hits['left'] == 3:
                        running = False
                    if self.hits['left'] == 1 and self.execute['player1']:
                        self.reward_after_bounce['player2'] = True
                else:
                    self.hits['right'] += 1
                    if self.hits['right'] == 3:
                        running = False                    
                    if self.hits['right'] == 1 and self.execute['player2']:
                        self.reward_after_bounce['player1'] = True
              
                    
            if self.ball_net_hit:
                self.hit_by_floor = True
                
            # # Reward before hit
            # if self.ball_body.velocity[0] < 0 and self.ball_body.position[0] < width//2:
            #     reward -= 0.05
            # if self.ball_body.velocity[0] > 0 and self.ball_body.position[0] > width//2:
            #     reward -= 0.05
                

            if self.render_mode:
                self.screen.fill(pygame.Color("black"))
                self.space.debug_draw(self.draw_options)
                self.screen.blit(
                    self.font.render("Score: " + str(self.cum_rewards['player1']), 1, pygame.Color("green")),
                    (60, 0),
                )
                self.screen.blit(
                    self.font.render("Score: " + str(self.cum_rewards['player2']), 1, pygame.Color("green")),
                    (600, 0),
                )
                if self.wind_magnitude > 0:
                    self.screen.blit(
                        self.font.render("Wind", 1, pygame.Color("white")),
                        (380, 70),
                        )                    
                    self.screen.blit(
                        self.font.render(">>>", 1, pygame.Color("gray")),
                        (380, 100),
                        )
                    if self.wind_magnitude >= 2.5:
                        self.screen.blit(
                            self.font.render(">>>", 1, pygame.Color("gray")),
                            (380, 120),
                            )                              
                elif self.wind_magnitude < 0:
                    self.screen.blit(
                        self.font.render("Wind", 1, pygame.Color("white")),
                        (380, 70),
                        )                                 
                    self.screen.blit(
                        self.font.render("<<<", 1, pygame.Color("gray")),
                        (380, 100),
                        )
                    if self.wind_magnitude <= -2.5:
                        self.screen.blit(
                            self.font.render("<<<", 1, pygame.Color("gray")),
                            (380, 120),
                            )         
                


#                 self.screen.blit(
#                     self.font.render("Hit Count: " + str(self.hits['player1']), 1, pygame.Color("green")),
#                     (60, 50),
#                 )
#                 self.screen.blit(
#                     self.font.render("Hit Count: " + str(self.hits['player2']), 1, pygame.Color("green")),
#                     (600, 50),
#                 )              
                
                
#                 self.screen.blit(
#                     self.font.render("Rewarded: " + str(self.reward_after_bounce['player1']), 1, pygame.Color("green")),
#                     (60, 100),
#                 )
#                 self.screen.blit(
#                     self.font.render("Rewarded: " + str(self.reward_after_bounce['player2']), 1, pygame.Color("green")),
#                     (600, 100),
                # )                                           
                pygame.display.flip()
                self.clock.tick(fps * render_speed_up)

        next_state = self.get_env_state()
        
        ball_player1_dist = self.get_distance(self.ball_body.position, self.player_body['1'].position)
        ball_player2_dist = self.get_distance(self.ball_body.position, self.player_body['2'].position)
        
        if (width // 2) - 10 <= self.ball_body.position[0] <= (width // 2) + 10:     
            # If the ball move to the right, Player 2 execute action
            if self.ball_body.velocity[0] > 0:
                self.execute['player2'] = True
                self.hits['player1'] = 0
            # If the ball move to the left, Player 1 execute action
            else:
                self.execute['player1'] = True
                self.hits['player2'] = 0
                
#         # Reward for successful hit
#         if self.reward_after_hit['player1'] and self.ball_body.velocity[0]>max_ball_speed//2 and self.ball_body.velocity[1]>max_ball_speed//2:
#             self.reward_after_hit['player1'] = False
#             rewards['player1'] += 1
#             self.reward_after_bounce['player2'] = True
#             # print('player 1 get reward for hit the ball')
            
#         elif self.reward_after_hit['player2'] and self.ball_body.velocity[0]<max_ball_speed//2 and self.ball_body.velocity[1]>max_ball_speed//2:
#             self.reward_after_hit['player2'] = False
#             rewards['player2'] += 1
#             self.reward_after_bounce['player1'] = True
#             # print('player 2 get reward for hit the ball')
  
        
        if self.reward_after_bounce['player2'] and self.hits['player1'] >= 1:
            self.reward_after_bounce['player2'] = False
            rewards['player2'] += 1
            self.cum_rewards['player2'] += 1
            self.hits['player1'] = 0
            # self.hits['player1'] = 0
            # Old reward Model saved in "result_multi_agent_ddpg_old"
            # rewards['player2'] += np.exp(-0.5*(ball_player1_dist/0.06)**2)

            # Delta Reward
            # if ball_player1_dist < ((bat_length * 1.5) / norm_dist):
            #     rewards['player2'] += 1

            
        elif self.reward_after_bounce['player1'] and self.hits['player2'] >= 1:
            self.reward_after_bounce['player1'] = False
            rewards['player1'] += 1
            self.cum_rewards['player1'] += 1
            self.hits['player2'] = 0
            # self.hits['player2'] = 0
                

            # rewards['player1'] += np.exp(-0.5*(ball_player2_dist/0.06)**2)
            
            # Delta Reward
            # if ball_player2_dist < ((bat_length * 1.5) / norm_dist):
            #     rewards['player1'] += 1


        # if self.ball_body.position[0] < 0 or self.ball_body.position[0] > width and self.hits['left'] < 2 and self.hits['right'] < 2:
        #     running=False
        #     reward -= 1      
                
#         if self.pretrain and self.t == (self.max_timestep // 10)-1:
#             running = False
            
        if self.t == self.max_timestep-1:
            running = False
            
        # if not(running) and not(self.pretrain):
        #     rewards['player1'] += (self.hits['player1'] + self.hits['player2'])
#         if not(running):
#             if self.hits['left'] >= 3:
#                 rewards['player1'] -= 1
                

#             elif self.hits['right'] >= 3:
#                 rewards['player2'] -= 1
            # print(rewards['player1'])
            # print(rewards['player2'])
                
        self.t += 1

        return next_state, rewards, not(running)

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
        self.ball_player_hit = {'1':0, '2':0}
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
        self.execute = {'player1':False, 'player2':False} 
        self.hit_by_floor = True
        self.hit_by_player1 = False
        self.hit_by_player2 = False
        self.reward_after_bounce = {'player1':False, 'player2': False}
        # For visualising scores
        self.cum_rewards = {'player1':0, 'player2':0}
        
        
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Times New Roman", 24)
            pymunk.pygame_util.positive_y_is_up = True
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)

        # hit counter
        self.hits = {}
        self.hits['left'] = 0
        self.hits['right'] = 0
        self.hits['target'] = 0
                                        
        for i in range(1,self.num_agents+1):         
            self.hits['player'+str(i)] = 0

        self.setup_level()
        
        if len(init_state) != 0:
            self.initialise_ball_state(init_state)
        self.state = self.get_env_state()
        
        
    def initialise_ball_state(self, ball_state):
        # Initialise state
        self.ball_body.position = ball_state[0] * norm_dist, ball_state[1] * norm_dist
        self.ball_body.velocity = ball_state[2] * max_speed, ball_state[3] * max_speed
        
        self.ball_shape.position = self.ball_body.position
        self.ball_shape.velocity = self.ball_body.velocity