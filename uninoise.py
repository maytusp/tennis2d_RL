import numpy as np
from collections import deque
import random
import torch

class UniNoise(object):
    def __init__(self, action_dim, low=0, high=1):
        self.action_dim   = action_dim
        self.low          = low
        self.high         = high
    
    def get_action(self, action):
        uni_state = np.random.uniform(low=self.low, high=self.high, size=self.action_dim)
        
        return torch.clamp(action + torch.tensor(uni_state), min=self.low, max=self.high)