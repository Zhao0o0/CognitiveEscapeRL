import torch
import numpy as np
import random
from collections import deque
import time

class replay_buffer(object):
    def __init__(self, buffer_size, state_dim):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device:{}".format(self.device))
        self.buffer_size = buffer_size

        self.state_now_memory = torch.empty((buffer_size, *state_dim), device=self.device)
        self.action_memory = torch.empty(buffer_size, device=self.device, dtype=torch.long)
        self.reward_memory = torch.empty(buffer_size, device=self.device)
        self.state_next_memory = torch.empty((buffer_size, *state_dim), device=self.device)
        self.done_memory = torch.empty(buffer_size, device=self.device)
        self.position = 0
        self.current_size = 0


    def store_experience(self, experience):
        self.state_now_memory[self.position] = torch.tensor(experience.state_now, device=self.device)
        self.action_memory[self.position] = torch.tensor(experience.action, device=self.device)
        self.reward_memory[self.position] = torch.tensor(experience.reward, device=self.device)
        self.state_next_memory[self.position] = torch.tensor(experience.state_next, device=self.device)
        self.done_memory[self.position] = torch.tensor(experience.done, device=self.device)

        self.position = (self.position + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = torch.randint(low=0, high=self.current_size, size=(batch_size,), device=self.device)

        state_now_batch = self.state_now_memory[indices]
        action_batch = self.action_memory[indices]
        reward_batch = self.reward_memory[indices]
        state_next_batch = self.state_next_memory[indices]
        done_batch = self.done_memory[indices]

        return state_now_batch, action_batch, reward_batch, state_next_batch, done_batch


