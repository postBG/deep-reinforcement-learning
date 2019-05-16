import random
from collections import deque, namedtuple

import torch
import numpy as np

from utils import transpose_list, to_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def reshape_states(states):
    batch_size, num_agents, state_size = states.size()
    return states.view(num_agents, batch_size, state_size)


def reshape_rewards(rewards):
    batch_size, num_agents = rewards.size()
    return rewards.view(num_agents, batch_size)


def reshape_dones(dones):
    batch_size, num_agents = dones.size()
    return dones.view(num_agents, batch_size)


class ReplayBuffer(object):
    def __init__(self, size, batch_size):
        super().__init__()
        self.size = size
        self.deque = deque(maxlen=self.size)
        self.batch_size = batch_size
        random.seed(0)

    def push(self, transition):
        """push into the buffer"""
        self.deque.append(transition)

    def sample(self):
        """sample from the buffer"""
        samples = random.sample(self.deque, self.batch_size)
        samples = [to_tensor(item) for item in transpose_list(samples)]
        states, full_states, actions, rewards, next_states, next_full_states, dones = samples

        states = reshape_states(states)
        rewards = reshape_rewards(rewards)
        next_states = reshape_states(next_states)
        dones = reshape_dones(dones)

        return [states, full_states, actions, rewards, next_states, next_full_states, dones]

    def __len__(self):
        return len(self.deque)
