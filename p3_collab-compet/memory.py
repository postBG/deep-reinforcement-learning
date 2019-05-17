import random
from collections import deque

import torch

from utils import transpose_list, to_tensor


def reshape_to_iter_by_agents_3d(tensors, num_agents):
    return torch.stack([tensors[:, i, :] for i in range(num_agents)])


def reshape_to_iter_by_agents_2d(tensors, num_agents):
    return torch.stack([tensors[:, i] for i in range(num_agents)])


class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, num_agents, seed=0):
        super().__init__()
        self.deque = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.num_agents = num_agents
        self.seed = random.seed(seed)

    def add(self, transition):
        """Add a new experience to memory."""
        self.deque.append(transition)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.deque, k=self.batch_size)
        experiences = [to_tensor(item) for item in transpose_list(experiences)]
        states, full_states, actions, rewards, next_states, next_full_states, dones = experiences

        states = reshape_to_iter_by_agents_3d(states, self.num_agents)
        rewards = reshape_to_iter_by_agents_2d(rewards, self.num_agents)
        next_states = reshape_to_iter_by_agents_3d(next_states, self.num_agents)
        dones = reshape_to_iter_by_agents_2d(dones, self.num_agents)

        return states, full_states, actions, rewards, next_states, next_full_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.deque)
