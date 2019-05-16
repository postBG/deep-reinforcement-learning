from collections import deque, namedtuple
import numpy as np
import torch
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=0):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "full_state", "action", "reward", "next_state", "full_next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, full_state, action, reward, next_state, full_next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, full_state, action, reward, next_state, full_next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        full_states = torch.from_numpy(np.vstack([e.full_state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        full_next_states = torch.from_numpy(np.vstack([e.full_next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        # print(states[0])
        return (states, full_states, actions, rewards, next_states, full_next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)