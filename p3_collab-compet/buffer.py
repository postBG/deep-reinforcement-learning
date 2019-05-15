import random
from collections import deque

from utils import transpose_list


class ReplayBuffer(object):
    def __init__(self, size, batch_size):
        super().__init__()
        self.size = size
        self.deque = deque(maxlen=self.size)
        self.batch_size = batch_size

    def push(self, transition):
        """push into the buffer"""

        replay_tuples = transpose_list(transition)

        for replay_tuple in replay_tuples:
            self.deque.append(replay_tuple)

    def sample(self):
        """sample from the buffer"""
        samples = random.sample(self.deque, self.batch_size)
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)
