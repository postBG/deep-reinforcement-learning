import random
from collections import deque

from utils import transpose_list


class ReplayBuffer(object):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self, transition):
        """push into the buffer"""

        replay_tuples = transpose_list(transition)

        for replay_tuple in replay_tuples:
            self.deque.append(replay_tuple)

    def sample(self, batchsize):
        """sample from the buffer"""
        samples = random.sample(self.deque, batchsize)

        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)
