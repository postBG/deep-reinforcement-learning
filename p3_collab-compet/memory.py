import random
from collections import deque, namedtuple

from utils import transpose_list, to_tensor


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed=0):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.deque = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "full_state", "action", "reward", "next_state",
                                                                "full_next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, transition):
        """Add a new experience to memory."""
        self.deque.append(transition)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.deque, k=self.batch_size)
        experiences = [to_tensor(item) for item in transpose_list(experiences)]
        states, full_states, actions, rewards, next_states, next_full_states, dones = experiences

        return states, full_states, actions, rewards, next_states, next_full_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.deque)
