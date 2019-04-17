import torch

from utils import calculate_future_rewards, normalize_rewards


def states_to_prob(policy, states):
    """states is list of tensors that length of it is max_episodes_len.
    Shape of each of its elements is [batch_size, state_size]"""
    states = torch.stack(states, dim=0)  # states' shape = [max_episodes_len, batch_size, state_size]
    policy_input = states.view(-1, *states.shape[-3:])
    return policy(policy_input).view(states.shape[:-3])


class ClippedSurrogate(object):
    def __init__(self, device):
        self.device = device

    def __call__(self, new_probs, old_probs, states, actions, rewards, discount=0.995, epsilon=0.1, beta=0.01):
        future_rewards = calculate_future_rewards(rewards, discount)
        normalized_future_rewards = normalize_rewards(future_rewards)

        actions = self.to_tensor(actions, dtype=torch.long)
        old_probs = self.to_tensor(old_probs, dtype=torch.float)
        new_probs = self.to_tensor(new_probs, dtype=torch.float)

    def to_tensor(self, x, dtype):
        return torch.tensor(x, dtype=dtype).to(self.device)
