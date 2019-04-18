import math

import numpy as np
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def calculate_future_rewards(rewards, gamma):
    """rewards is list of episodes where length of list is max_episode_length.
    And each elements is rewards of each batches. So, shape of the rewards becomes [max_episode_length, batch_size]"""
    discounts = gamma ** np.arange(len(rewards))
    discounted_rewards = np.asarray(rewards) * discounts[:, np.newaxis]
    future_rewards = discounted_rewards[::-1].cumsum(axis=0)[::-1]
    return future_rewards


def normalize_rewards(rewards):
    means = np.mean(rewards, axis=1)
    stds = np.std(rewards, axis=1) + 1e-10
    return (rewards - means[:, np.newaxis]) / stds[:, np.newaxis]


def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_dens = -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - logstd
    return log_dens.sum(1, keepdim=True)


def to_tensor_long(numpy_array):
    return torch.LongTensor(numpy_array).to(device)


def to_tensor(numpy_array):
    return torch.Tensor(numpy_array).to(device)


def sample_actions(mu, std):
    actions = torch.normal(mu, std)
    actions = actions.cpu().data.numpy()
    return actions
