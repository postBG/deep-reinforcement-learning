import math
from itertools import accumulate as accum

import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def accumulate(x, discount_rate=1):
    reversed_x = x[::-1]
    accumulated_x = np.asarray(list(accum(reversed_x, lambda prev, curr: prev * discount_rate + curr)))[::-1]
    return accumulated_x


def batch_normalize(x):
    means = np.mean(x, axis=1)
    stds = np.std(x, axis=1) + 1e-10
    return (x - means[:, np.newaxis]) / stds[:, np.newaxis]


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
