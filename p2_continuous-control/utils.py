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


def log_density(x, mu, std, log_std):
    var = std.pow(2)
    log_dens = -(x - mu).pow(2) / (2 * var) - 0.5 * math.log(2 * math.pi) - log_std
    return log_dens.sum(1, keepdim=True)


def to_tensor_long(numpy_array):
    return torch.LongTensor(numpy_array).to(device)


def to_tensor(numpy_array):
    return torch.Tensor(numpy_array).to(device)


def sample_actions(mu, std):
    return torch.clamp(torch.normal(mu, std), -1, 1)


def collect_trajectories(env, actor, tmax=2049):
    # get the default brain
    brain_name = env.brain_names[0]

    state_list = []
    action_list = []
    old_log_probs = []
    reward_list = []

    # reset the environment
    env_info = env.reset()[brain_name]

    for t in range(tmax):
        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        states = env_info.vector_observations
        states_t = to_tensor(states)
        with torch.no_grad():
            mu_t, std_t, log_std_t = actor(states_t)
            actions_t = sample_actions(mu_t, std_t)
            actions_log_prob_t = log_density(actions_t, mu_t, std_t, log_std_t)
            actions_log_prob = actions_log_prob_t.cpu().numpy()

            actions = actions_t.cpu().numpy()
            env_info = env.step(actions)[brain_name]

        # store the result
        state_list.append(states)
        action_list.append(actions)
        old_log_probs.append(actions_log_prob)
        reward_list.append(env_info.rewards)

        # stop if any of the trajectories is done
        # we want all the lists to be retangular
        if np.any(env_info.local_done):
            break

    # return pi_theta, states, actions, rewards, probability
    return np.asarray(state_list), np.asarray(action_list), np.asarray(old_log_probs), np.asarray(reward_list)
