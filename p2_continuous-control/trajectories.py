import math

import numpy as np
import torch

from utils import to_tensor, accumulate, batch_normalize


def calculate_future_rewards(rewards, gamma):
    """rewards is list of episodes where length of list is max_episode_length.
    And each elements is rewards of each batches. So, shape of the rewards becomes [max_episode_length, batch_size]"""
    return accumulate(rewards, gamma)


def calculate_td_errors(rewards, values, gamma):
    """rewards/values is list of reward/value where length of list is max_episode_length.
        And each elements is rewards of each batches.
        So, shape of the rewards becomes [max_episode_length, batch_size]"""
    rewards = np.asarray(rewards) if type(rewards) is list else rewards
    _, batch_size = rewards.shape
    next_values = np.append(values[1:], np.zeros([1, batch_size]), axis=0)
    td_errors = rewards + gamma * next_values - values
    return td_errors


def calculate_gae(rewards, values, gamma, lamb, normalize=True):
    td_errors = calculate_td_errors(rewards, values, gamma)
    discount_rate = gamma * lamb
    advantages = accumulate(td_errors, discount_rate)
    future_returns = calculate_future_rewards(rewards, gamma)

    if normalize:
        advantages = batch_normalize(advantages)

    return advantages, future_returns


class Trajectories(object):
    def __init__(self, states: np.ndarray, actions: np.ndarray, old_log_probs: np.ndarray, rewards: np.ndarray):
        self.states = states
        self.actions = actions
        self.old_log_probs = old_log_probs
        self.rewards = rewards

        self.states_t = to_tensor(self.states)
        self.actions_t = to_tensor(self.actions)
        self.old_log_probs_t = to_tensor(self.old_log_probs)
        self.rewards_t = to_tensor(self.rewards)

    def get_as_numpy(self):
        return self.states, self.actions, self.old_log_probs, self.rewards

    def get_as_tensor(self):
        return self.states_t, self.actions_t, self.old_log_probs_t, self.rewards_t

    def get_gae(self, values, gamma, lamb, normalize=True, as_tensor=True):
        advantages, returns = calculate_gae(self.rewards, values, gamma, lamb, normalize=normalize)

        advantages = to_tensor(advantages.copy()) if as_tensor else advantages
        returns = to_tensor(returns.copy()) if as_tensor else returns
        return advantages, returns

    def total_rewards(self):
        return np.sum(self.rewards, axis=0)

    def __len__(self):
        return len(self.states)

    @property
    def batch_size(self):
        return self.states.shape[1]


def collect_trajectories(env, model, max_episodes_len=None):
    # get the default brain
    brain_name = env.brain_names[0]
    max_episodes_len = max_episodes_len if max_episodes_len else math.inf

    state_list = []
    action_list = []
    old_log_probs = []
    reward_list = []

    # reset the environment
    env_info = env.reset()[brain_name]

    is_train = model.training
    model.eval()
    for t in range(max_episodes_len):
        # probs will only be used as the pi_old
        # no gradient propagation is needed
        # so we move it to the cpu
        states = env_info.vector_observations
        states_t = to_tensor(states)
        with torch.no_grad():
            actions_t, actions_log_prob_t, entropy, values = model(states_t)
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

    if is_train:
        model.train()

    return Trajectories(np.asarray(state_list), np.asarray(action_list), np.asarray(old_log_probs),
                        np.asarray(reward_list))
