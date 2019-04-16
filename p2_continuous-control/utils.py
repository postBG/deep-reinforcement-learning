import numpy as np


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
