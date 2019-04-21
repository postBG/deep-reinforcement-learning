import numpy as np

from utils import accumulate, batch_normalize


def calculate_future_rewards(rewards, gamma):
    """rewards is list of episodes where length of list is max_episode_length.
    And each elements is rewards of each batches. So, shape of the rewards becomes [max_episode_length, batch_size]"""
    discounts = gamma ** np.arange(len(rewards))
    discounted_rewards = np.asarray(rewards) * discounts[:, np.newaxis]
    future_rewards = discounted_rewards[::-1].cumsum(axis=0)[::-1]
    return future_rewards


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

    if normalize:
        advantages = batch_normalize(advantages)

    estimated_returns = advantages + values
    return advantages, estimated_returns
