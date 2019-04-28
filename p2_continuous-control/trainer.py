import random
from collections import deque

import torch
import torch.nn as nn
import numpy as np

from losses import calculate_clipped_surrogate
from trajectories import collect_trajectories
from utils import DEVICE, log_density, print_ratio_for_debugging


class Trainer(object):
    def __init__(self, env, actor, critic, actor_optimizer, critic_optimizer, options):
        self.env = env
        self.device = DEVICE

        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        self.max_epoches = options['MAX_EPOCHES']
        self.max_trajectories_len = options['TRAJECTORY_SIZE']
        self.ppo_batch_size = options['PPO_BATCH_SIZE']
        self.ppo_epoches = options['PPO_EPOCHES']
        self.ppo_eps = options['PPO_EPS']
        self.gamma = options['GAMMA']
        self.lamb = options['GAE_LAMBDA']
        self.beta = options['ENTROPY_WEIGHT']

        self.actor_ckpt = options['ACTOR_CKPT']
        self.critic_ckpt = options['CRITIC_CKPT']
        self.debug = options.get('DEBUG', False)

        self.criterion = nn.MSELoss()

    def train(self):
        mean_rewards = []
        last_100_mean_rewards = deque(maxlen=100)

        for i_episode in range(1, self.max_epoches + 1):
            trajectories = collect_trajectories(self.env, self.actor, self.max_trajectories_len)
            total_rewards = trajectories.total_rewards()
            mean_reward = np.mean(total_rewards)

            mean_rewards.append(mean_reward)
            last_100_mean_rewards.append(mean_reward)

            states, actions, _, rewards = trajectories.get_as_tensor()

            old_mu, old_std, old_log_std = self.actor(states)
            old_mu, old_std, old_log_std = old_mu.detach(), old_std.detach(), old_log_std.detach()
            old_log_probs = log_density(actions, old_mu, old_std, old_log_std)
            old_advantages, old_returns = trajectories.get_gae(self.critic, self.gamma, self.lamb, as_tensor=True)

            sum_value_loss = 0
            sum_policy_loss = 0
            for pe in range(self.ppo_epoches):
                self.beta *= 0.995
                value_loss, policy_loss = self.run_one_ppo_epoch(states, actions, old_log_probs, old_advantages,
                                                                 old_returns, self.ppo_eps, self.beta, i_episode)
                sum_value_loss += value_loss
                sum_policy_loss += policy_loss

            if i_episode % 10 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}\tMean Value loss: {:.5f}\tMean policy loss {:.5f}'.format(
                    i_episode, np.mean(last_100_mean_rewards),
                    sum_value_loss / self.ppo_epoches, sum_policy_loss / self.ppo_epoches))

            if np.mean(last_100_mean_rewards) >= 30.0:
                print('Env solved in {:d} episodes!\tAvg. Score: {:.2f}'.format(i_episode - 100,
                                                                                np.mean(last_100_mean_rewards)))
                torch.save(self.actor.state_dict(), self.actor_ckpt)
                torch.save(self.critic.state_dict(), self.critic_ckpt)
                break

        return mean_rewards, last_100_mean_rewards

    def run_one_ppo_epoch(self, states, actions, old_log_probs, old_advantages, old_returns, epsilon, beta, i_episode):

        max_episodes_len = len(states)
        sum_loss_value = 0.0
        sum_loss_policy = 0.0
        count_steps = 0
        start_index = random.randint(0, self.ppo_batch_size - 1)

        for batch_ofs in range(start_index, max_episodes_len, self.ppo_batch_size):
            end_index = min(batch_ofs + self.ppo_batch_size, max_episodes_len)
            sampled_states = states[batch_ofs:end_index]
            sampled_actions = actions[batch_ofs:end_index]
            sampled_old_advantages = old_advantages[batch_ofs:end_index].unsqueeze(-1)
            sampled_returns = old_returns[batch_ofs:end_index]
            sampled_old_log_probs = old_log_probs[batch_ofs:end_index]

            # critic training
            self.critic_optimizer.zero_grad()
            new_values = self.critic(sampled_states)
            value_loss = self.criterion(new_values.squeeze(-1), sampled_returns)
            value_loss.backward()
            self.critic_optimizer.step()

            # actor training
            self.actor_optimizer.zero_grad()
            mu, std, log_std = self.actor(sampled_states)

            new_log_probs = log_density(sampled_actions, mu, std, log_std)
            if self.debug:
                print_ratio_for_debugging(i_episode, new_log_probs, sampled_old_log_probs)
            clipped_surrogate = calculate_clipped_surrogate(sampled_old_advantages, sampled_old_log_probs,
                                                            new_log_probs, epsilon)
            # loss_policy = -(clipped_surrogate + beta * calculate_entropy(new_log_probs))
            loss_policy = -clipped_surrogate
            loss_policy.backward()
            self.actor_optimizer.step()

            sum_loss_value += value_loss.item()
            sum_loss_policy += loss_policy.item()
            count_steps += 1

        return sum_loss_value / count_steps, sum_loss_policy / count_steps
