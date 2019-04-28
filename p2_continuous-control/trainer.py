from collections import deque

import numpy as np
import torch
import torch.nn as nn

from losses import calculate_clipped_surrogate
from trajectories import collect_trajectories
from utils import DEVICE, print_ratio_for_debugging, to_tensor_long


class Trainer(object):
    def __init__(self, env, model, actor_optimizer, options):
        self.env = env
        self.device = DEVICE

        self.model = model.to(self.device)

        self.model_optimizer = actor_optimizer

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
            trajectories = collect_trajectories(self.env, self.model, self.max_trajectories_len)
            total_rewards = trajectories.total_rewards()
            mean_reward = np.mean(total_rewards)

            mean_rewards.append(mean_reward)
            last_100_mean_rewards.append(mean_reward)

            states, actions, _, rewards = trajectories.get_as_tensor()

            with torch.no_grad():
                _, old_log_probs, _, values = self.model(states, actions)
                values = values.cpu().numpy()
            old_advantages, old_returns = trajectories.get_gae(values, self.gamma, self.lamb, as_tensor=True)

            sum_value_loss = 0
            sum_policy_loss = 0
            for pe in range(self.ppo_epoches):
                max_episodes_len = len(states)
                arr = np.arange(max_episodes_len)
                value_loss, policy_loss = self.run_one_ppo_epoch(states, actions, old_log_probs, old_advantages,
                                                                 old_returns, self.ppo_eps, i_episode, arr, self.beta)
                sum_value_loss += value_loss
                sum_policy_loss += policy_loss

            if i_episode % 10 == 0:
                print(('\rEpisode {}\tRecent Score: {:.2f}\tAverage Score: {:.2f}' +
                       '\tMean Value loss: {:.5f}\tMean policy loss {:.5f}').format(
                    i_episode, mean_reward, np.mean(last_100_mean_rewards),
                    sum_value_loss / self.ppo_epoches, sum_policy_loss / self.ppo_epoches))

            if np.mean(last_100_mean_rewards) >= 30.0:
                print('Env solved in {:d} episodes!\tAvg. Score: {:.2f}'.format(i_episode - 100,
                                                                                np.mean(last_100_mean_rewards)))
                torch.save(self.model.state_dict(), self.actor_ckpt)
                torch.save(self.critic.state_dict(), self.critic_ckpt)
                break

        return mean_rewards, last_100_mean_rewards

    def run_one_ppo_epoch(self, states, actions, old_log_probs, old_advantages, old_returns, epsilon, i_episode, arr,
                          beta):

        max_episodes_len = len(states)
        np.random.shuffle(arr)

        sum_value_loss = 0.0
        sum_actor_loss = 0.0
        count_steps = 0

        for batch_ofs in range(0, max_episodes_len, self.ppo_batch_size):
            batch_indices = to_tensor_long(arr[batch_ofs:batch_ofs + self.ppo_batch_size])

            sampled_states = states[batch_indices]
            sampled_actions = actions[batch_indices]
            sampled_old_advantages = old_advantages[batch_indices]
            sampled_returns = old_returns[batch_indices]
            sampled_old_log_probs = old_log_probs[batch_indices]

            _, new_log_probs, entropy, new_values = self.model(sampled_states, sampled_actions)

            if self.debug:
                print_ratio_for_debugging(i_episode, new_log_probs, sampled_old_log_probs)
            clipped_surrogate = calculate_clipped_surrogate(sampled_old_advantages, sampled_old_log_probs,
                                                            new_log_probs, epsilon)
            actor_loss = -(clipped_surrogate + beta * entropy.mean())
            value_loss = self.criterion(new_values, sampled_returns)
            loss = actor_loss + value_loss

            self.model_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
            self.model_optimizer.step()

            sum_value_loss += value_loss.item()
            sum_actor_loss += actor_loss.item()
            count_steps += 1

        return sum_value_loss / count_steps, sum_actor_loss / count_steps
