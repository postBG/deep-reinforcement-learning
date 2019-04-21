import torch
import torch.nn as nn

from losses import calculate_clipped_surrogate
from trajectories import collect_trajectories
from utils import DEVICE, log_density


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

        self.criterion = nn.MSELoss()

    def train(self):
        for e in range(self.max_epoches):
            trajectories = collect_trajectories(self.env, self.actor, self.max_trajectories_len)

            states, actions, old_log_probs, rewards = trajectories.get_as_tensor()
            old_advantages, old_returns = trajectories.get_gae(self.critic, self.gamma, self.lamb, as_tensor=True)

            for pe in range(self.ppo_epoches):
                self.run_one_ppo_epoch(states, actions, old_log_probs, old_advantages, old_returns)

    def run_one_ppo_epoch(self, states, actions, old_log_probs, old_advantages, old_returns):

        max_episodes_len = len(states)
        sum_loss_value = 0.0
        sum_loss_policy = 0.0
        count_steps = 0

        for batch_ofs in range(0, max_episodes_len, self.ppo_batch_size):
            sampled_states = states[batch_ofs:batch_ofs + self.ppo_batch_size]
            sampled_actions = actions[batch_ofs:batch_ofs + self.ppo_batch_size]
            sampled_old_advantages = old_advantages[batch_ofs:batch_ofs + self.ppo_batch_size].unsqueeze(-1)
            sampled_returns = old_returns[batch_ofs:batch_ofs + self.ppo_batch_size]
            sampled_old_log_probs = old_log_probs[batch_ofs:batch_ofs + self.ppo_batch_size]

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
            clipped_surrogate = calculate_clipped_surrogate(sampled_old_advantages, sampled_old_log_probs,
                                                            new_log_probs, self.ppo_eps)
            loss_policy_v = -clipped_surrogate
            loss_policy_v.backward()
            self.actor_optimizer.step()

            sum_loss_value += value_loss.item()
            sum_loss_policy += loss_policy_v.item()
            count_steps += 1
