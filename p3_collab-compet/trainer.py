# main function that sets up environments
# perform training loop

import os
from collections import deque

import numpy as np
import torch

from buffer import ReplayBuffer
from utils import to_tensor, raw_score_plotter, plotter


class Trainer(object):
    def __init__(self, env, maddpg, num_agents, options):
        self.maddpg = maddpg

        self.env = env
        self.num_agents = num_agents

        self.max_epoches = options['MAX_EPOCHES']
        self.max_episode_len = options['EPISODE_SIZE']
        self.batch_size = options['BATCH_SIZE']
        self.episode_per_update = options.get('EPISODE_PER_UPDATE', 2)
        self.gamma = options['GAMMA']
        self.noise = options.get('NOISE', 2)
        self.noise_reduction = options.get('NOISE_REDUCTION', 0.999)
        self.print_period = options.get('PRINT_PERIOD', 10)
        self.threshold = 0.5

        self.ckpt = options['CKPT']
        self.debug = options.get('DEBUG', False)

        self.episode_rewards = []
        self.avg_rewards = []
        self.last_100_episode_rewards = deque(maxlen=100)

    def train(self):
        self.episode_rewards = []
        self.avg_rewards = []
        self.last_100_episode_rewards = deque(maxlen=100)

        buffer = ReplayBuffer(int(500000), self.batch_size)

        for episode in range(self.max_epoches):
            brain_name = self.env.brain_names[0]
            env_info = self.env.reset(train_mode=True)[brain_name]

            episode_rewards = [0, 0]

            for agent in self.maddpg.ddpg_agents:
                agent.noise.reset()

            for episode_t in range(self.max_episode_len):
                states = env_info.vector_observations
                staets_t = to_tensor(states)

                with torch.no_grad():
                    action_ts = self.maddpg.act(staets_t, noise=self.noise)
                    self.noise *= self.noise_reduction

                actions = torch.stack(action_ts).numpy()
                env_info = self.env.step(actions)[brain_name]

                next_states = env_info.vector_observations
                rewards = env_info.rewards
                dones = env_info.local_done

                for i in range(self.num_agents):
                    episode_rewards[i] += rewards[i]
                # add data to buffer

                full_states = np.concatenate(states)
                next_full_states = np.concatenate(next_states)

                buffer.push((states, full_states, actions, rewards, next_states, next_full_states, dones))

                critic_losses = []
                actor_losses = []
                if len(buffer) > self.batch_size and episode % self.episode_per_update == 0:
                    for i in range(self.num_agents):
                        samples = buffer.sample()
                        cl, al = self.maddpg.update(samples, i)
                        critic_losses.append(cl)
                        actor_losses.append(al)
                    self.maddpg.update_targets()  # soft update the target network towards the actual networks

                # if episode_t % self.print_period == 0 and len(critic_losses) == self.num_agents:
                #     for i in range(self.num_agents):
                #         print("Agent{}\tCritic loss: {:.4f}\tActor loss: {:.4f}".format(i, critic_losses[i],
                #                                                                         actor_losses[i]))

                if np.any(dones):
                    # if any of the agents are done break
                    break

            episode_reward = max(episode_rewards)
            self.episode_rewards.append(episode_reward)
            self.last_100_episode_rewards.append(episode_reward)
            self.avg_rewards.append(np.mean(self.last_100_episode_rewards))

            print('\rEpisode {}\tAverage Score: {:.4f}\tScore: {:.4f}'.format(episode, self.avg_rewards[-1],
                                                                              episode_reward), end="")

            if episode % self.print_period == 0:
                print('\rEpisode {}\tAverage Score: {:.4f}'.format(episode, self.avg_rewards[-1]))

            # saving successful model
            # training ends when the threshold value is reached.
            if self.avg_rewards[-1] >= self.threshold:
                save_dict_list = []

                for i in range(self.num_agents):
                    save_dict = {'actor_params': self.maddpg.ddpg_agents[i].actor.state_dict(),
                                 'actor_optim_params': self.maddpg.ddpg_agents[i].actor_optimizer.state_dict(),
                                 'critic_params': self.maddpg.ddpg_agents[i].critic.state_dict(),
                                 'critic_optim_params': self.maddpg.ddpg_agents[i].critic_optimizer.state_dict()}
                    save_dict_list.append(save_dict)

                    torch.save(save_dict_list,
                               os.path.join(self.ckpt, 'episode-{}.pt'.format(episode)))

                raw_score_plotter(self.episode_rewards)
                plotter('Tennis', len(self.episode_rewards), self.avg_rewards, self.threshold)
                break
