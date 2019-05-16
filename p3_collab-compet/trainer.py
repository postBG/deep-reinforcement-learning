# main function that sets up environments
# perform training loop

import os
from collections import deque

import numpy as np
import torch

from memory import ReplayBuffer
from utils import to_tensor, raw_score_plotter, plotter


class Trainer(object):
    def __init__(self, env, maddpg, num_agents, options, brain_name):
        self.maddpg = maddpg

        self.env = env
        self.num_agents = num_agents
        self.brain_name = brain_name

        self.number_of_episodes = options['MAX_EPOCHES']
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
        self.options = options

        self.episode_rewards = []
        self.avg_rewards = []
        self.last_100_episode_rewards = deque(maxlen=100)

    def train(self):
        # initialize memory buffer
        buffer = ReplayBuffer(int(500000), self.batch_size, 0)

        # use keep_awake to keep workspace from disconnecting
        for episode in range(self.number_of_episodes):

            env_info = self.env.reset(train_mode=True)[self.brain_name]  # reset the environment
            state = env_info.vector_observations  # get the current state (for each agent)
            episode_reward_agent0 = 0
            episode_reward_agent1 = 0

            for agent in self.maddpg.maddpg_agent:
                agent.noise.reset()

            for episode_t in range(self.max_episode_len):

                actions = self.maddpg.act(torch.tensor(state, dtype=torch.float), noise=self.noise)
                self.noise *= self.noise_reduction

                actions_array = torch.stack(actions).detach().numpy()

                env_info = self.env.step(actions_array)[self.brain_name]
                next_state = env_info.vector_observations

                reward = env_info.rewards
                done = env_info.local_done

                episode_reward_agent0 += reward[0]
                episode_reward_agent1 += reward[1]
                # add data to buffer

                '''
                I can either hstack or concat two states here or do it in the update function in MADDPG
                However I think it's easier to do it here, since in the update function I have batch_size to deal with
                Although the replay buffer would have to hold more data by preprocessing and creating 2 new variables that 
                hold essentially the same info as state, and next_state, but just concatenated.
                '''
                full_state = np.concatenate((state[0], state[1]))
                full_next_state = np.concatenate((next_state[0], next_state[1]))

                buffer.add(state, full_state, actions_array, reward, next_state, full_next_state, done)

                state = next_state

                # update once after every episode_per_update
                critic_losses = []
                actor_losses = []
                if len(buffer) > self.batch_size and episode % self.episode_per_update == 0:
                    for i in range(self.num_agents):
                        samples = buffer.sample()
                        cl, al = self.maddpg.update(samples, i)
                        critic_losses.append(cl)
                        actor_losses.append(al)
                    self.maddpg.update_targets()  # soft update the target network towards the actual networks

                # if episode_t % PRINT_EVERY == 0 and len(critic_losses) == num_agents:
                #     for i in range(num_agents):
                #         print("Agent{}\tCritic loss: {:.4f}\tActor loss: {:.4f}".format(i, critic_losses[i],
                #                                                                         actor_losses[i]))

                if np.any(done):
                    # if any of the agents are done break
                    break

            episode_reward = max(episode_reward_agent0, episode_reward_agent1)
            self.episode_rewards.append(episode_reward)
            self.last_100_episode_rewards.append(episode_reward)
            self.avg_rewards.append(np.mean(self.last_100_episode_rewards))
            # scores.append(episode_reward)
            print('\rEpisode {}\tAverage Score: {:.4f}\tScore: {:.4f}'.format(episode, self.avg_rewards[-1],
                                                                              episode_reward),
                  end="")

            if episode % self.print_period == 0:
                print('\rEpisode {}\tAverage Score: {:.4f}'.format(episode, self.avg_rewards[-1]))

            # saving successful model
            # training ends when the threshold value is reached.
            if self.avg_rewards[-1] >= self.threshold:
                save_dict_list = []

                for i in range(self.num_agents):
                    save_dict = {'actor_params': self.maddpg.maddpg_agent[i].actor.state_dict(),
                                 'actor_optim_params': self.maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                                 'critic_params': self.maddpg.maddpg_agent[i].critic.state_dict(),
                                 'critic_optim_params': self.maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                    save_dict_list.append(save_dict)

                    torch.save(save_dict_list,
                               os.path.join(self.ckpt, 'episode-{}.pt'.format(episode)))

                raw_score_plotter(self.episode_rewards)
                plotter('Tennis', len(self.episode_rewards), self.avg_rewards, self.threshold)
                break
