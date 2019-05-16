# main function that sets up environments
# perform training loop

from maddpg import MADDPG
from memory import ReplayBuffer
import torch
import numpy as np
import os
from unityagents import UnityEnvironment
from utils import raw_score_plotter, plotter

from collections import deque


def seeding(seed=1):
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    options = {
        'MAX_EPOCHES': 10000,
        'GAMMA': 0.95,
        'EPISODE_SIZE': 10000,
        'TAU': 0.02,
        'BATCH_SIZE': 128,
        'EPISODE_PER_UPDATE': 2,
        'PRINT_PERIOD': 5,
        'CKPT': 'model.pth',
        'SEED': 1,
        'HIDDEN_UNITS': 128,
        'NOISE': 1.0,
        'NOISE_REDUCTION': 0.9999
    }

    seeding()
    # number of parallel agents

    env = UnityEnvironment(file_name="Tennis.app")

    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    num_agents = len(env_info.agents)

    # size of each action
    action_size = brain.vector_action_space_size

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[-1]

    # number of training episodes.
    # change this to higher number to experiment. say 30000.
    number_of_episodes = options['MAX_EPOCHES']
    episode_length = options['EPISODE_SIZE']
    batch_size = options['BATCH_SIZE']

    # amplitude of OU noise
    # this slowly decreases to 0
    noise = options['NOISE']
    noise_reduction = options['NOISE_REDUCTION']

    # initialize memory buffer
    buffer = ReplayBuffer(int(500000), batch_size, 0)

    # initialize policy and critic
    maddpg = MADDPG(state_size, action_size, num_agents, discount_factor=options['GAMMA'], tau=options['TAU'])

    # how often to update the MADDPG model
    episode_per_update = options['EPISODE_PER_UPDATE']
    # training loop

    PRINT_EVERY = options['PRINT_PERIOD']
    scores_deque = deque(maxlen=100)

    # holds raw scores
    scores = []
    # holds avg scores of last 100 epsiodes
    avg_last_100 = []

    threshold = 0.5

    # use keep_awake to keep workspace from disconnecting
    for episode in range(number_of_episodes):

        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations  # get the current state (for each agent)
        episode_reward_agent0 = 0
        episode_reward_agent1 = 0

        for agent in maddpg.maddpg_agent:
            agent.noise.reset()

        for episode_t in range(episode_length):

            actions = maddpg.act(torch.tensor(state, dtype=torch.float), noise=noise)
            noise *= noise_reduction

            actions_array = torch.stack(actions).detach().numpy()

            env_info = env.step(actions_array)[brain_name]
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
            if len(buffer) > batch_size and episode % episode_per_update == 0:
                for i in range(num_agents):
                    samples = buffer.sample()
                    cl, al = maddpg.update(samples, i)
                    critic_losses.append(cl)
                    actor_losses.append(al)
                maddpg.update_targets()  # soft update the target network towards the actual networks

            # if episode_t % PRINT_EVERY == 0 and len(critic_losses) == num_agents:
            #     for i in range(num_agents):
            #         print("Agent{}\tCritic loss: {:.4f}\tActor loss: {:.4f}".format(i, critic_losses[i],
            #                                                                         actor_losses[i]))

            if np.any(done):
                # if any of the agents are done break
                break

        episode_reward = max(episode_reward_agent0, episode_reward_agent1)
        scores.append(episode_reward)
        scores_deque.append(episode_reward)
        avg_last_100.append(np.mean(scores_deque))
        # scores.append(episode_reward)
        print('\rEpisode {}\tAverage Score: {:.4f}\tScore: {:.4f}'.format(episode, avg_last_100[-1],
                                                                          episode_reward),
              end="")

        if episode % PRINT_EVERY == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(episode, avg_last_100[-1]))

        # saving successful model
        # training ends when the threshold value is reached.
        if avg_last_100[-1] >= threshold:
            save_dict_list = []

            for i in range(num_agents):
                save_dict = {'actor_params': maddpg.maddpg_agent[i].actor.state_dict(),
                             'actor_optim_params': maddpg.maddpg_agent[i].actor_optimizer.state_dict(),
                             'critic_params': maddpg.maddpg_agent[i].critic.state_dict(),
                             'critic_optim_params': maddpg.maddpg_agent[i].critic_optimizer.state_dict()}
                save_dict_list.append(save_dict)

                torch.save(save_dict_list,
                           os.path.join(options['CKPT'], 'episode-{}.pt'.format(episode)))
            # plots graphs
            raw_score_plotter(scores)
            plotter('Tennis', len(scores), avg_last_100, threshold)
            break


if __name__ == '__main__':
    main()
