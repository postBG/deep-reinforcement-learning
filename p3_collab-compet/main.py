# main function that sets up environments
# perform training loop

from maddpg import MADDPG
from memory import ReplayBuffer
import torch
import numpy as np
import os
from unityagents import UnityEnvironment

from trainer import Trainer
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

    # initialize policy and critic
    maddpg = MADDPG(state_size, action_size, num_agents, discount_factor=options['GAMMA'], tau=options['TAU'])

    trainer = Trainer(env, maddpg, num_agents, options, brain_name)
    trainer.train()






if __name__ == '__main__':
    main()
