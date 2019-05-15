from unityagents import UnityEnvironment

from maddpg import MADDPG
from trainer import Trainer
from utils import seeding


def main():
    options = {
        'MAX_EPOCHES': 10000,
        'GAMMA': 0.99,
        'EPISODE_SIZE': 10000,
        'TAU': 0.001,
        'BATCH_SIZE': 128,
        'EPISODE_PER_UPDATE': 10,
        'PRINT_PERIOD': 100,
        'CKPT': 'model.pth',
        'SEED': 15,
        'HIDDEN_UNITS': 64,
        'NOISE': 2,
        'NOISE_REDUCTION': 0.995
    }

    env = UnityEnvironment(file_name="Tennis.app")

    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    print(options)
    seeding(options['SEED'])

    maddpg = MADDPG(state_size, action_size, num_agents, options['GAMMA'], options['TAU'])
    trainer = Trainer(env, maddpg, num_agents, options)
    trainer.train()


if __name__ == '__main__':
    main()
