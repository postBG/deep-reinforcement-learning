import numpy as np
from torch.optim import Adam

from OUNoise import OUNoise
from models import Actor, Critic
from utils import hard_update, DEVICE


class DDPGAgent:
    def __init__(self, state_size, action_size, num_agents, lr_actor=1.0e-4, lr_critic=1.0e-3):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(state_size, action_size).to(DEVICE)
        self.critic = Critic(state_size, action_size, num_agents).to(DEVICE)
        self.target_actor = Actor(state_size, action_size).to(DEVICE)
        self.target_critic = Critic(state_size, action_size, num_agents).to(DEVICE)

        self.noise = OUNoise(action_size, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic)

    def act(self, states, noise=0.0):
        states = states.to(DEVICE)
        self.actor.eval()
        actions = self.actor(states).cpu().data.numpy() + noise * self.noise.noise()
        return np.clip(actions, -1, 1)

    def target_act(self, states, noise=0.0):
        states = states.to(DEVICE)
        self.target_actor.eval()
        actions = self.target_actor(states).cpu().data.numpy() + noise * self.noise.noise()
        return np.clip(actions, -1, 1)
