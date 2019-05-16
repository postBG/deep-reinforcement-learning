import torch
from torch.optim import Adam

# add OU noise for exploration
from OUNoise import OUNoise
from models import Actor, Critic
from utils import hard_update, DEVICE


class DDPGAgent(object):
    def __init__(self, state_size, action_size, num_agents, fc1_units=64, fc2_units=64, lr_actor=1.0e-3,
                 lr_critic=1.0e-3):
        super().__init__()

        # self.actor = Actor(state_size, action_size, fc1_units, fc2_units).to(DEVICE)
        # self.critic = Critic(state_size * num_agents, action_size * num_agents, fc1_units, fc2_units).to(DEVICE)
        # self.target_actor = Actor(state_size, action_size, fc1_units, fc2_units).to(DEVICE)
        # self.target_critic = Critic(state_size * num_agents, action_size * num_agents, fc1_units, fc2_units).to(DEVICE)
        self.actor = Actor(state_size, action_size).to(DEVICE)
        self.critic = Critic(state_size, action_size, num_agents).to(DEVICE)

        self.target_actor = Actor(state_size, action_size).to(DEVICE)
        self.target_critic = Critic(state_size, action_size, num_agents).to(DEVICE)

        self.noise = OUNoise(action_size, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=0)

    def act(self, states, noise=0.0):
        states = states.to(DEVICE)
        self.actor.eval()
        action = self.actor(states) + noise * self.noise.noise()
        return torch.clamp(action, -1, 1)

    def target_act(self, states, noise=0.0):
        states = states.to(DEVICE)
        self.target_actor.to(DEVICE)
        action = self.target_actor(states) + noise * self.noise.noise()
        return torch.clamp(action, -1, 1)
