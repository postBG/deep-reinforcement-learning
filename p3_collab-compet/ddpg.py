from models import Actor, Critic
from torch.optim import Adam

# add OU noise for exploration
from OUNoise import OUNoise
from utils import hard_update, DEVICE


class DDPGAgent:
    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64, lr_actor=1.0e-2, lr_critic=1.0e-2):
        super(DDPGAgent, self).__init__()

        self.actor = Actor(state_size, action_size, fc1_units, fc2_units).to(DEVICE)
        self.critic = Critic(state_size, action_size, fc1_units, fc2_units).to(DEVICE)
        self.target_actor = Actor(state_size, action_size, fc1_units, fc2_units).to(DEVICE)
        self.target_critic = Critic(state_size, action_size, fc1_units, fc2_units).to(DEVICE)

        self.noise = OUNoise(action_size, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)

    def act(self, states, noise=0.0):
        states = states.to(DEVICE)
        action = self.actor(states) + noise * self.noise.noise()
        return action

    def target_act(self, states, noise=0.0):
        states = states.to(DEVICE)
        action = self.target_actor(states) + noise * self.noise.noise()
        return action
