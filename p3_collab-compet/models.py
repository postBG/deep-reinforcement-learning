import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import DEVICE


def swish(x):
    return x * F.sigmoid(x)


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.actor_fc = nn.Linear(fc2_units, action_size)

        self.std = nn.Parameter(torch.zeros(action_size))

    def forward(self, states, actions=None):
        h = swish(self.fc1(states))
        h = swish(self.fc2(h))
        mu = F.tanh(self.actor_fc(h))

        dist = torch.distributions.Normal(mu, F.softplus(self.std))

        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)
        return actions, log_prob, entropy


class Critic(nn.Module):
    """Critic Model."""

    def __init__(self, state_size, action_size, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.critic_fc = nn.Linear(fc2_units, 1)

    def forward(self, states, actions):
        """Build a network that maps states, actions -> q values"""
        x = torch.cat([states, actions], dim=1).to(DEVICE)
        h = swish(self.fc1(x))
        h = swish(self.fc2(h))
        q_values = self.critic_fc(h).squeeze(-1)

        return q_values
