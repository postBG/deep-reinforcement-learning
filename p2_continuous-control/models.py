import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    return x * F.sigmoid(x)


class ActorCritic(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super().__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        self.actor_fc = nn.Linear(fc2_units, action_size)
        self.critic_fc = nn.Linear(fc2_units, 1)

        self.std = nn.Parameter(torch.zeros(action_size))

    def forward(self, state, actions=None):
        """Build a network that maps state -> actions mu."""
        h = swish(self.fc1(state))
        h = swish(self.fc2(h))

        mu = F.tanh(self.actor_fc(h))
        values = self.critic_fc(h).squeeze(-1)

        dist = torch.distributions.Normal(mu, F.softplus(self.std))

        if actions is None:
            actions = dist.sample()
        log_prob = dist.log_prob(actions)
        log_prob = torch.sum(log_prob, dim=-1)
        entropy = torch.sum(dist.entropy(), dim=-1)
        return actions, log_prob, entropy, values
