import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# The model with only 1 batchnorm layer performed better for some reason.

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, fc_units=256, fc_units1=128, fc_units2=128, fc_units3=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        # self.seed = torch.manual_seed(seed)

        # only the first layer has batch normalization
        self.bn = nn.BatchNorm1d(state_size)
        self.bn1 = nn.BatchNorm1d(fc_units)
        self.bn2 = nn.BatchNorm1d(fc_units1)
        self.bn3 = nn.BatchNorm1d(fc_units2)
        self.bn4 = nn.BatchNorm1d(fc_units3)

        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units1)
        self.fc3 = nn.Linear(fc_units1, fc_units2)
        self.fc4 = nn.Linear(fc_units2, fc_units3)
        self.fc5 = nn.Linear(fc_units3, action_size)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        # only the first layer has batch normalization
        x = F.relu(self.fc1(self.bn(state)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return torch.tanh(self.fc5(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, num_agent, fc_units=256, fc_units1=128, fc_units2=128, fc_units3=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        # self.seed = torch.manual_seed(seed)
        # only the first layer has batch normalization
        self.bn = nn.BatchNorm1d(state_size * num_agent)
        self.bn1 = nn.BatchNorm1d(fc_units)
        self.bn2 = nn.BatchNorm1d(fc_units1)
        self.bn3 = nn.BatchNorm1d(fc_units2)
        self.bn4 = nn.BatchNorm1d(fc_units3)

        self.fc1 = nn.Linear((state_size + action_size) * num_agent, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units1)
        self.fc3 = nn.Linear(fc_units1, fc_units2)
        self.fc4 = nn.Linear(fc_units2, fc_units3)
        self.fc5 = nn.Linear(fc_units3, 1)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(*hidden_init(self.fc4))
        self.fc5.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""

        x = torch.cat((self.bn(state), action), dim=1)
        # only the first layer has batch normalization
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))

        return self.fc5(x).squeeze(-1)
