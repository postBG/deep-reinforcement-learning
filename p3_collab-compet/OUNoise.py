import torch


class OUNoise(object):
    def __init__(self, action_size, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        super().__init__()
        self.action_dimension = action_size
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = torch.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = torch.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * torch.randn(len(x))
        self.state = x + dx
        return self.state * self.scale
