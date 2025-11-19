# models/diffusion.py

import torch
import torch.nn as nn


def mlp(sizes, activation=nn.ReLU, output_activation=None):
    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i+1]),
                   act() if act else nn.Identity()]
    return nn.Sequential(*layers)


class DiffusionModel(nn.Module):
    """
    Denoising Diffusion for transitions.
    Input: x_t concatenated with timestep t.
    Output: predicted noise.
    """

    def __init__(self, dim, hidden=256, timesteps=50):
        super().__init__()
        self.dim = dim
        self.timesteps = timesteps
        self.net = mlp([dim + 1, hidden, hidden, dim])

    def forward(self, x, t):
        t = t.float().unsqueeze(1) / self.timesteps
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0)
        alpha = (1 - 0.02) ** t.float().unsqueeze(1)
        xt = alpha * x0 + (1 - alpha) * noise
        return xt, noise
