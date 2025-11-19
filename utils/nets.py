# utils/nets.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(input_dim, output_dim, hidden_sizes=(256, 256), activation=nn.ReLU):
    layers = []
    last_dim = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last_dim, h))
        layers.append(activation())
        last_dim = h
    layers.append(nn.Linear(last_dim, output_dim))
    return nn.Sequential(*layers)


class GaussianPolicy(nn.Module):
    """
    Stochastic policy for SAC (outputs mean + log_std of a Gaussian),
    squashed with Tanh to [-1, 1].
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256), log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = mlp(obs_dim, 2 * act_dim, hidden_sizes)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.act_dim = act_dim

    def forward(self, obs):
        mu_logstd = self.net(obs)
        mu, log_std = mu_logstd[:, : self.act_dim], mu_logstd[:, self.act_dim:]
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t

        # log_prob for SAC entropy term
        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        mu_action = torch.tanh(mu)
        return action, log_prob, mu_action
