# models/synthetic_generator.py

import torch
import numpy as np


class SyntheticGenerator:
    def __init__(self, model, device="cpu"):
        self.model = model
        self.device = device

    @torch.no_grad()
    def sample(self, n_samples):
        T = self.model.timesteps
        x = torch.randn(n_samples, self.model.dim).to(self.device)

        for t in reversed(range(T)):
            t_batch = torch.full((n_samples,), t, device=self.device)
            noise_pred = self.model(x, t_batch)
            alpha = (1 - 0.02) ** t
            x = (x - (1 - alpha) * noise_pred) / alpha

        return x.cpu().numpy()
