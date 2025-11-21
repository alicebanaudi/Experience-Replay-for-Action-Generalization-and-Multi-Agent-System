# models/diffusion_trainer.py

import torch
import numpy as np


class DiffusionTrainer:
    def __init__(self, model, lr=1e-4, device="cpu"):
        self.model = model.to(device)
        self.optim = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = device

    def train_step(self, replay, batch_size=64):
        batch = replay.sample(batch_size)
        x0 = self._pack(batch).to(self.device)

        t = torch.randint(0, self.model.timesteps, (batch_size,), device=self.device)
        xt, noise = self.model.add_noise(x0, t)

        noise_pred = self.model(xt, t)
        loss = ((noise - noise_pred) ** 2).mean()

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return loss.item()

    def _pack(self, batch):
        """
        Convert replay batch → flattened vectors.
        """
        obs = torch.as_tensor(batch["obs"])
        act = torch.as_tensor(batch["actions"])
        rew = torch.as_tensor(batch["rewards"])
        next_obs = torch.as_tensor(batch["next_obs"])
        dones = torch.as_tensor(batch["dones"])

        return torch.cat([obs, act, rew, next_obs, dones], dim=1)
