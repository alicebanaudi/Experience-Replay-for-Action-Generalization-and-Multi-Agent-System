# agents/redq_sac.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List

from utils.nets import mlp, GaussianPolicy
import copy


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, 1, hidden_sizes)

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.net(x)


class REDQSACAgent:
    """
    REDQ-SAC + optional PGR (Policy Gradient Regularization)
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        device: str = "cpu",
        num_q_nets: int = 10,
        num_q_samples: int = 2,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        utd_ratio: int = 20,
        batch_size: int = 256,
        target_entropy: float = None,
        use_pgr: bool = False,
        pgr_coef: float = 0.0,
        pgr_every=10, 
        pgr_batch=32,
    ):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.lr = lr
        self.utd_ratio = utd_ratio
        self.batch_size = batch_size
        self.num_q_nets = num_q_nets
        self.num_q_samples = num_q_samples

        self.use_pgr = use_pgr
        self.pgr_coef = pgr_coef
        self.pgr_every = pgr_every
        self.pgr_batch = pgr_batch
        self.update_steps = 0

        # Actor
        self.actor = GaussianPolicy(obs_dim, act_dim).to(device)

        # --------------------
        # PGR: reference policy
        # --------------------
        if self.use_pgr:
            self.actor_real = copy.deepcopy(self.actor)
            self.actor_real.requires_grad_(False)

        # Q ensemble + target ensemble
        self.q_nets: List[QNetwork] = [
            QNetwork(obs_dim, act_dim).to(device) for _ in range(num_q_nets)
        ]
        self.q_targets: List[QNetwork] = [
            QNetwork(obs_dim, act_dim).to(device) for _ in range(num_q_nets)
        ]

        for q, q_targ in zip(self.q_nets, self.q_targets):
            q_targ.load_state_dict(q.state_dict())
            q_targ.requires_grad_(False)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opts = [
            torch.optim.Adam(q.parameters(), lr=lr) for q in self.q_nets
        ]

        # Entropy coefficient alpha (auto-tuned)
        if target_entropy is None:
            target_entropy = -act_dim  # SAC default
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs: np.ndarray, deterministic: bool = False):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if deterministic:
                mu, log_std = self.actor(obs_t)
                action = torch.tanh(mu)
            else:
                action, _, _ = self.actor.sample(obs_t)
        return action.cpu().numpy()[0]

    def _soft_update_targets(self):
        with torch.no_grad():
            for q, q_targ in zip(self.q_nets, self.q_targets):
                for p, p_targ in zip(q.parameters(), q_targ.parameters()):
                    p_targ.data.mul_(1 - self.tau)
                    p_targ.data.add_(self.tau * p.data)

    def update(self, replay_buffer):
        logs = {}
        for _ in range(self.utd_ratio):
            self.update_steps += 1
            batch = replay_buffer.sample(self.batch_size)

            obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
            actions = torch.as_tensor(batch["actions"], dtype=torch.float32, device=self.device)
            rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device)
            next_obs = torch.as_tensor(batch["next_obs"], dtype=torch.float32, device=self.device)
            dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device)

            # ----------------------- Q update -------------------------
            with torch.no_grad():
                next_action, next_log_prob, _ = self.actor.sample(next_obs)

                q_targ_vals = []
                idxs = np.random.choice(self.num_q_nets, self.num_q_samples, replace=False)

                for i in idxs:
                    q_targ_vals.append(self.q_targets[i](next_obs, next_action))

                q_targ_vals = torch.cat(q_targ_vals, dim=1)
                min_q_next = q_targ_vals.min(dim=1, keepdim=True)[0]

                target_q = rewards + self.gamma * (1 - dones) * \
                           (min_q_next - self.alpha * next_log_prob)

            q_losses = []
            for q_net, q_opt in zip(self.q_nets, self.q_opts):
                q_pred = q_net(obs, actions)
                q_loss = F.mse_loss(q_pred, target_q)
                q_loss = torch.clamp(q_loss, max=50.0)

                q_opt.zero_grad()
                q_loss.backward()

                # ===== Gradient Clipping for Q-net =====
                torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)

                q_opt.step()

                q_losses.append(q_loss.item())


            # ----------------------- Actor update ----------------------
            new_actions, log_probs, _ = self.actor.sample(obs)

            q_values_new = torch.cat(
                [q_net(obs, new_actions) for q_net in self.q_nets], dim=1
            )
            mean_q_new = q_values_new.mean(dim=1, keepdim=True)

            actor_loss = (self.alpha * log_probs - mean_q_new).mean()
            actor_loss = torch.clamp(actor_loss, max=50.0)


            # ----------------------- PGR (lightweight version) ----------------------
            if (
                self.use_pgr
                and self.pgr_coef > 0.0
                and self.update_steps % self.pgr_every == 0
            ):
                # Subsample a smaller batch for PGR
                idx = torch.randperm(obs.size(0))[: self.pgr_batch]
                obs_pgr = obs[idx]

                mu_pgr, _ = self.actor(obs_pgr)

                pgr_loss = 0.0
                for m in mu_pgr:
                    grads = torch.autograd.grad(
                        outputs=m,
                        inputs=list(self.actor.parameters()),
                        retain_graph=True,
                        create_graph=True
                    )
                    for g in grads:
                        pgr_loss += (g**2).sum()

                # Add scaled PGR penalty
                actor_loss = actor_loss + self.pgr_coef * pgr_loss

            # ===== Gradient Clipping for Actor =====
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)

            self.actor_opt.step()


            # ----------------------- Alpha update ----------------------
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # ----------------------- Target update ---------------------
            self._soft_update_targets()

        logs["q_loss"] = np.mean(q_losses)
        logs["actor_loss"] = actor_loss.item()
        logs["alpha"] = self.alpha.item()
        return logs
