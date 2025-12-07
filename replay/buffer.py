# replay/buffer.py
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr, self.size = 0, 0
        
        # 1. Standard Buffer (REAL DATA) - Works exactly like before
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

        # 2. Secret Compartment (SYNTHETIC DATA)
        self.syn_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.syn_act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.syn_rew = np.zeros((capacity, 1), dtype=np.float32)
        self.syn_next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.syn_done = np.zeros((capacity, 1), dtype=np.float32)
        self.syn_ptr, self.syn_size = 0, 0

    def add(self, state, action, reward, next_state, done):
        """Standard add - touches REAL data only."""
        self.obs[self.ptr] = state
        self.act[self.ptr] = action
        self.rew[self.ptr] = reward
        self.next_obs[self.ptr] = next_state
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_synthetic(self, synthetic_batch):
        """Only used when you actually have synthetic data."""
        # ... (Same vectorized code I gave you before) ...
        # If you don't call this, the buffer remains pure.
        pass # (Paste the implementation from previous message here)

    def sample(self, batch_size, synthetic_ratio=0.0):
        """
        SAFE UPDATE:
        If synthetic_ratio is 0.0 (default), it returns ONLY real data.
        This preserves your exact original behavior.
        """
        n_syn = int(batch_size * synthetic_ratio)
        n_real = batch_size - n_syn

        if n_syn > 0 and self.syn_size == 0:
            n_real = batch_size
            n_syn = 0

        batch = {'obs': [], 'act': [], 'rew': [], 'next_obs': [], 'done': []}

        # Real Sampling (Standard)
        if n_real > 0:
            idx = np.random.randint(0, self.size, size=n_real)
            batch['obs'].append(self.obs[idx])
            batch['act'].append(self.act[idx])
            batch['rew'].append(self.rew[idx])
            batch['next_obs'].append(self.next_obs[idx])
            batch['done'].append(self.done[idx])

        # Synthetic Sampling (Only if requested)
        if n_syn > 0:
            idx = np.random.randint(0, self.syn_size, size=n_syn)
            batch['obs'].append(self.syn_obs[idx])
            batch['act'].append(self.syn_act[idx])
            batch['rew'].append(self.syn_rew[idx])
            batch['next_obs'].append(self.syn_next_obs[idx])
            batch['done'].append(self.syn_done[idx])

        # Merge and return
        return {
            "obs": np.concatenate(batch['obs']),
            "actions": np.concatenate(batch['act']), # Note: kept key 'actions' to match your agent
            "rewards": np.concatenate(batch['rew']),
            "next_obs": np.concatenate(batch['next_obs']),
            "dones": np.concatenate(batch['done'])
        }
    
    def __len__(self):
        return self.size