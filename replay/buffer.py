# replay/buffer.py

import numpy as np
from typing import NamedTuple, List


class Transition(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)

        self.size = 0
        self.ptr = 0

    def add(self, state, action, reward, next_state, done):
        self.obs_buf[self.ptr] = state
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_state
        self.done_buf[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(
            obs=self.obs_buf[idxs],
            actions=self.act_buf[idxs],
            rewards=self.rew_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            dones=self.done_buf[idxs],
        )
        return batch
    
    def add_synthetic(self, synthetic_batch):
        """
    Inserts a batch of synthetic transitions into the replay buffer.

    synthetic_batch: np.ndarray of shape [N, transition_dim]
        where transition_dim = obs_dim + act_dim + obs_dim + 1 + 1
          → [state | action | reward | next_state | done]
    """

        obs_dim = self.obs_buf.shape[1]
        act_dim = self.act_buf.shape[1]

        for x in synthetic_batch:
            # Extract components
            s = x[:obs_dim]
            a = x[obs_dim : obs_dim + act_dim]
            r = float(x[obs_dim + act_dim])
            ns = x[obs_dim + act_dim + 1 : obs_dim + act_dim + 1 + obs_dim]
            d = bool(x[-1] > 0.5)  # convert to boolean

            # Insert into buffer
            self.add(s, a, r, ns, d)



    def __len__(self):
        return self.size
