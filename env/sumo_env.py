# envs/sumo_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from sumo_rl import SumoEnvironment
except ImportError:
    raise ImportError("Please install sumo-rl: pip install sumo-rl")



class SumoContinuousEnv(gym.Env):
    """
    Wraps SUMO-RL into a continuous control env:
    - Observation: flattened traffic state
    - Action: continuous vector in [-1, 1], mapped to green durations, etc.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        net_file: str,
        route_file: str,
        use_gui: bool = False,
        min_green: float = 5.0,
        max_green: float = 60.0,
    ):
        super().__init__()

        self.min_green = min_green
        self.max_green = max_green

        # underlying SUMO-RL env (single-agent for now)
        self.sumo_env = SumoEnvironment(
            net_file=net_file,
            route_file=route_file,
            out_csv_name=None,
            single_agent=True,
            use_gui=False,
        )


        # Get one observation to define shape
        obs, _ = self.sumo_env.reset()
        obs = self._process_obs(obs)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        # Assume 1-dimensional continuous action: green phase duration
        # You can extend this later (e.g. one per phase)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    def _process_obs(self, obs):
        """
        SUMO-RL often returns a dict or np array.
        We flatten + cast to float32.
        """
        if isinstance(obs, dict):
            # example: concat all values
            obs = np.concatenate([np.array(v, dtype=np.float32).ravel()
                                  for v in obs.values()], axis=0)
        else:
            obs = np.array(obs, dtype=np.float32).ravel()
        return obs

    def _map_action(self, action):
        """
        Map action in [-1, 1] to [min_green, max_green].
        """
        action = np.clip(action, -1.0, 1.0)
        scaled = self.min_green + 0.5 * (action + 1.0) * (self.max_green - self.min_green)
        return float(scaled[0])  # scalar duration

    def reset(self, seed=None, options=None):
        obs, info = self.sumo_env.reset()
        obs = self._process_obs(obs)
        return obs, info

    def step(self, action):
            # action = array([-0.3]) or tensor
        if isinstance(action, (np.ndarray, list)):
            a = float(action[0])
        else:
            a = float(action)

        # get traffic signal info
        ts = self.sumo_env.traffic_signals[self.sumo_env.ts_ids[0]]
        n_green = len(ts.green_phases)

        # map [-1,1] → [0, n_green-1]
        idx = (a + 1) / 2 * (n_green - 1)
        phase_idx = int(np.clip(round(idx), 0, n_green - 1))

        # discrete action into SUMO-RL
        next_obs, reward, terminated, truncated, info = self.sumo_env.step(phase_idx)

        next_obs = self._process_obs(next_obs)
        reward = float(reward)

        return next_obs, reward, terminated, truncated, info


    def render(self):
        # Use SUMO GUI if desired
        pass

    def close(self):
        self.sumo_env.close()
