# env/sumo_env.py

import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from sumo_rl import SumoEnvironment
except ImportError:
    raise ImportError("Please install sumo-rl: pip install sumo-rl")

import traci  # we use TraCI directly for congestion checks

# Reduce noise in logs
os.environ["SUMO_NO_WARNINGS"] = "1"


class SumoContinuousEnv(gym.Env):
    """
    Continuous wrapper around SUMO-RL.
    - Continuous action in [-1, 1] -> discrete traffic-light phase
    - Fixed episode length (max_steps)
    - Early termination when too many vehicles are in the network
      (to avoid congestion and long SUMO steps)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        net_file: str,
        route_file: str,
        use_gui: bool = False,
        min_green: float = 5.0,
        max_green: float = 60.0,
        max_steps: int = 60,        # shorter episodes to avoid buildup
        max_vehicles: int = 250,    # cutoff if too many cars in network
    ):
        super().__init__()

        self.min_green = min_green
        self.max_green = max_green
        self.max_steps = max_steps
        self.max_vehicles = max_vehicles
        self.current_steps = 0

        self.net_file = net_file
        self.route_file = route_file
        self.use_gui = use_gui

        # Create underlying SUMO-RL env
        self.sumo_env = SumoEnvironment(
            net_file=self.net_file,
            route_file=self.route_file,
            out_csv_name=None,
            single_agent=True,
            use_gui=self.use_gui,
        )

        # Give SUMO a tiny moment to start
        time.sleep(0.05)

        # Get observation shape
        obs, _ = self.sumo_env.reset()
        obs = self._process_obs(obs)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )

        # Continuous action in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

    # -------------------------
    # Utils
    # -------------------------

    def _process_obs(self, obs):
        if isinstance(obs, dict):
            obs = np.concatenate(
                [np.array(v, dtype=np.float32).ravel() for v in obs.values()],
                axis=0
            )
        else:
            obs = np.array(obs, dtype=np.float32).ravel()
        return obs

    def _map_action_to_phase(self, action):
        """
        Map a scalar action in [-1, 1] to a discrete green phase index.
        """
        if isinstance(action, (np.ndarray, list)):
            a = float(action[0])
        else:
            a = float(action)

        # Get number of green phases
        ts = self.sumo_env.traffic_signals[self.sumo_env.ts_ids[0]]
        n_green = len(ts.green_phases)

        # Map [-1,1] → [0, n_green-1]
        idx = (a + 1.0) / 2.0 * (n_green - 1)
        phase_idx = int(np.clip(round(idx), 0, n_green - 1))
        return phase_idx

    def _too_many_vehicles(self):
        """
        Use TraCI to estimate how many vehicles are in / expected in the network.
        If it exceeds a threshold, we terminate the episode early to avoid jams.
        """
        try:
            # minExpectedNumber = vehicles running + waiting to depart
            num = traci.simulation.getMinExpectedNumber()
        except Exception:
            # If TraCI is not ready for some reason, don't crash.
            return False

        return num > self.max_vehicles

    # -------------------------
    # Gym API
    # -------------------------

    def reset(self, seed=None, options=None):
        self.current_steps = 0
        obs, info = self.sumo_env.reset()
        return self._process_obs(obs), info

    def step(self, action):
        # Map continuous → discrete phase
        phase_idx = self._map_action_to_phase(action)

        # Step SUMO
        next_obs, reward, terminated, truncated, info = self.sumo_env.step(phase_idx)
        next_obs = self._process_obs(next_obs)
        reward = float(reward)

        # --- Episode bookkeeping ----
        self.current_steps += 1
        time_limit = self.current_steps >= self.max_steps

        # --- Congestion cutoff ----
        congestion = self._too_many_vehicles()

        done = terminated or truncated or time_limit or congestion

        # You can log congestion events if you want:
        # if congestion:
        #     print("[Env] Ending episode early due to congestion.")

        return next_obs, reward, done, False, info

    def close(self):
        self.sumo_env.close()