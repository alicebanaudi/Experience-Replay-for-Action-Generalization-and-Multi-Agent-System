import gymnasium as gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action  # <--- NEW IMPORT

class OvercookedMAEnv(gym.Env):
    """
    Wrapper for Overcooked-AI that presents it as a single 'Joint' environment.
    """
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, layout_name="cramped_room", horizon=400):
        super().__init__()
        
        # 1. Initialize the real Overcooked Game Engine
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)
        self.horizon = horizon
        
        # 2. Define Action Space (Joint Action)
        # 6 actions: North, South, East, West, Stay, Interact
        self.action_space = gym.spaces.MultiDiscrete([6, 6])

        # 3. Define Observation Space
        # FIX: Ask the MDP directly for a standard starting state to avoid NoneType errors
        dummy_state = self.mdp.get_standard_start_state()
        
        self.obs_shape = self.base_env.lossless_state_encoding_mdp(dummy_state)[0].shape
        
        # Stack agent observations
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, 
            shape=(self.obs_shape[0], self.obs_shape[1], self.obs_shape[2] * 2),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.base_env.seed(seed)
            
        self.base_env.reset()
        # Capture the state directly
        state = self.base_env.state
        return self._get_obs(state), {}

    def step(self, action):
        """
        action: [agent_1_index, agent_2_index] (e.g., [3, 4])
        """
        # --- FIX START ---
        # The library expects Direction Tuples (e.g., (0, 1)), but RL sends Indices (e.g., 3).
        # We must map them using Action.INDEX_TO_ACTION.
        joint_action = [Action.INDEX_TO_ACTION[a] for a in action]
        # -----------------

        # Step the environment
        next_state, reward, done, info = self.base_env.step(joint_action)
        
        # Sum rewards for cooperative objective
        total_reward = sum(info['shaped_r_by_agent']) 

        obs = self._get_obs(next_state)
        terminated = done
        truncated = False # Overcooked handles time limits internally
        
        return obs, total_reward, terminated, truncated, info

    def _get_obs(self, state):
        obs_tuple = self.base_env.lossless_state_encoding_mdp(state)
        return np.concatenate(obs_tuple, axis=-1).astype(np.float32)

    def render(self):
        return self.base_env.render()