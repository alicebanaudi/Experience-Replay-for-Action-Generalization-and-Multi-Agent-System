import gymnasium as gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action

class OvercookedMAEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, layout_name="cramped_room", horizon=400):
        super().__init__()
        
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        self.base_env = OvercookedEnv.from_mdp(self.mdp, horizon=horizon)
        self.horizon = horizon
        
        # --- THE FIX: FAKE CONTINUOUS SPACE ---
        # We pretend the action space is continuous (Box) so SAC is happy.
        # Shape (2,) -> One float for Agent 1, One float for Agent 2.
        # Range [-1, 1]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # Observation Space (Stacked)
        dummy_state = self.mdp.get_standard_start_state()
        self.obs_shape = self.base_env.lossless_state_encoding_mdp(dummy_state)[0].shape
        self.observation_space = gym.spaces.Box(
            low=0, high=np.inf, 
            shape=(self.obs_shape[0], self.obs_shape[1], self.obs_shape[2] * 2),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.base_env.seed(seed)
        self.base_env.reset()
        state = self.base_env.state
        return self._get_obs(state), {}

    def step(self, action):
        """
        action: [float_agent_1, float_agent_2] (Continuous from SAC)
        We bin them into 6 discrete actions.
        """
        # 1. Discretize the Continuous Action
        # Map range [-1, 1] into 6 buckets (0, 1, 2, 3, 4, 5)
        # We add 1.0 to shift to [0, 2], multiply by 2.99, then floor.
        joint_action_indices = []
        for a in action:
            # Simple binning logic
            # -1.0 to -0.66 -> Action 0
            # -0.66 to -0.33 -> Action 1
            # ...
            # 0.66 to 1.0   -> Action 5
            norm_a = (np.clip(a, -1, 1) + 1) / 2.0  # Now 0.0 to 1.0
            idx = int(norm_a * 6)  # Now 0 to 6
            idx = min(idx, 5)      # Clip to max 5
            joint_action_indices.append(idx)

        # 2. Convert Indices to Overcooked Directions
        joint_action_env = [Action.INDEX_TO_ACTION[i] for i in joint_action_indices]

        # 3. Step
        next_state, reward, done, info = self.base_env.step(joint_action_env)
        
        total_reward = sum(info['shaped_r_by_agent']) 
        obs = self._get_obs(next_state)
        terminated = done
        truncated = False 
        
        return obs, total_reward, terminated, truncated, info

    def _get_obs(self, state):
        obs_tuple = self.base_env.lossless_state_encoding_mdp(state)
        return np.concatenate(obs_tuple, axis=-1).astype(np.float32)

    def render(self):
        # We manually print the grid.
        
        # 1. Get the grid from the MDP
        grid = self.mdp.terrain_mtx  # List of lists representing the map
        height = len(grid)
        width = len(grid[0])
        
        # 2. Create a mutable copy to draw agents on
        display_grid = [list(row) for row in grid]
        
        # 3. Place Agents
        for i, agent in enumerate(self.base_env.state.players):
            # Agent position is (x, y) = (col, row)
            x, y = agent.position
            # Agent orientation (direction they are facing)
            orientation = agent.orientation
            
            # Draw Agent Index (1 or 2)
            # If holding an object (Soup/Onion), add a symbol
            symbol = str(i + 1)
            if agent.held_object:
                obj_name = agent.held_object.name
                if obj_name == 'onion': symbol = 'O'
                elif obj_name == 'dish': symbol = 'D'
                elif obj_name == 'soup': symbol = 'S'
            
            display_grid[y][x] = symbol

        # 4. Print to Console
        print("-" * (width + 2))
        for row in display_grid:
            print("|" + "".join(row) + "|")
        print("-" * (width + 2))
        
        # Print textual info about held objects
        for i, agent in enumerate(self.base_env.state.players):
            holding = agent.held_object.name if agent.held_object else "Nothing"
            print(f"Agent {i+1}: Holding {holding} at {agent.position}")