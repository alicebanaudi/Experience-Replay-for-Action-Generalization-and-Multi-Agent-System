import gymnasium as gym
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.actions import Action
from env.bot import SimpleBot

class OvercookedSingleAgentEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"]}

    def __init__(self, layout_name="asymmetric_advantages", horizon=400):
        super().__init__()
        
        # 1. Setup MDP
        self.mdp = OvercookedGridworld.from_layout_name(layout_name)
        
        # Set standard params (still good for Pot/Soup rewards)
        self.mdp.reward_shaping_params = {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5
        }

        # 2. Create Env
        self.base_env = OvercookedEnv.from_mdp(
            self.mdp, 
            horizon=horizon, 
            info_level=1
        )
        
        self.horizon = horizon
        self.agent_idx = 0 
        self.bot_idx = 1   
        self.bot = SimpleBot(self.bot_idx)

        # Action/Obs Spaces
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        dummy_state = self.mdp.get_standard_start_state()
        obs_shape = self.base_env.lossless_state_encoding_mdp(dummy_state)[0].shape
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=obs_shape, dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.base_env.seed(seed)
        self.base_env.reset()
        return self.base_env.lossless_state_encoding_mdp(self.base_env.state)[self.agent_idx].astype(np.float32), {}

    def step(self, action):
        # --- A. CAPTURE PREVIOUS STATE ---
        prev_player = self.base_env.state.players[self.agent_idx]
        prev_held = prev_player.held_object.name if prev_player.held_object else None
        
        # Capture position to check for Pot/Serving station interactions
        px, py = prev_player.position
        dx, dy = prev_player.orientation
        target_x, target_y = px + dx, py + dy

        # --- B. EXECUTE STEP ---
        val = (np.clip(action[0], -1, 1) + 1) / 2.0
        idx = min(int(val * 6), 5)
        agent_action = Action.INDEX_TO_ACTION[idx]

        bot_action = self.bot.get_action(self.base_env.state, self.mdp)
        joint_action = [agent_action, bot_action] if self.agent_idx == 0 else [bot_action, agent_action]

        next_state, reward, done, info = self.base_env.step(joint_action)

        # --- C. CALCULATE REWARDS ---
        shaped_rewards = list(info.get('shaped_r_by_agent', [0]*2))
        dense_reward = shaped_rewards[self.agent_idx]

        curr_player = next_state.players[self.agent_idx]
        curr_held = curr_player.held_object.name if curr_player.held_object else None
        
        # --- MANUAL REWARD INJECTION ---

        # 1. PICKUP REWARD (Onion OR Dish)
        # If we went from Nothing -> Something, it's good (unless we just swapped)
        if prev_held is None and curr_held in ['onion', 'dish']:
            dense_reward += 3.0

        # 2. POTTING REWARD (Onion -> Nothing at Pot)
        # Heuristic: Lost onion, check if facing Pot
        is_facing_pot = False
        grid = self.mdp.terrain_mtx
        if 0 <= target_y < len(grid) and 0 <= target_x < len(grid[0]):
             if grid[target_y][target_x] == 'P':
                 is_facing_pot = True

        if prev_held == 'onion' and curr_held is None and is_facing_pot:
             dense_reward += 5.0  # Good job cooking!

        # 3. SCOOPING REWARD (Dish -> Soup) [THE FIX YOU NEED]
        if prev_held == 'dish' and curr_held == 'soup':
             dense_reward += 5.0  # Good job scooping!

        # 4. ANTI-JUGGLING PENALTY
        # If we dropped an item on the floor/counter (not a pot, not serving)
        # and we didn't just deliver soup.
        is_drop = (prev_held is not None and curr_held is None)
        is_delivery = (prev_held == 'soup' and curr_held is None) # Delivery gives huge sparse reward anyway
        
        if is_drop and not is_facing_pot and not is_delivery:
            dense_reward -= 3.0 # Stop dropping things!

        # 5. UPDATE INFO
        shaped_rewards[self.agent_idx] = dense_reward
        info['shaped_r_by_agent'] = shaped_rewards

        # Combine (Multiply sparse reward to make delivery the Ultimate Goal)
        total_reward = dense_reward + (reward * 5.0) 

        # --- D. FINALIZE ---
        obs_tuple = self.base_env.lossless_state_encoding_mdp(next_state)
        next_obs = obs_tuple[self.agent_idx].astype(np.float32)
        
        truncated = False
        terminated = done
        if self.base_env.state.timestep >= self.horizon:
            truncated = True
            terminated = False 

        return next_obs, total_reward, terminated, truncated, info
    
    # Add this method to the end of OvercookedSingleAgentEnv class
    def render(self, mode='human'):
        # 1. Get the map grid
        grid = [list(row) for row in self.mdp.terrain_mtx]
        
        # 2. Place Players on the grid
        for i, player in enumerate(self.base_env.state.players):
            x, y = player.position
            
            # Determine symbol: A=Agent(0), B=Bot(1)
            # If holding something, change symbol: O=Onion, D=Dish, S=Soup
            symbol = 'A' if i == 0 else 'B'
            
            if player.held_object:
                obj = player.held_object.name
                if obj == 'onion': symbol = '🧅' if i == 0 else 'o'
                elif obj == 'dish': symbol = '🍽️' if i == 0 else 'd'
                elif obj == 'soup': symbol = '🍲' if i == 0 else 's'
            
            # Safety check bounds
            if 0 <= y < len(grid) and 0 <= x < len(grid[0]):
                grid[y][x] = symbol

        # 3. Print to Console
        print(f"\n--- Time: {self.base_env.state.timestep} ---")
        print("-" * (len(grid[0]) + 2))
        for row in grid:
            # Join row and print
            print("|" + "".join(row) + "|")
        print("-" * (len(grid[0]) + 2))