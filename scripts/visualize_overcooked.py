import sys
import os
import torch
import numpy as np
import time

# --- IMPORTS ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from agents.redq_sac_overcooked import REDQSACAgent
from env.overcooked_wrapper import OvercookedSingleAgentEnv

# ============================================================
# CONFIGURATION
# ============================================================
# FIX 1: Removed leading space in string " results/..." -> "results/..."
MODEL_PATH = "results/overcooked_vectorized_baseline_final.pth" 
LAYOUT = "asymmetric_advantages"

def main():
    device = "cpu"
    print(f"🍿 Starting Visualization for layout: {LAYOUT}")
    
    # 1. Init Environment
    env = OvercookedSingleAgentEnv(layout_name=LAYOUT)
    
    # 2. Get Correct Dimensions
    # Note: 520 is correct for Single Agent. 
    # (5 width * 4 height * 26 features) = 520. 
    # The "1040" expectation was for 2 agents, but here we only see 1 agent's view.
    obs_dim = int(np.prod(env.observation_space.shape)) 
    act_dim = 1 # Box(1,)

    print(f"   Obs Dim: {obs_dim}")
    print(f"   Act Dim: {act_dim}")

    # 3. Init Agent & Load Weights
    agent = REDQSACAgent(obs_dim, act_dim, device=device)
    
    if os.path.exists(MODEL_PATH):
        print(f"   Loading model from {MODEL_PATH}...")
        try:
            # Load weights
            agent.actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("✅ Model loaded successfully!")
        except RuntimeError as e:
            print(f"❌ Model Load Error: {e}")
            return
    else:
        print(f"⚠️ Warning: File '{MODEL_PATH}' not found. Bot will play against Random Agent.")

    # 4. Play Loop
    obs, _ = env.reset()
    obs = obs.flatten().astype(np.float32) 
    
    done = False
    total_reward = 0
    step = 0

    while not done:
        # Select Deterministic Action
        action = agent.select_action(obs, deterministic=True)
        
        # Step
        # Ensure action is passed as a list/array if needed by the wrapper logic
        # Our wrapper expects shape (1,)
        if np.isscalar(action): action = [action]

        next_obs, reward, terminated, truncated, info = env.step(action)
        
        obs = next_obs.flatten().astype(np.float32)
        done = terminated or truncated
        total_reward += reward
        step += 1

        # RENDER
        # This will now call your NEW manual render method
        env.render()
        
        print(f"Step: {step} | Reward: {reward:.2f}")
        time.sleep(0.2) # Speed up slightly (0.5 is slow)

    print(f"🏁 Game Over! Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()