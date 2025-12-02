import sys
import os
import torch
import numpy as np
import time

# --- IMPORTS ---
# Ensure Python can find your folders
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from agents.redq_sac import REDQSACAgent
from env.overcooked_wrapper import OvercookedMAEnv

# ============================================================
# CONFIGURATION
# ============================================================
# Make sure this matches the filename you see in your 'results' folder
MODEL_PATH = " results/overcooked_BASELINE_final.pth"  # <--- CHECK FILENAME
LAYOUT = "cramped_room"

def main():
    device = "cpu" # Visualization is fast enough on CPU
    print(f"🍿 Starting Visualization for layout: {LAYOUT}")
    
    # 1. Init Environment
    env = OvercookedMAEnv(layout_name=LAYOUT)
    
    # --- FIX 1: Calculate Correct Input Size (1040) ---
    # We must flatten the 3D grid (5x4x52) into a 1D vector (1040)
    obs_dim = int(np.prod(env.observation_space.shape)) 
    act_dim = env.action_space.shape[0]

    print(f"   Obs Dim: {obs_dim} (Expected: 1040)")
    print(f"   Act Dim: {act_dim}")

    # 2. Init Agent & Load Weights
    agent = REDQSACAgent(obs_dim, act_dim, device=device)
    
    if os.path.exists(MODEL_PATH):
        print(f"   Loading model from {MODEL_PATH}...")
        try:
            agent.actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("✅ Model loaded successfully!")
        except RuntimeError as e:
            print(f"❌ Model Load Error: {e}")
            print("💡 Tip: Did you forget to flatten the observation in the training script?")
            return
    else:
        print(f"⚠️ Warning: File {MODEL_PATH} not found. Running with random weights.")

    # 3. Play Loop
    # --- FIX 2: Flatten Initial State ---
    obs, _ = env.reset()
    obs = obs.flatten().astype(np.float32) 
    
    done = False
    total_reward = 0
    step = 0

    while not done:
        # Select Deterministic Action (No random noise)
        action = agent.select_action(obs, deterministic=True)
        
        # Step
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # --- FIX 3: Flatten Next State ---
        obs = next_obs.flatten().astype(np.float32)
        
        done = terminated or truncated
        total_reward += reward
        step += 1

        # RENDER: Print the kitchen to the console
        print(f"\n--- Step {step} | Reward: {reward:.2f} ---")
        env.render() 
        
        # Slow down so you can watch
        time.sleep(0.5) 

    print(f"🏁 Game Over! Total Reward: {total_reward}")
    env.close()

if __name__ == "__main__":
    main()