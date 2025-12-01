# training/train_overcooked.py

import os
import sys
import numpy as np
import torch
from tqdm import trange
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# --- IMPORTS ---
# 1. The Environment (Swapped from SUMO to Overcooked)
# Note: Using 'env' because that is where your wrapper is located
from env.overcooked_wrapper import OvercookedMAEnv 

# 2. The Agent & Replay Buffer
from agents.redq_sac import REDQSACAgent
from replay.buffer import ReplayBuffer

# 3. The Generative Models (SYNTHER)
from models.diffusion import DiffusionModel
from models.diffusion_trainer import DiffusionTrainer
from models.synthetic_generator import SyntheticGenerator

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "exp_name": "overcooked_synther_v1",
    "layout": "cramped_room",   # Options: cramped_room, asymmetric_advantages
    
    "total_steps": 50_000,      # Overcooked learns faster than SUMO usually
    "start_steps": 2_000,       # Collect real gameplay first
    
    # Synther Settings
    "use_synther": True,        # Set to False to run a Baseline
    "diffusion_freq": 5_000,    # Train diffusion every 5k steps
    "diffusion_steps": 500,     # Gradient steps for diffusion
    "generate_count": 5_000,    # How many fake samples to generate
    "synthetic_ratio": 0.5,     # 50% Real / 50% Fake mixing
}

# ============================================================
# Environment Helper
# ============================================================
def make_env():
    # Initializes your custom wrapper that converts Continuous Actions -> Discrete
    return OvercookedMAEnv(layout_name=CONFIG["layout"])

# ============================================================
# Main Training Loop
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🍳 Starting OVERCOOKED Training on {device}")
    
    # --- FIX 1: Create results folder so it doesn't crash at the end ---
    if not os.path.exists("results"):
        os.makedirs("results")
        print("📁 Created 'results' directory.")

    # 1. Init Environment
    env = make_env()
    obs_dim = int(np.prod(env.observation_space.shape)) # 1040
    act_dim = env.action_space.shape[0]

    # 2. Init Dual Buffer
    buffer = ReplayBuffer(500_000, obs_dim, act_dim, device=device)

    # 3. Init Agent
    agent = REDQSACAgent(obs_dim, act_dim, device=device, batch_size=256, utd_ratio=10)

    # 4. Init Generative Models
    transition_dim = obs_dim + act_dim + 1 + obs_dim + 1
    diff_model = DiffusionModel(dim=transition_dim, hidden=256).to(device)
    diff_trainer = DiffusionTrainer(diff_model, lr=3e-4, device=device)
    syn_generator = SyntheticGenerator(diff_model, device=device)

    # 5. Training Loop
    obs, _ = env.reset()
    obs = obs.flatten().astype(np.float32) # Correct
    
    episode_return = 0    
    pbar = trange(CONFIG["total_steps"])
    
    for t in pbar:
        # Checkpointing
        if t % 5000 == 0 and t > 0:
            torch.save(agent.actor.state_dict(), f"results/{CONFIG['exp_name']}_step_{t}.pth")

        # A. Data Collection
        if t < CONFIG["start_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        # --- FIX 2: FLATTEN NEXT_OBS HERE ---
        # The environment returns (5,4,52), we need (1040,)
        next_obs = next_obs.flatten().astype(np.float32) 
        # ------------------------------------
        
        done = terminated or truncated

        # Now shapes match: (1040,) -> (1040,)
        buffer.add(obs, action, reward, next_obs, done)
        
        obs = next_obs
        episode_return += reward

        if done:
            pbar.set_description(f"Step {t} | Return: {episode_return:.2f}")
            obs, _ = env.reset()
            obs = obs.flatten().astype(np.float32) # Correct
            episode_return = 0

        # B. Synther Logic
        if CONFIG["use_synther"] and t > CONFIG["start_steps"] and t % CONFIG["diffusion_freq"] == 0:
            print(f"\n🎨 [Step {t}] Training Diffusion...")
            for _ in range(CONFIG["diffusion_steps"]):
                diff_trainer.train_step(buffer, batch_size=256)
            
            fake_data = syn_generator.sample(CONFIG["generate_count"])
            buffer.add_synthetic(fake_data)
            print("   Injection Complete.")

        # C. Agent Training
        if t >= CONFIG["start_steps"]:
            ratio = CONFIG["synthetic_ratio"] if CONFIG["use_synther"] else 0.0
            agent.update(buffer, synthetic_ratio=ratio)

    torch.save(agent.actor.state_dict(), f"results/{CONFIG['exp_name']}_actor.pth")
    env.close()
    print("🏁 Cooking Training Complete!")

if __name__ == "__main__":
    main()