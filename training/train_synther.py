# training/train_synther.py

import os
import numpy as np
import torch
from tqdm import trange

# Reuse your existing environment and agent setup
from env.sumo_env import SumoContinuousEnv
from agents.redq_sac import REDQSACAgent
from sumo_rl import __file__ as sumo_rl_path

# Import the NEW components
# Ensure 'buffer.py' contains the 'GenerativeReplayBuffer' class I gave you
from replay.buffer import ReplayBuffer
from models.diffusion import DiffusionModel
from models.diffusion_trainer import DiffusionTrainer
from models.synthetic_generator import SyntheticGenerator

# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    "exp_name": "debug_run_v1",
    "total_steps": 1000,        # Stop quickly
    "start_steps": 200,         # Start training RL almost immediately
    
    # Synther Settings - Trigger them FAST
    "diffusion_freq": 300,      # Trigger diffusion training at step 300
    "diffusion_steps": 5,       # Only 5 steps (just to check if it runs)
    "generate_count": 50,       # Generate tiny amount of fake data
    "synthetic_ratio": 0.5,     # Test the mixing logic
    
    # Batch sizes (Small to prevent memory errors during debug)
    "synther_batch": 10,
}

# ============================================================
# Environment Helper (Same as before)
# ============================================================
def make_env():
    SUMO_RL_DIR = os.path.dirname(sumo_rl_path)
    NETS_DIR = os.path.join(SUMO_RL_DIR, "nets")
    EXPERIMENT = "single-intersection"
    net_file = os.path.join(NETS_DIR, EXPERIMENT, f"{EXPERIMENT}.net.xml")
    route_file = os.path.join(NETS_DIR, EXPERIMENT, f"{EXPERIMENT}.rou.xml")
    
    return SumoContinuousEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        min_green=5.0,
        max_green=60.0,
    )

# ============================================================
# Main Experimental Loop
# ============================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Starting SYNTHER Training on {device}")

    # 1. Init Environment
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # 2. Init Dual Buffer (The new class)
    # Make sure replay/buffer.py has the GenerativeReplayBuffer class!
    buffer = ReplayBuffer(  
        capacity=1_000_000, 
        obs_dim=obs_dim, 
        act_dim=act_dim, 
        device=device
    )

    # 3. Init Agent
    agent = REDQSACAgent(
        obs_dim=obs_dim, act_dim=act_dim, device=device,
        batch_size=256, utd_ratio=20
    )

    # 4. Init Generative Models
    # Input dim = Obs + Act + Reward + NextObs + Done
    transition_dim = obs_dim + act_dim + 1 + obs_dim + 1
    
    diff_model = DiffusionModel(dim=transition_dim, hidden=256).to(device)
    diff_trainer = DiffusionTrainer(diff_model, lr=3e-4, device=device)
    syn_generator = SyntheticGenerator(diff_model, device=device)

    # 5. Training Loop
    obs, _ = env.reset()
    
    for t in trange(CONFIG["total_steps"]):
        
        # --- A. Data Collection (Real World) ---
        if t < CONFIG["start_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Add to REAL buffer
        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs

        if done:
            obs, _ = env.reset()

        # --- B. Synther Logic (Periodic Injection) ---
        if t > CONFIG["start_steps"] and t % CONFIG["diffusion_freq"] == 0:
            print(f"\n🎨 [Step {t}] Training Diffusion Model on REAL data...")
            
            # 1. Train Diffusion (Buffer defaults to Real Data only)
            diff_loss = 0
            for _ in range(CONFIG["diffusion_steps"]):
                loss = diff_trainer.train_step(buffer, batch_size=256)
                diff_loss += loss
            
            print(f"   Avg Diffusion Loss: {diff_loss / CONFIG['diffusion_steps']:.4f}")

            # 2. Generate Synthetic Data
            print(f"   Generating {CONFIG['generate_count']} synthetic samples...")
            fake_data = syn_generator.sample(CONFIG["generate_count"])
            
            # 3. Add to Synthetic Buffer
            buffer.add_synthetic(fake_data)
            print("   Injection Complete.")

        # --- C. Agent Training (Every Step) ---
        if t >= CONFIG["start_steps"]:
            # ASK FOR MIXED DATA
            agent.update(buffer, synthetic_ratio=CONFIG["synthetic_ratio"])

    # Save
    torch.save(agent.actor.state_dict(), f"results/{CONFIG['exp_name']}.pth")
    env.close()

if __name__ == "__main__":
    main()