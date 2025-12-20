import os
import sys
import numpy as np
import torch
import gymnasium as gym
from tqdm import trange

# --- PATH SETUP ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# --- IMPORTS ---
from env.overcooked_wrapper import OvercookedSingleAgentEnv 
from agents.redq_sac_overcooked import REDQSACAgent
from replay.buffer import ReplayBuffer
# Assuming you have these files for Synther:
from models.diffusion import DiffusionModel
from models.diffusion_trainer import DiffusionTrainer
from models.synthetic_generator import SyntheticGenerator

class Silence:
    """Context manager to suppress stdout/stderr"""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
# ============================================================
# 🎛️ EXPERIMENT CONTROL CENTER
# ============================================================
# OPTIONS: "baseline" | "synther" | "pgr"
MODE = "baseline" 

CONFIG = {
    # --- Environment ---
    "layout": "asymmetric_advantages", 
    "total_steps": 500_000,     
    "start_steps": 10_000,        # Collect real data first
    "num_envs": 20,               # A100 Speed Boost
    
    # --- REDQ / Training ---
    "batch_size": 2048,           
    "utd_ratio": 20,              
    "exp_name": f"overcooked_vectorized_{MODE}",
    
    # --- Defaults (Modified by Mode below) ---
    "use_synther": False,
    "use_pgr": False,
    "pgr_coef": 0.0,
    "synthetic_ratio": 0.5,       # Half real, half fake in batch
    
    # --- Diffusion Settings ---
    "diffusion_freq": 50_000,     # Train diffusion every 50k steps
    "diffusion_train_steps": 5_000, 
    "generate_count": 50_000,     # How many synthetic samples to create
}

# --- AUTOMATIC MODE SETUP ---
if MODE == "baseline":
    CONFIG["use_synther"] = False
    CONFIG["use_pgr"] = False

elif MODE == "synther":
    CONFIG["use_synther"] = True
    CONFIG["use_pgr"] = False

elif MODE == "pgr":
    CONFIG["use_synther"] = True
    CONFIG["use_pgr"] = True
    CONFIG["pgr_coef"] = 1e-4  # Penalty weight for PGR

# ============================================================
# MAIN LOOP
# ============================================================

def make_env(layout):
    def _init():
        with Silence():
            return OvercookedSingleAgentEnv(layout_name=layout)
    return _init

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🧪 STARTING EXPERIMENT: {MODE.upper()}")
    print(f"   Envs: {CONFIG['num_envs']} | Device: {device}")
    print(f"   Synther: {CONFIG['use_synther']} | PGR: {CONFIG['use_pgr']}")

    if not os.path.exists("results"):
        os.makedirs("results")

    # 1. Vectorized Envs
    envs = gym.vector.AsyncVectorEnv(
        [make_env(CONFIG["layout"]) for _ in range(CONFIG["num_envs"])]
    )
    
    # Get Shapes
    obs_dim = np.prod(envs.single_observation_space.shape)
    act_dim = 1 

    # 2. Buffer
    buffer = ReplayBuffer(1_000_000, obs_dim, act_dim, device=device)

    # 3. Agent (Initialized with PGR params)
    agent = REDQSACAgent(
        obs_dim=obs_dim, act_dim=act_dim, device=device,
        batch_size=CONFIG["batch_size"], 
        utd_ratio=CONFIG["utd_ratio"],
        use_pgr=CONFIG["use_pgr"],
        pgr_coef=CONFIG["pgr_coef"]
    )

    # 4. Diffusion (Only for Synther/PGR modes)
    if CONFIG["use_synther"]:
        # Transition = (Obs + Act + Rew + NextObs + Done)
        # Dimensions: Obs(N) + Act(M) + Rew(1) + NextObs(N) + Done(1)
        transition_dim = obs_dim + act_dim + 1 + obs_dim + 1
        
        diff_model = DiffusionModel(dim=transition_dim, hidden=512).to(device)
        diff_trainer = DiffusionTrainer(diff_model, lr=3e-4, device=device)
        syn_generator = SyntheticGenerator(diff_model, device=device)
    else:
        diff_trainer = None
        syn_generator = None

    # 5. Training Loop
    obs, _ = envs.reset()
    obs = obs.reshape(CONFIG["num_envs"], -1) # Flatten

    total_timesteps = 0
    # Steps are divided by num_envs because each loop ticks all 20 envs
    pbar = trange(int(CONFIG["total_steps"] // CONFIG["num_envs"]))
    
    for step in pbar:
        total_timesteps += CONFIG["num_envs"]

        # --- A. Collect Data (Real) ---
        if total_timesteps < CONFIG["start_steps"]:
            actions = np.random.uniform(-1, 1, size=(CONFIG["num_envs"], 1))
        else:
            # Agent handles batch prediction
            actions = agent.select_action(obs, deterministic=False)
            if actions.ndim == 1: actions = actions.reshape(-1, 1)

        next_obs, rewards, terminated, truncated, _ = envs.step(actions)
        next_obs_flat = next_obs.reshape(CONFIG["num_envs"], -1)
        dones = np.logical_or(terminated, truncated)

        # Vectorized Add to Buffer
        for i in range(CONFIG["num_envs"]):
            buffer.add(obs[i], actions[i], rewards[i], next_obs_flat[i], dones[i])
        
        obs = next_obs_flat

        # --- B. Synther Logic (Periodic) ---
        if CONFIG["use_synther"] and total_timesteps > CONFIG["start_steps"]:
            if step % (CONFIG["diffusion_freq"] // CONFIG["num_envs"]) == 0:
                pbar.write(f"⚡ Training Diffusion Model (Step {total_timesteps})...")
                
                # Train Diffusion
                for _ in range(CONFIG["diffusion_train_steps"]):
                    diff_trainer.train_step(buffer, batch_size=2048) # Large batch for A100
                
                # Generate Synthetic Data
                pbar.write(f"   Generating {CONFIG['generate_count']} synthetic samples...")
                fake_data = syn_generator.sample(CONFIG['generate_count'])
                buffer.add_synthetic(fake_data)

        # --- C. Update Agent ---
        if total_timesteps >= CONFIG["start_steps"]:
            ratio = CONFIG["synthetic_ratio"] if CONFIG["use_synther"] else 0.0
            agent.update(buffer, synthetic_ratio=ratio)

        # Logging
        if step % 100 == 0:
            pbar.set_description(f"Step {total_timesteps} | Rew: {np.mean(rewards):.3f}")
            # Checkpoint
            if step % 5000 == 0:
                torch.save(agent.actor.state_dict(), f"results/{CONFIG['exp_name']}_ckpt.pth")

    # Final Save
    torch.save(agent.actor.state_dict(), f"results/{CONFIG['exp_name']}_final.pth")
    envs.close()
    print("🏁 FINISHED!")

if __name__ == "__main__":
    main()