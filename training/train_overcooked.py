import os
import sys
import numpy as np
import torch
from tqdm import trange

# --- PATH SETUP ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# --- IMPORTS ---
from env.overcooked_wrapper import OvercookedMAEnv 
from agents.redq_sac import REDQSACAgent
from replay.buffer import ReplayBuffer
from models.diffusion import DiffusionModel
from models.diffusion_trainer import DiffusionTrainer
from models.synthetic_generator import SyntheticGenerator

# ============================================================
# 🎛️ EXPERIMENT CONTROL CENTER
# ============================================================
# CHOOSE YOUR MODE HERE: "baseline" | "synther" | "pgr"
MODE = "baseline" 

CONFIG = {
    "layout": "cramped_room",
    "total_steps": 300_000,      
    "start_steps": 10_000,       
    "batch_size": 256,
    
    # Mode-Specific Defaults (Will be overwritten by setup logic below)
    "exp_name": f"overcooked_{MODE}",
    "utd_ratio": 10,            # High UTD for all (REDQ standard)
    
    # Synther / PGR Settings
    "use_synther": False,       # Default OFF
    "use_pgr": False,           # Default OFF
    "pgr_coef": 0.0,
    
    # Diffusion Settings
    "diffusion_freq": 30_000,   # Retrain slightly more often (every 15k)
    "diffusion_steps": 5_000,   # Good depth
    "generate_count": 40_000,   # Generate 25k samples per cycle
    "synthetic_ratio": 0.5,
}

# --- AUTOMATIC CONFIG SETUP ---
if MODE == "baseline":
    CONFIG["use_synther"] = False
    CONFIG["use_pgr"] = False
    CONFIG["exp_name"] = "overcooked_BASELINE"
    # Baseline usually runs better with lower UTD if no synthetic data is present
    # But for fair comparison, we often keep UTD high or lower it to 1.
    # Let's keep it high (REDQ style) to see if it overfits without data.

elif MODE == "synther":
    CONFIG["use_synther"] = True
    CONFIG["use_pgr"] = False
    CONFIG["exp_name"] = "overcooked_SYNTHER"

elif MODE == "pgr":
    CONFIG["use_synther"] = True
    CONFIG["use_pgr"] = True
    CONFIG["pgr_coef"] = 1e-5  # Standard penalty weight
    CONFIG["exp_name"] = "overcooked_PGR"

# ============================================================
# Main Loop
# ============================================================
def make_env():
    return OvercookedMAEnv(layout_name=CONFIG["layout"])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🧪 STARTING EXPERIMENT: {MODE.upper()}")
    print(f"   Name: {CONFIG['exp_name']}")
    print(f"   Synther: {CONFIG['use_synther']} | PGR: {CONFIG['use_pgr']}")
    
    if not os.path.exists("results"):
        os.makedirs("results")

    # 1. Init Env
    env = make_env()
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.shape[0]

    # 2. Init Buffer
    buffer = ReplayBuffer(1_000_000, obs_dim, act_dim, device=device)

    # 3. Init Agent (With PGR wiring)
    agent = REDQSACAgent(
        obs_dim=obs_dim, act_dim=act_dim, device=device,
        batch_size=CONFIG["batch_size"], 
        utd_ratio=CONFIG["utd_ratio"],
        # PGR Params
        use_pgr=CONFIG["use_pgr"],
        pgr_coef=CONFIG["pgr_coef"]
    )

    # 4. Init Diffusion (Only if Synther is ON)
    if CONFIG["use_synther"]:
        transition_dim = obs_dim + act_dim + 1 + obs_dim + 1
        diff_model = DiffusionModel(dim=transition_dim, hidden=256).to(device)
        diff_trainer = DiffusionTrainer(diff_model, lr=3e-4, device=device)
        syn_generator = SyntheticGenerator(diff_model, device=device)
    else:
        diff_trainer = None
        syn_generator = None

    # 5. Loop
    obs, _ = env.reset()
    obs = obs.flatten().astype(np.float32)
    episode_return = 0    
    pbar = trange(CONFIG["total_steps"], desc=f"{MODE.upper()} Training")
    
    for t in pbar:
        # Checkpoint
        if t % 25_000 == 0 and t > 0:
            torch.save(agent.actor.state_dict(), f"results/{CONFIG['exp_name']}_step_{t}.pth")

        # A. Collect
        if t < CONFIG["start_steps"]:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = next_obs.flatten().astype(np.float32)
        done = terminated or truncated

        buffer.add(obs, action, reward, next_obs, done)
        obs = next_obs
        episode_return += reward

        if done:
            pbar.set_description(f"Step {t} | Ret: {episode_return:.2f}")
            obs, _ = env.reset()
            obs = obs.flatten().astype(np.float32)
            episode_return = 0

        # B. Synther Injection
        if CONFIG["use_synther"] and t > CONFIG["start_steps"] and t % CONFIG["diffusion_freq"] == 0:
            # Train Diffusion
            for _ in range(CONFIG["diffusion_steps"]):
                diff_trainer.train_step(buffer, batch_size=256)
            
            # Generate
            fake_data = syn_generator.sample(CONFIG["generate_count"])
            buffer.add_synthetic(fake_data)

        # C. Agent Update
        if t >= CONFIG["start_steps"]:
            ratio = CONFIG["synthetic_ratio"] if CONFIG["use_synther"] else 0.0
            agent.update(buffer, synthetic_ratio=ratio)

    # Final Save
    torch.save(agent.actor.state_dict(), f"results/{CONFIG['exp_name']}_final.pth")
    env.close()
    print(f"🏁 {MODE.upper()} FINISHED!")

if __name__ == "__main__":
    main()