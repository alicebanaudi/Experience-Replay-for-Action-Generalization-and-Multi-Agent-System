# training/train_sumo.py

import os
import numpy as np
import torch
from tqdm import trange

from env.sumo_env import SumoContinuousEnv
from replay.buffer import ReplayBuffer
from agents.redq_sac import REDQSACAgent
from sumo_rl import __file__ as sumo_rl_path

from models.diffusion import DiffusionModel
from models.diffusion_trainer import DiffusionTrainer
from models.synthetic_generator import SyntheticGenerator


# ============================================================
# CONFIG — choose which version of the algorithm to run
# ============================================================

CONFIG = {
    "mode": "synther_pgr",       # baseline | synther | synther_pgr

    # SYNTHER parameters
    "synther_batch": 20,        # only once at step 5000
    "diffusion_interval": 20_000,  # unused for now (disabled)

    # PGR parameters
    "pgr_coef": 1.0,
}


# ============================================================
# Create SUMO environment
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
        use_gui=True,
        min_green=5.0,
        max_green=60.0,
    )


# ============================================================
# Main Training Loop
# ============================================================

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mode = CONFIG["mode"]
    use_synther = (mode in ["synther", "synther_pgr"])
    use_pgr = (mode == "synther_pgr")

    print(f"\n=== TRAINING MODE: {mode} ===")
    print(f"Synther enabled? {use_synther}")
    print(f"PGR enabled?     {use_pgr}\n")

    # -----------------------
    # Environment + buffers
    # -----------------------
    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    replay_buffer = ReplayBuffer(int(1e6), obs_dim, act_dim)

    # -----------------------
    # Agent (lighter version)
    # -----------------------
    agent = REDQSACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        num_q_nets=3,        # reduced
        num_q_samples=2,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        utd_ratio=2,         # reduced
        batch_size=256,
        use_pgr=use_pgr,
        pgr_coef=CONFIG["pgr_coef"] if use_pgr else 0.0,
    )

    # -----------------------
    # SYNTHER SETUP
    # -----------------------
    if use_synther:
        transition_dim = obs_dim + act_dim + obs_dim + 2  # s+a+s'+r+done
        diffusion_model = DiffusionModel(dim=transition_dim, timesteps=5)
        diffusion_trainer = DiffusionTrainer(diffusion_model, device=device)
        synthetic_gen = SyntheticGenerator(diffusion_model, device=device)
    else:
        diffusion_model = diffusion_trainer = synthetic_gen = None

    # -----------------------
    # Training loop
    # -----------------------
    total_steps = 10_000
    start_steps = 8_000
    eval_interval = 5_000

    obs, _ = env.reset()
    episode_return = 0.0
    episode_len = 0

    for t in trange(total_steps):
        # ---------------------------------------------------------
        # Action selection
        # ---------------------------------------------------------
        if t < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        replay_buffer.add(obs, action, reward, next_obs, done)

        obs = next_obs
        episode_return += reward
        episode_len += 1

        # ---------------------------------------------------------
        # Episode ended
        # ---------------------------------------------------------
        if done:
            print(f"Step {t}: episode return={episode_return:.2f}, len={episode_len}")
            obs, _ = env.reset()
            episode_return = 0.0
            episode_len = 0

        # ---------------------------------------------------------
        # Agent update
        # ---------------------------------------------------------
        if t >= start_steps and len(replay_buffer) >= agent.batch_size:
            logs = agent.update(replay_buffer)

        # ---------------------------------------------------------
        # === SINGLE SYNTHER INJECTION AT STEP 5000 ===
        # ---------------------------------------------------------
        if use_synther and t == start_steps:
            synthetic = synthetic_gen.sample(CONFIG["synther_batch"])
            replay_buffer.add_synthetic(synthetic)
            print(f"[Synther] Added {len(synthetic)} synthetic transitions")

        # ---------------------------------------------------------
        # Evaluation 
        # ---------------------------------------------------------
        if (t + 1) % eval_interval == 0:
            eval_return = evaluate(env, agent, episodes=1)
            print(f"[Eval @ step {t+1}] avg return={eval_return:.2f}")


    torch.save(agent.actor.state_dict(), "actor_final.pth")
    env.close()


# ============================================================
# Evaluation
# ============================================================

def evaluate(env, agent, episodes=5):
    returns = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_ret = 0.0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            ep_ret += reward
        returns.append(ep_ret)
    return np.mean(returns)


if __name__ == "__main__":
    main()
