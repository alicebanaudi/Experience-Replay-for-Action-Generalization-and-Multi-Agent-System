import os
import numpy as np
import torch
import time

from env.sumo_env import SumoContinuousEnv
from agents.redq_sac import REDQSACAgent
from sumo_rl import __file__ as sumo_rl_path

MODEL_PATH = "actor_final.pth"   # same name as in train_sumo.py

def make_env(use_gui: bool):
    SUMO_RL_DIR = os.path.dirname(sumo_rl_path)
    NETS_DIR = os.path.join(SUMO_RL_DIR, "nets")
    EXPERIMENT = "single-intersection"

    net_file = os.path.join(NETS_DIR, EXPERIMENT, f"{EXPERIMENT}.net.xml")
    route_file = os.path.join(NETS_DIR, EXPERIMENT, f"{EXPERIMENT}.rou.xml")

    env = SumoContinuousEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=use_gui,
        min_green=5.0,
        max_green=60.0,
    )
    return env


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ----------------------------------------------------
    # 1) Build a dummy env to infer obs_dim and act_dim
    # ----------------------------------------------------
    tmp_env = make_env(use_gui=False)
    obs_dim = tmp_env.observation_space.shape[0]
    act_dim = tmp_env.action_space.shape[0]
    tmp_env.close()

    print(f"Detected obs_dim={obs_dim}, act_dim={act_dim}")

    # ----------------------------------------------------
    # 2) Rebuild agent with SAME dims as training
    # ----------------------------------------------------
    agent = REDQSACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        # the following don't matter for loading the actor,
        # but we set small ones for cleanliness
        num_q_nets=3,
        num_q_samples=2,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        utd_ratio=1,
        batch_size=256,
        use_pgr=False,        # PGR not needed for evaluation
        pgr_coef=0.0,
    )

    # Load actor weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    agent.actor.load_state_dict(state_dict)
    agent.actor.eval()
    print("Loaded actor weights from", MODEL_PATH)

    # ----------------------------------------------------
    # 3) Create GUI environment for visualization
    # ----------------------------------------------------
    env = make_env(use_gui=True)
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    step = 0

    print("\nRunning visualization episode with GUI...\n")

    while not done:
        # Deterministic (mean) action from policy
        action = agent.select_action(obs, deterministic=True)

        # Gymnasium step: obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        step += 1
        time.sleep(1)

    env.close()
    print(f"\n[VIS] Episode finished: steps={step}, total reward={total_reward:.2f}")


if __name__ == "__main__":
    main()
