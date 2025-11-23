import os
import numpy as np
import torch
from env.sumo_env import SumoContinuousEnv
from agents.redq_sac import REDQSACAgent

# ============================================================
# Load the final trained agent
# ============================================================

MODEL_PATH = "agent_final.pth"     # or whatever name you used

device = "cuda" if torch.cuda.is_available() else "cpu"

# These must match your training settings
obs_dim = 50      # CHANGE THIS if your actual observation dimension differs
act_dim = 1

agent = REDQSACAgent(
    obs_dim=obs_dim,
    act_dim=act_dim,
    device=device,
    num_q_nets=5,
    num_q_samples=2,
    use_pgr=False,      # PGR not needed for inference
)

agent.actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
agent.actor.eval()

# ============================================================
# Correct SUMO path for your environment
# ============================================================

NET_DIR = "/usr/local/lib/python3.12/dist-packages/sumo_rl/nets/single-intersection"

net_file = os.path.join(NET_DIR, "single-intersection.net.xml")
route_file = os.path.join(NET_DIR, "single-intersection.rou.xml")

# ============================================================
# Create SUMO env for visualization
# ============================================================

env = SumoContinuousEnv(
    net_file=net_file,
    route_file=route_file,
    use_gui=True,          # <<<< SHOW GUI
    min_green=5.0,
    max_green=60.0,
)

# ============================================================
# Run one episode using the learned policy
# ============================================================

obs, _ = env.reset()
done = False
step = 0
total_reward = 0

print("\nRunning visualization episode...\n")

while not done:
    action = agent.select_action(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    total_reward += reward
    step += 1

env.close()

print(f"\nEpisode finished: steps={step}, total reward={total_reward:.2f}")
