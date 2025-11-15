# training/train_sumo.py

import os
import numpy as np
import torch
from tqdm import trange

from env.sumo_env import SumoContinuousEnv
from replay.buffer import ReplayBuffer
from agents.redq_sac import REDQSACAgent
import os
from sumo_rl import __file__ as sumo_rl_path


def make_env():
    SUMO_RL_DIR = os.path.dirname(sumo_rl_path)
    NETS_DIR = os.path.join(SUMO_RL_DIR, "nets")

    EXPERIMENT = "single-intersection"
    # TODO: point to your own SUMO net/route files
    net_file = os.path.join(NETS_DIR, EXPERIMENT, f"{EXPERIMENT}.net.xml")
    route_file = os.path.join(NETS_DIR, EXPERIMENT, f"{EXPERIMENT}.rou.xml")
    
    env = SumoContinuousEnv(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,
        min_green=5.0,
        max_green=60.0,
    )
    return env


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = make_env()
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    buffer_capacity = int(1e6)
    replay_buffer = ReplayBuffer(buffer_capacity, obs_dim, act_dim)

    agent = REDQSACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        num_q_nets=10,
        num_q_samples=2,
        gamma=0.99,
        tau=0.005,
        lr=3e-4,
        utd_ratio=20,      # as in SYNTHER online experiments
        batch_size=256,
    )

    total_steps = 100_000
    start_steps = 10_000  # fill buffer with random actions before training
    eval_interval = 5_000

    obs, _ = env.reset()
    episode_return = 0.0
    episode_len = 0

    for t in trange(total_steps):
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

        if done:
            print(f"Step {t}: episode return={episode_return:.2f}, len={episode_len}")
            obs, _ = env.reset()
            episode_return = 0.0
            episode_len = 0

        # Update agent after we have enough data
        if t >= start_steps and len(replay_buffer) >= agent.batch_size:
            logs = agent.update(replay_buffer)

        # Simple evaluation hook (you can expand later)
        if (t + 1) % eval_interval == 0:
            eval_return = evaluate(env, agent, episodes=3)
            print(f"[Eval @ step {t+1}] avg return={eval_return:.2f}")

    env.close()


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
