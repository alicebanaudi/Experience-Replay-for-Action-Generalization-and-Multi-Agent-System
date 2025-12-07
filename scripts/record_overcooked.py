import sys
import os
import torch
import numpy as np
import pygame
import imageio  # For saving the GIF

# Set "Dummy" video driver so pygame doesn't crash on a server without a screen
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Path setup
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from agents.redq_sac import REDQSACAgent
from env.overcooked_wrapper import OvercookedMAEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer

# --- CONFIG ---
MODEL_PATH = "results/overcooked_synther_v1_actor.pth"
OUTPUT_FILE = "cooking_agent.gif"
LAYOUT = "cramped_room"
MAX_STEPS = 400

def main():
    print(f"📹 Starting Video Recording for: {LAYOUT}")
    
    # 1. Init Environment
    env = OvercookedMAEnv(layout_name=LAYOUT)
    obs_dim = int(np.prod(env.observation_space.shape))
    act_dim = env.action_space.shape[0]

    # 2. Init Visualizer (The tool that draws the graphics)
    visualizer = StateVisualizer()    
    # 3. Load Agent
    device = "cpu"
    agent = REDQSACAgent(obs_dim, act_dim, device=device)
    
    if os.path.exists(MODEL_PATH):
        print(f"   Loading weights from {MODEL_PATH}...")
        agent.actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("⚠️ Model not found. Recording random actions.")

    # 4. Recording Loop
    obs, _ = env.reset()
    obs = obs.flatten().astype(np.float32)
    
    frames = []  # We will store images here
    done = False
    step = 0
    
    print("   Recording frames...")
    
    while not done and step < MAX_STEPS:
        # A. Render the current state to a Pygame Surface
        # internal_state is the true Overcooked state object
        state_obj = env.base_env.state 
        
        # Draw it!
        surface = visualizer.render_state(state_obj, grid=env.mdp.terrain_mtx)
        
        # Convert Pygame Surface -> Numpy Array (RGB)
        img_data = pygame.surfarray.array3d(surface)
        
        # Rotate and Flip (Pygame uses transposed coordinates)
        img_data = img_data.swapaxes(0, 1)
        
        frames.append(img_data)

        # B. Agent Act
        action = agent.select_action(obs, deterministic=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        
        obs = next_obs.flatten().astype(np.float32)
        done = terminated or truncated
        step += 1
        
        if step % 50 == 0:
            print(f"   Captured {step} frames...")

    env.close()

    # 5. Save to GIF
    print(f"💾 Saving GIF to {OUTPUT_FILE}...")
    # duration=0.1 means 10 frames per second
    imageio.mimsave(OUTPUT_FILE, frames, duration=0.1, loop=0)
    print("✅ Done! Download 'cooking_agent.gif' to watch it.")

if __name__ == "__main__":
    main()