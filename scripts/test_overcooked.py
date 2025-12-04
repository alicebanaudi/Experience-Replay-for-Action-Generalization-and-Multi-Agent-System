# scripts/test_overcooked.py
import sys
import os
import numpy as np

# --- PATH SETUP ---
# This ensures Python can find your 'src' folder even if you didn't run 'pip install -e .'
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

print(f"📂 Looking for project modules in: {project_root}")
# ------------------

def run_test():
    print("🔄 Attempting to import Overcooked Wrapper...")
    try:
        # Import your specific wrapper
        from env.overcooked_wrapper import OvercookedMAEnv
    except ImportError as e:
        print("\n❌ CRITICAL ERROR: Could not import the Wrapper.")
        print(f"Error details: {e}")
        print("💡 Tip: Did you create 'src/env/overcooked_wrapper.py'?")
        return

    print("✅ Import successful.")
    
    try:
        # Initialize the environment (using a simple layout)
        env = OvercookedMAEnv(layout_name="cramped_room")
        print("✅ Environment initialized.")
        
        # Check Observation Space
        obs_shape = env.observation_space.shape
        print(f"👀 Observation Shape: {obs_shape}")
        # Expecting: (Width, Height, 52) because we stacked 2 agents (26 feats * 2)
        
        # Reset
        obs, _ = env.reset()
        print("✅ Reset successful.")

        # Run a short simulation loop
        print("\n🏃 Running 50 random steps...")
        total_reward = 0
        
        for step in range(50):
            # Sample a random joint action
            action = env.action_space.sample()
            
            # Step the environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            
            # Basic sanity checks on data types
            assert isinstance(reward, (float, int)), "Reward must be a number"
            assert isinstance(terminated, bool), "Terminated must be a bool"
            
            if step % 10 == 0:
                print(f"   Step {step}: Reward={reward}, Terminated={terminated}")

            if terminated or truncated:
                env.reset()
                
        print(f"\n✅ Test Finished! Total Reward collected: {total_reward}")
        print("🎉 The Overcooked environment is installed and working correctly!")

    except Exception as e:
        print(f"\n❌ RUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_test()