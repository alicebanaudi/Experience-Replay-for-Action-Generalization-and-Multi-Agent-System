import numpy as np
from env.overcooked_wrapper import OvercookedSingleAgentEnv
from overcooked_ai_py.mdp.actions import Action
import copy

def super_debug():
    print("🤖 STARTING MAP SCANNER DEBUG...")
    
    # 1. Init Env
    env = OvercookedSingleAgentEnv(layout_name="asymmetric_advantages")
    env.reset()
    
    # 2. SCAN THE MAP for an Onion ('O')
    grid = env.mdp.terrain_mtx
    height = len(grid)
    width = len(grid[0])
    
    target_pos = None
    agent_pos = None
    agent_dir = None
    
    print("\n🗺️  MAP LAYOUT:")
    for y in range(height):
        row_str = ""
        for x in range(width):
            char = grid[y][x]
            row_str += char
            # Find first onion
            if char == 'O' and target_pos is None:
                target_pos = (x, y)
        print(f"   {row_str}")

    if not target_pos:
        print("❌ ERROR: No Onions found in map!")
        return

    # 3. Calculate position next to onion
    tx, ty = target_pos
    print(f"\n🎯 FOUND ONION AT: {target_pos}")
    
    # Try valid adjacent spots
    potential_spots = [
        ((tx+1, ty), (-1, 0)), # Right of onion, face Left
        ((tx-1, ty), (1, 0)),  # Left of onion, face Right
        ((tx, ty+1), (0, -1)), # Below onion, face Up
        ((tx, ty-1), (0, 1)),  # Above onion, face Down
    ]
    
    for pos, orientation in potential_spots:
        px, py = pos
        # Check bounds and if empty space (' ')
        if 0 <= py < height and 0 <= px < width:
            if grid[py][px] == ' ':
                agent_pos = pos
                agent_dir = orientation
                break
    
    if not agent_pos:
        print("❌ ERROR: Could not find empty standing spot next to onion.")
        return

    # 4. TELEPORT
    print(f"⚡ TELEPORTING Agent to {agent_pos} Facing {agent_dir}...")
    
    p0 = env.base_env.state.players[0]
    new_p0 = copy.deepcopy(p0)
    new_p0.position = agent_pos
    new_p0.orientation = agent_dir
    
    # Update State
    current_players = list(env.base_env.state.players)
    current_players[0] = new_p0
    env.base_env.state.players = tuple(current_players)

    # 5. INTERACT
    print("👇 PRESSING INTERACT...")
    fake_interact = np.array([0.9]) 
    obs, reward, done, trunc, info = env.step(fake_interact)
    
    # 6. REPORT
    shaped = info.get('shaped_r_by_agent', [0, 0])
    held = env.base_env.state.players[0].held_object
    held_name = held.name if held else "None"
    
    print(f"\n📊 RESULT:")
    print(f"   Held Object:  {held_name}")
    print(f"   Dense Reward: {shaped[0]}")
    
    if shaped[0] > 0 and held_name == 'onion':
        print("\n✅ SUCCESS! The environment is creating rewards.")
        print("🚀 YOU CAN START TRAINING NOW.")
    else:
        print("\n❌ STILL FAILING. Check reward_shaping_params.")

if __name__ == "__main__":
    super_debug()