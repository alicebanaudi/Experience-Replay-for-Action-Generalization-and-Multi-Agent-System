import numpy as np
from overcooked_ai_py.mdp.actions import Action, Direction

class SimpleBot:
    """
    A robust heuristic bot.
    Fixes:
    - No dependency on 'Action.movements'
    - No dependency on 'mdp.is_valid_position'
    - Manually checks grid bounds and walls
    """
    def __init__(self, agent_index=1):
        self.agent_index = agent_index
        
        # Hard-coded standard Overcooked movements (dx, dy)
        self.POSSIBLE_MOVES = [
            (0, -1), # North
            (0, 1),  # South
            (1, 0),  # East
            (-1, 0)  # West
        ]

    def get_action(self, state, mdp):
        agent = state.players[self.agent_index]
        
        # --- 1. SAFE TARGET FINDING ---
        targets = []
        try:
            if not agent.has_object():
                targets = mdp.get_onion_dispenser_locations()
            elif agent.held_object.name == 'onion':
                targets = mdp.get_pot_locations()
            elif agent.held_object.name == 'dish':
                targets = mdp.get_pot_locations() 
            elif agent.held_object.name == 'soup':
                targets = mdp.get_serving_locations()
            else:
                return Action.STAY
        except AttributeError:
            return Action.STAY

        if not targets:
            return Action.STAY
            
        # --- 2. ROBUST MOTION PLANNING ---
        best_action = Action.STAY
        min_dist = float('inf')
        
        x, y = agent.position
        
        # Check all 4 directions
        for (dx, dy) in self.POSSIBLE_MOVES:
            nx, ny = x + dx, y + dy
            
            # 1. Check Map Bounds
            if ny < 0 or ny >= len(mdp.terrain_mtx) or nx < 0 or nx >= len(mdp.terrain_mtx[0]):
                continue

            # 2. Check Collisions (Walls/Counters)
            terrain_type = mdp.terrain_mtx[ny][nx]
            if terrain_type == 'X':
                continue
            # --------------------------------------

            # Check distance to nearest target
            for (tx, ty) in targets:
                dist = abs(nx - tx) + abs(ny - ty)
                if dist < min_dist:
                    min_dist = dist
                    # Convert (dx, dy) back to Action object
                    if (dx, dy) == (0, -1): best_action = Action.INDEX_TO_ACTION[0] # North
                    if (dx, dy) == (0, 1):  best_action = Action.INDEX_TO_ACTION[1] # South
                    if (dx, dy) == (1, 0):  best_action = Action.INDEX_TO_ACTION[2] # East
                    if (dx, dy) == (-1, 0): best_action = Action.INDEX_TO_ACTION[3] # West
                    
        # --- 3. INTERACTION LOGIC ---
        for (tx, ty) in targets:
            # Manhattan distance of 1 means we are adjacent
            if abs(x - tx) + abs(y - ty) == 1:
                # Check if facing
                if (x + agent.orientation[0] == tx) and (y + agent.orientation[1] == ty):
                    return Action.INTERACT
                
                # Turn to face target
                if x < tx: return Action.INDEX_TO_ACTION[2] # East
                if x > tx: return Action.INDEX_TO_ACTION[3] # West
                if y < ty: return Action.INDEX_TO_ACTION[1] # South
                if y > ty: return Action.INDEX_TO_ACTION[0] # North

        return best_action