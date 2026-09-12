import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from env.map_generator import generate_random_scenario
from env.simulator import HexaUdonSimulator
from strategy.lookahead import LookaheadPlanner
from copy import deepcopy

for game_idx in range(100):
    game_seed = 42 + game_idx
    cfg, map_data, agents = generate_random_scenario(seed=game_seed)
    sim = HexaUdonSimulator(cfg, map_data)
    planner = LookaheadPlanner(cfg, map_data, sim)
    
    state = sim.reset(deepcopy(agents))
    day = 1
    while not sim.is_done(state):
        try:
            orders = planner.plan(state)
            state, _ = sim.apply_day(state, orders)
            day += 1
        except Exception as e:
            print(f"CRASH at game_seed={game_seed}, day={day}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
print("All 100 games completed successfully (no crash) in scratch test.")
