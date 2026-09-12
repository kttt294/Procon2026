import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))

from env.map_generator import generate_random_scenario
from env.simulator import HexaUdonSimulator
from rl.mappo import MAPPOTrainer
from strategy.mcts import MCTSPlanner
from copy import deepcopy

def dry_run_mcts():
    print("Initializing random scenario...")
    cfg, map_data, agents = generate_random_scenario(seed=42)
    sim = HexaUdonSimulator(cfg, map_data)
    
    print("Initializing MAPPO model (CPU)...")
    trainer = MAPPOTrainer(device="cpu")
    rl_model = trainer.model
    rl_model.eval()
    
    # Let's set the fuel max so that state normalization works correctly
    observed = [a.fuel for a in agents if a.is_patrol()]
    if observed:
        rl_model.set_fuel_max(max(observed))
        print(f"Fuel max set to {max(observed)}")
        
    print("Initializing MCTS Planner...")
    mcts_planner = MCTSPlanner(cfg, map_data, sim, rl_model, time_budget_ms=1000, beam_width=3, max_depth=2)
    
    print("Resetting simulator...")
    state = sim.reset(deepcopy(agents))
    
    print("Running MCTS planning...")
    try:
        orders = mcts_planner.plan(state)
        print(f"Planning succeeded! Generated orders for {len(orders)} agents:")
        for o in orders:
            print(f"  Agent {o.agent_id}: {o.actions}")
    except Exception as e:
        print(f"MCTS Plan crashed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    dry_run_mcts()
