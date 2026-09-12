import pytest
import torch
import config as C
from env.hex_grid import HexGrid
from env.models import (
    AgentAction, AgentState, Cell, CMD_MOVE, CMD_STAY,
    DayOrder, DayState, MapData, Spot, MatchConfig
)
from env.simulator import HexaUdonSimulator
from rl.mappo import MAPPOTrainer
from strategy.mcts import MCTSPlanner

def test_mcts_planner_dry_run():
    # Setup simple map and config
    cfg = MatchConfig(
        width=5, height=5,
        total_days=3,
        steps_per_day=[20, 20, 20],
        n_teams=2,
        traffic_threshold_busy=3.0,
        traffic_threshold_congested=7.0,
    )
    cells = [Cell(id=i, terrain=C.TERRAIN_PLAIN) for i in range(25)]
    spots = [Spot(cell_id=5, series_id=1, max_inventory=2)]
    map_data = MapData(cells=cells, spots=spots)
    
    sim = HexaUdonSimulator(cfg, map_data)
    
    # Initialize randomly initialized MAPPO model on CPU
    trainer = MAPPOTrainer(max_spots=10, max_series=5, max_width=10, max_height=10, device="cpu")
    rl_model = trainer.model
    rl_model.eval()
    rl_model.set_fuel_max(20)
    
    # Initialize MCTS Planner
    planner = MCTSPlanner(cfg, map_data, sim, rl_model, time_budget_ms=100, beam_width=2, max_depth=1)
    
    # Create agents
    agents = [
        AgentState(id=1, type=C.AGENT_PATROL, cell=0, fuel=20),
        AgentState(id=2, type=C.AGENT_SUPPLY, cell=10, fuel=0)
    ]
    
    state = sim.reset(agents)
    
    # Plan
    orders = planner.plan(state)
    assert len(orders) == 2
    
    # Verify that we can apply the day without crash
    next_state, reward = sim.apply_day(state, orders)
    assert next_state is not None
