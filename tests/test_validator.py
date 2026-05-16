"""Unit tests for the pre-submission validator."""
import pytest
import config as C
from env.hex_grid import HexGrid
from env.models import (
    AgentAction, AgentState, Cell, CMD_MOVE, CMD_STAY,
    DayOrder, DayState, MapData, Spot,
)
from env.validator import validate_orders


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def make_grid(w=4, h=4):
    return HexGrid(w, h)


def make_map(w=4, h=4, terrain_overrides=None):
    terrain_overrides = terrain_overrides or {}
    cells = [
        Cell(id=i, terrain=terrain_overrides.get(i, C.TERRAIN_PLAIN))
        for i in range(w * h)
    ]
    return MapData(cells=cells, spots=[])


def make_state(agents, steps_left=100, traffic=None):
    return DayState(
        day=1,
        steps_left=steps_left,
        time_limit_ms=5000,
        traffic=traffic or {},
        spot_inventory={},
        my_agents=agents,
        opponent_cells=[],
        collected_series=set(),
        daily_series=[],
        total_udon=0,
    )


def patrol(cell, fuel=20, aid=1):
    return AgentState(id=aid, type=C.AGENT_PATROL, cell=cell, fuel=fuel)


def supply(cell, aid=2):
    return AgentState(id=aid, type=C.AGENT_SUPPLY, cell=cell, fuel=0)


def order(agent_id, *actions):
    return DayOrder(agent_id=agent_id, actions=list(actions))


def mv(direction):
    return AgentAction(cmd=CMD_MOVE, direction=direction)


def stay():
    return AgentAction(cmd=CMD_STAY)


# ------------------------------------------------------------------ #
# Valid orders — baseline                                              #
# ------------------------------------------------------------------ #

class TestValidOrders:
    def test_empty_orders_valid(self):
        mp    = make_map()
        grid  = make_grid()
        state = make_state([patrol(0)])
        ok, errors = validate_orders([], state, mp, grid)
        assert ok
        assert errors == []

    def test_stay_action_valid(self):
        mp    = make_map()
        grid  = make_grid()
        state = make_state([patrol(0)])
        ok, errors = validate_orders([order(1, stay())], state, mp, grid)
        assert ok
        assert errors == []

    def test_single_valid_move(self):
        mp    = make_map()
        grid  = make_grid()
        state = make_state([patrol(0)])
        ok, errors = validate_orders([order(1, mv(2))], state, mp, grid)  # E to cell 1
        assert ok
        assert errors == []

    def test_multiple_valid_moves(self):
        mp    = make_map()
        grid  = make_grid()
        state = make_state([patrol(0, fuel=20)])
        ok, errors = validate_orders([order(1, mv(2), mv(2))], state, mp, grid)
        assert ok
        assert errors == []


# ------------------------------------------------------------------ #
# Lake                                                                 #
# ------------------------------------------------------------------ #

class TestLakeValidation:
    def test_move_into_lake_is_invalid(self):
        mp    = make_map(terrain_overrides={1: C.TERRAIN_LAKE})
        grid  = make_grid()
        state = make_state([patrol(0)])
        ok, errors = validate_orders([order(1, mv(2))], state, mp, grid)
        assert not ok
        assert any("lake" in e for e in errors)

    def test_move_away_from_lake_destination_valid(self):
        # Move SE (dir 3) to cell 4 — not a lake
        mp    = make_map(terrain_overrides={1: C.TERRAIN_LAKE})
        grid  = make_grid()
        state = make_state([patrol(0)])
        ok, errors = validate_orders([order(1, mv(3))], state, mp, grid)
        assert ok


# ------------------------------------------------------------------ #
# Map edge                                                             #
# ------------------------------------------------------------------ #

class TestEdgeValidation:
    def test_move_off_edge_is_invalid(self):
        mp    = make_map()
        grid  = make_grid()
        state = make_state([patrol(0)])
        # Cell 0 (top-left): dir 0 (NW) goes off map
        ok, errors = validate_orders([order(1, mv(0))], state, mp, grid)
        assert not ok
        assert any("off map" in e for e in errors)

    def test_valid_direction_from_corner(self):
        mp    = make_map()
        grid  = make_grid()
        state = make_state([patrol(0)])
        # Dir 2 (E) from cell 0 → cell 1, valid
        ok, errors = validate_orders([order(1, mv(2))], state, mp, grid)
        assert ok


# ------------------------------------------------------------------ #
# Invalid direction value                                              #
# ------------------------------------------------------------------ #

class TestInvalidDirection:
    def test_direction_out_of_range(self):
        mp    = make_map()
        grid  = make_grid()
        state = make_state([patrol(0)])
        ok, errors = validate_orders([order(1, AgentAction(cmd=CMD_MOVE, direction=6))],
                                     state, mp, grid)
        assert not ok
        assert any("invalid direction" in e for e in errors)

    def test_direction_none(self):
        mp    = make_map()
        grid  = make_grid()
        state = make_state([patrol(0)])
        ok, errors = validate_orders([order(1, AgentAction(cmd=CMD_MOVE, direction=None))],
                                     state, mp, grid)
        assert not ok
        assert any("invalid direction" in e for e in errors)


# ------------------------------------------------------------------ #
# Step budget                                                          #
# ------------------------------------------------------------------ #

class TestStepBudget:
    def test_exact_budget_is_valid(self):
        mp    = make_map()
        grid  = make_grid()
        # plain cost=2, budget=2 → exactly 1 move fits
        state = make_state([patrol(0)], steps_left=2)
        ok, errors = validate_orders([order(1, mv(2))], state, mp, grid)
        assert ok

    def test_over_budget_is_invalid(self):
        mp    = make_map()
        grid  = make_grid()
        state = make_state([patrol(0)], steps_left=2)
        # Two moves would need 4 steps but budget is 2
        ok, errors = validate_orders([order(1, mv(2), mv(2))], state, mp, grid)
        assert not ok
        assert any("step budget" in e for e in errors)

    def test_budget_shared_across_agents(self):
        mp    = make_map()
        grid  = make_grid()
        # Agent 1 uses 2 steps (1 move), Agent 2 tries 1 move but 0 steps remain
        state = make_state([patrol(0, aid=1), supply(4, aid=2)], steps_left=2)
        ok, errors = validate_orders([
            order(1, mv(2)),    # uses all 2 steps
            order(2, mv(2)),    # 0 left → error
        ], state, mp, grid)
        assert not ok
        assert any("step budget" in e for e in errors)


# ------------------------------------------------------------------ #
# Fuel budget (patrol only)                                            #
# ------------------------------------------------------------------ #

class TestFuelBudget:
    def test_patrol_fuel_exhaustion_detected(self):
        mp    = make_map()
        grid  = make_grid()
        # Patrol with fuel=1 on plain (fuel_cost=1). Two moves → runs out.
        state = make_state([patrol(0, fuel=1)])
        ok, errors = validate_orders([order(1, mv(2), mv(2))], state, mp, grid)
        assert not ok
        assert any("fuel" in e for e in errors)

    def test_patrol_exact_fuel_is_valid(self):
        mp    = make_map()
        grid  = make_grid()
        state = make_state([patrol(0, fuel=1)])
        ok, errors = validate_orders([order(1, mv(2))], state, mp, grid)
        assert ok

    def test_supply_car_no_fuel_check(self):
        mp    = make_map()
        grid  = make_grid()
        # Supply car has fuel=0 but should not be fuel-checked
        state = make_state([supply(0, aid=1)])
        ok, errors = validate_orders([order(1, mv(2), mv(2))], state, mp, grid)
        assert ok

    def test_mountain_fuel_cost_detected(self):
        mp    = make_map(terrain_overrides={0: C.TERRAIN_MOUNTAIN})
        grid  = make_grid()
        # Mountain fuel cost=2. Patrol with fuel=1 cannot move off mountain.
        state = make_state([patrol(0, fuel=1)])
        ok, errors = validate_orders([order(1, mv(2))], state, mp, grid)
        assert not ok
        assert any("fuel" in e for e in errors)


# ------------------------------------------------------------------ #
# Unknown agent                                                         #
# ------------------------------------------------------------------ #

class TestUnknownAgent:
    def test_unknown_agent_id_reported(self):
        mp    = make_map()
        grid  = make_grid()
        state = make_state([patrol(0, aid=1)])
        ok, errors = validate_orders([order(99, mv(2))], state, mp, grid)
        assert not ok
        assert any("not found" in e for e in errors)
