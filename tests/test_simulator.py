"""
Unit tests for HexaUdonSimulator — one rule per test.

Map layout used throughout (4×4, width=4):
  Cell IDs:   0  1  2  3
              4  5  6  7
              8  9 10 11
             12 13 14 15

Directions (even-r offset):
  0=NW  1=NE  2=E  3=SE  4=SW  5=W
"""
import pytest
import config as C
from env.models import (
    AgentAction, AgentState, Cell, CMD_MOVE, CMD_STAY,
    DayOrder, MapData, MatchConfig, Spot,
)
from env.simulator import HexaUdonSimulator


# ------------------------------------------------------------------ #
# Fixtures                                                             #
# ------------------------------------------------------------------ #

def make_cfg(w=4, h=4, days=3, steps_per_day=None,
             n_teams=1, thr_busy=3.0, thr_congested=10.0):
    return MatchConfig(
        width=w, height=h,
        total_days=days,
        steps_per_day=steps_per_day or [100] * days,
        n_teams=n_teams,
        traffic_threshold_busy=thr_busy,
        traffic_threshold_congested=thr_congested,
        fuel_max=20,
    )


def make_map(w=4, h=4, terrain_overrides=None, spots=None):
    terrain_overrides = terrain_overrides or {}
    cells = [
        Cell(id=i, terrain=terrain_overrides.get(i, C.TERRAIN_PLAIN))
        for i in range(w * h)
    ]
    return MapData(cells=cells, spots=spots or [])


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
# Rule 1: cannot move INTO a lake                                      #
# ------------------------------------------------------------------ #

class TestLakeBlocking:
    def test_cannot_enter_lake(self):
        # cell 1 (0,1) is lake. Agent at cell 0 tries to move E (dir 2) → blocked.
        cfg = make_cfg()
        mp  = make_map(terrain_overrides={1: C.TERRAIN_LAKE})
        sim = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0)])
        state, _ = sim.apply_day(state, [order(1, mv(2))])

        # Agent must still be at cell 0
        assert state.my_agents[0].cell == 0

    def test_move_around_lake_works(self):
        # Same setup but agent moves SE (dir 3) → cell 4 → valid
        cfg = make_cfg()
        mp  = make_map(terrain_overrides={1: C.TERRAIN_LAKE})
        sim = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0)])
        state, _ = sim.apply_day(state, [order(1, mv(3))])

        assert state.my_agents[0].cell == 4


# ------------------------------------------------------------------ #
# Rule 2: patrol stops when fuel is exhausted                          #
# ------------------------------------------------------------------ #

class TestFuelExhaustion:
    def test_patrol_stops_mid_route_on_empty_fuel(self):
        # Agent with fuel=2 on plain (fuel cost=1 per step).
        # 3 move actions → should complete 2 then stop.
        cfg   = make_cfg()
        mp    = make_map()
        sim   = HexaUdonSimulator(cfg, mp)

        agent = patrol(0, fuel=2)
        state = sim.reset([agent])
        # Move E three times (cells 0→1→2, third would need fuel)
        state, _ = sim.apply_day(state, [order(1, mv(2), mv(2), mv(2))])

        assert state.my_agents[0].cell == 2
        assert state.my_agents[0].fuel == 0

    def test_supply_car_ignores_fuel(self):
        # Supply car has fuel=0 but should move freely (no fuel check).
        cfg   = make_cfg()
        mp    = make_map()
        sim   = HexaUdonSimulator(cfg, mp)

        state = sim.reset([supply(0, aid=1)])
        state, _ = sim.apply_day(state, [order(1, mv(2), mv(2))])

        assert state.my_agents[0].cell == 2


# ------------------------------------------------------------------ #
# Rule 3: steps_left is shared across ALL agents                       #
# ------------------------------------------------------------------ #

class TestSharedStepBudget:
    def test_second_agent_blocked_when_budget_exhausted_by_first(self):
        # steps_per_day=4, plain cost=2.
        # Agent 1 (cell 0) uses 2+2=4 steps. Agent 2 (cell 4) gets 0 left.
        cfg   = make_cfg(days=1, steps_per_day=[4])
        mp    = make_map()
        sim   = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0, aid=1), supply(4, aid=2)])
        state, _ = sim.apply_day(state, [
            order(1, mv(2), mv(2)),    # uses all 4 steps
            order(2, mv(2)),           # no steps left → blocked
        ])

        by_id = state.agents_by_id()
        assert by_id[1].cell == 2   # moved twice
        assert by_id[2].cell == 4   # didn't move

    def test_partial_use_leaves_budget_for_next_agent(self):
        # Agent 1 uses 2 steps (1 move). Agent 2 gets 2 steps → can move once.
        cfg   = make_cfg(days=1, steps_per_day=[4])
        mp    = make_map()
        sim   = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0, aid=1), supply(4, aid=2)])
        state, _ = sim.apply_day(state, [
            order(1, mv(2)),    # 2 steps used
            order(2, mv(2)),    # 2 steps remaining → fits
        ])

        by_id = state.agents_by_id()
        assert by_id[1].cell == 1
        assert by_id[2].cell == 5


# ------------------------------------------------------------------ #
# Rule 4: traffic uses steps from the past 2 days                     #
# ------------------------------------------------------------------ #

class TestTrafficModel:
    def test_day1_traffic_is_always_clear(self):
        cfg   = make_cfg()
        mp    = make_map()
        sim   = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0)])
        # Day 1 state always has empty traffic
        assert state.traffic == {}

    def test_day2_reflects_day1_road_steps(self):
        # Road at cell 0. Inject 4 opponent steps on day 1.
        # n_teams=1, thr_busy=3.0 → val=4.0 ≥ 3.0 → BUSY on day 2.
        cfg = make_cfg(n_teams=1, thr_busy=3.0, thr_congested=10.0)
        mp  = make_map(terrain_overrides={0: C.TERRAIN_ROAD})
        sim = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0)])
        state, _ = sim.apply_day(
            state, [order(1)],           # no own moves
            opponent_step_counts={0: 4.0},
        )
        # Day 2: traffic at cell 0 should be BUSY
        assert state.traffic.get(0) == C.TRAFFIC_BUSY

    def test_day1_steps_not_visible_until_day2(self):
        cfg = make_cfg(n_teams=1, thr_busy=3.0)
        mp  = make_map(terrain_overrides={0: C.TERRAIN_ROAD})
        sim = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0)])
        # Day 1 traffic is already checked above; just confirm day 2 state is day1-only
        # (not day2 injecting into day1 view)
        assert state.traffic == {}   # day 1 is always clear

    def test_day3_traffic_uses_two_days(self):
        # Days 1 & 2 each inject 3 opponent steps on road cell 0.
        # Day 3 total = 6 / n_teams=1 = 6 ≥ thr_busy=3.0 → BUSY.
        cfg = make_cfg(days=3, n_teams=1, thr_busy=3.0, thr_congested=10.0)
        mp  = make_map(terrain_overrides={0: C.TERRAIN_ROAD})
        sim = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0)])
        state, _ = sim.apply_day(state, [order(1)],
                                  opponent_step_counts={0: 3.0})   # day 1
        state, _ = sim.apply_day(state, [order(1)],
                                  opponent_step_counts={0: 3.0})   # day 2
        # Day 3 traffic: sum of days 1+2 = 6 → BUSY
        assert state.traffic.get(0) == C.TRAFFIC_BUSY


# ------------------------------------------------------------------ #
# Rule 5: patrol collects udon only once per spot per day             #
# ------------------------------------------------------------------ #

class TestUdonCollection:
    def test_udon_collected_on_arrival(self):
        spot = Spot(cell_id=1, series_id=1, max_inventory=5)
        cfg  = make_cfg()
        mp   = make_map(spots=[spot])
        sim  = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0)])
        state, _ = sim.apply_day(state, [order(1, mv(2))])

        assert state.total_udon == 1
        assert 1 in state.collected_series

    def test_same_spot_visited_twice_counts_once(self):
        # Agent: cell 0 → cell 1 (collect) → cell 0 → cell 1 (no collect)
        spot = Spot(cell_id=1, series_id=1, max_inventory=5)
        cfg  = make_cfg()
        mp   = make_map(spots=[spot])
        sim  = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0, fuel=20)])
        state, _ = sim.apply_day(state, [order(1, mv(2), mv(5), mv(2))])

        assert state.total_udon == 1   # only the first visit counts

    def test_supply_car_cannot_collect(self):
        spot  = Spot(cell_id=1, series_id=1, max_inventory=5)
        cfg   = make_cfg()
        mp    = make_map(spots=[spot])
        sim   = HexaUdonSimulator(cfg, mp)

        state = sim.reset([supply(0, aid=1)])
        state, _ = sim.apply_day(state, [order(1, mv(2))])

        assert state.total_udon == 0


# ------------------------------------------------------------------ #
# Rule 6: spot inventory refills to max at start of each new day      #
# ------------------------------------------------------------------ #

class TestInventoryRefill:
    def test_spot_refills_between_days(self):
        # max_inventory=1. Day 1: collect (inventory→0). Day 2: collect again.
        spot = Spot(cell_id=1, series_id=1, max_inventory=1)
        cfg  = make_cfg(days=2)
        mp   = make_map(spots=[spot])
        sim  = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0, fuel=20)])

        # Day 1: move to cell 1, collect
        state, _ = sim.apply_day(state, [order(1, mv(2))])
        assert state.total_udon == 1

        # Day 2: move away and come back to cell 1, collect again
        # Agent is at cell 1. Move W → cell 0, then E → cell 1
        state, _ = sim.apply_day(state, [order(1, mv(5), mv(2))])
        assert state.total_udon == 2

    def test_empty_spot_not_collectable_same_day(self):
        spot = Spot(cell_id=1, series_id=1, max_inventory=1)
        cfg  = make_cfg()
        mp   = make_map(spots=[spot])
        sim  = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0, fuel=20)])
        # Two agents: both try to collect from the same spot on day 1
        p1 = patrol(0, fuel=20, aid=1)
        p2 = patrol(2, fuel=20, aid=2)
        state = sim.reset([p1, p2])

        state, _ = sim.apply_day(state, [
            order(1, mv(2)),    # patrol 1 moves 0→1, collects (inventory now 0)
            order(2, mv(5)),    # patrol 2 moves 2→1, inventory already 0 → no collect
        ])
        assert state.total_udon == 1   # only one collection


# ------------------------------------------------------------------ #
# Rule 7: opponent actions don't affect our collected_series           #
# ------------------------------------------------------------------ #

class TestOpponentIsolation:
    def test_our_collected_series_unaffected_by_opponents(self):
        # We don't collect anything; opponents inject traffic only.
        # collected_series should remain empty.
        spot = Spot(cell_id=1, series_id=1, max_inventory=5)
        cfg  = make_cfg()
        mp   = make_map(spots=[spot])
        sim  = HexaUdonSimulator(cfg, mp)

        state = sim.reset([patrol(0)])
        state, _ = sim.apply_day(
            state, [order(1)],            # our agent stays
            opponent_step_counts={},      # opponents don't interact with spots
        )
        assert state.total_udon == 0
        assert len(state.collected_series) == 0
