"""Unit tests for A* pathfinder: reachability, costs, budget constraints."""
import pytest
import config as C
from env.hex_grid import HexGrid
from env.models import AgentAction, CMD_MOVE
from pathfinding.astar import find_path, multi_waypoint_path


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def plain_map(w=4, h=4):
    """All-plain terrain dict and a matching HexGrid."""
    grid    = HexGrid(w, h)
    terrain = {i: C.TERRAIN_PLAIN for i in range(w * h)}
    return grid, terrain


def apply_path(actions, start, grid, terrain):
    """Walk a path and return the final cell (to verify correctness)."""
    cur = start
    for act in actions:
        assert act.cmd == CMD_MOVE
        nxt = grid.neighbor_in_dir(cur, act.direction)
        assert nxt is not None, f"path goes off map from {cur}"
        assert terrain.get(nxt) != C.TERRAIN_LAKE, f"path enters lake at {nxt}"
        cur = nxt
    return cur


# ------------------------------------------------------------------ #
# Basic reachability                                                   #
# ------------------------------------------------------------------ #

class TestBasicReachability:
    def test_same_src_dst(self):
        grid, terrain = plain_map()
        r = find_path(grid, terrain, {}, 0, 0, step_budget=100)
        assert r.reachable
        assert r.actions     == []
        assert r.total_steps == 0
        assert r.total_fuel  == 0

    def test_adjacent_plain(self):
        grid, terrain = plain_map()
        r = find_path(grid, terrain, {}, 0, 1, step_budget=100)
        assert r.reachable
        assert len(r.actions)  == 1
        assert r.total_steps   == C.STEP_COST[C.TERRAIN_PLAIN]
        assert r.total_fuel    == C.FUEL_COST[C.TERRAIN_PLAIN]

    def test_path_lands_on_destination(self):
        grid, terrain = plain_map(6, 6)
        for dst in [5, 11, 25, 35]:
            r = find_path(grid, terrain, {}, 0, dst, step_budget=1000)
            assert r.reachable
            assert apply_path(r.actions, 0, grid, terrain) == dst

    def test_two_step_same_row(self):
        grid, terrain = plain_map()
        # cell 0 → cell 2 along row 0: 2 hops, cost = 2 × plain
        r = find_path(grid, terrain, {}, 0, 2, step_budget=100)
        assert r.reachable
        assert len(r.actions)  == 2
        assert r.total_steps   == 2 * C.STEP_COST[C.TERRAIN_PLAIN]

    def test_lake_destination_unreachable(self):
        grid, terrain = plain_map()
        terrain[1] = C.TERRAIN_LAKE
        r = find_path(grid, terrain, {}, 0, 1, step_budget=100)
        assert not r.reachable

    def test_lake_forces_detour(self):
        # 4x4 map: cell 1 (0,1) is lake. Path 0→2 must detour (≥ 3 hops).
        grid, terrain = plain_map()
        terrain[1] = C.TERRAIN_LAKE
        r = find_path(grid, terrain, {}, 0, 2, step_budget=100)
        assert r.reachable
        assert len(r.actions) >= 3
        assert apply_path(r.actions, 0, grid, terrain) == 2


# ------------------------------------------------------------------ #
# Budget constraints                                                   #
# ------------------------------------------------------------------ #

class TestBudgets:
    def test_step_budget_too_small(self):
        grid, terrain = plain_map()
        # plain cost = 2; need at least 2 to reach cell 1
        r = find_path(grid, terrain, {}, 0, 1, step_budget=1)
        assert not r.reachable

    def test_step_budget_exact(self):
        grid, terrain = plain_map()
        r = find_path(grid, terrain, {}, 0, 1, step_budget=C.STEP_COST[C.TERRAIN_PLAIN])
        assert r.reachable

    def test_fuel_budget_too_small(self):
        grid, terrain = plain_map()
        # fuel cost = 1 per plain hop; need at least 1
        r = find_path(grid, terrain, {}, 0, 1, step_budget=100, fuel_budget=0)
        assert not r.reachable

    def test_fuel_budget_exact(self):
        grid, terrain = plain_map()
        r = find_path(grid, terrain, {}, 0, 1, step_budget=100,
                      fuel_budget=C.FUEL_COST[C.TERRAIN_PLAIN])
        assert r.reachable

    def test_fuel_budget_none_means_no_limit(self):
        grid, terrain = plain_map(6, 6)
        # fuel_budget=None → supply car, no fuel constraint
        r = find_path(grid, terrain, {}, 0, 35, step_budget=10000, fuel_budget=None)
        assert r.reachable

    def test_mountain_costs_more(self):
        grid, terrain = plain_map()
        terrain[0] = C.TERRAIN_MOUNTAIN
        r = find_path(grid, terrain, {}, 0, 1, step_budget=100)
        assert r.reachable
        assert r.total_steps == C.STEP_COST[C.TERRAIN_MOUNTAIN]
        assert r.total_fuel  == C.FUEL_COST[C.TERRAIN_MOUNTAIN]


# ------------------------------------------------------------------ #
# Traffic-dependent road cost                                          #
# ------------------------------------------------------------------ #

class TestRoadCost:
    def _road_map(self):
        grid, terrain = plain_map()
        terrain[0] = C.TERRAIN_ROAD
        return grid, terrain

    def test_road_clear(self):
        grid, terrain = self._road_map()
        r = find_path(grid, terrain, {}, 0, 1, step_budget=100)
        assert r.reachable
        assert r.total_steps == C.STEP_COST[C.TERRAIN_ROAD][C.TRAFFIC_CLEAR]

    def test_road_busy(self):
        grid, terrain = self._road_map()
        r = find_path(grid, terrain, {0: C.TRAFFIC_BUSY}, 0, 1, step_budget=100)
        assert r.reachable
        assert r.total_steps == C.STEP_COST[C.TERRAIN_ROAD][C.TRAFFIC_BUSY]

    def test_road_congested(self):
        grid, terrain = self._road_map()
        r = find_path(grid, terrain, {0: C.TRAFFIC_CONGESTED}, 0, 1, step_budget=100)
        assert r.reachable
        assert r.total_steps == C.STEP_COST[C.TERRAIN_ROAD][C.TRAFFIC_CONGESTED]

    def test_congested_exceeds_budget(self):
        grid, terrain = self._road_map()
        # congested cost = 4; budget = 3 → unreachable
        r = find_path(grid, terrain, {0: C.TRAFFIC_CONGESTED}, 0, 1, step_budget=3)
        assert not r.reachable


# ------------------------------------------------------------------ #
# Multi-waypoint                                                       #
# ------------------------------------------------------------------ #

class TestMultiWaypoint:
    def test_chain_two_waypoints(self):
        grid, terrain = plain_map()
        # 0 → 1 → 2: two hops, each costs 2 steps
        actions = multi_waypoint_path(grid, terrain, {}, 0, [1, 2], step_budget=100)
        cur = apply_path(actions, 0, grid, terrain)
        assert cur == 2
        assert len(actions) == 2

    def test_stops_at_unreachable_waypoint(self):
        grid, terrain = plain_map()
        terrain[1] = C.TERRAIN_LAKE
        # 0 → 1 (lake) unreachable → returns empty list, doesn't proceed to 2
        actions = multi_waypoint_path(grid, terrain, {}, 0, [1, 2], step_budget=100)
        assert len(actions) == 0

    def test_budget_respected_across_segments(self):
        grid, terrain = plain_map()
        # plain cost=2; budget=2 → only the first hop (0→1) fits; second hop fails
        actions = multi_waypoint_path(grid, terrain, {}, 0, [1, 2],
                                      step_budget=C.STEP_COST[C.TERRAIN_PLAIN])
        assert len(actions) == 1
        assert apply_path(actions, 0, grid, terrain) == 1

    def test_fuel_respected_across_segments(self):
        grid, terrain = plain_map()
        # plain fuel=1; fuel_budget=1 → only first hop fits
        actions = multi_waypoint_path(grid, terrain, {}, 0, [1, 2],
                                      step_budget=1000,
                                      fuel_budget=C.FUEL_COST[C.TERRAIN_PLAIN])
        assert len(actions) == 1
