"""
Weighted A* pathfinder on a hex grid.

Returns the sequence of AgentActions (move/stay) needed to travel from
src to dst within step and fuel budgets.

Step cost  = SOURCE cell terrain cost (see config.STEP_COST).
Fuel cost  = SOURCE cell terrain cost (see config.FUEL_COST).
Lake cells = impassable (cannot be dst or intermediate node).
"""
from __future__ import annotations

import heapq
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, List, Optional, Tuple

import config as C
from env.hex_grid import HexGrid
from env.models import AgentAction, CMD_MOVE, CMD_STAY


class PathResult:
    def __init__(
        self,
        actions: List[AgentAction],
        total_steps: int,
        total_fuel: int,
        reachable: bool,
    ):
        self.actions     = actions
        self.total_steps = total_steps
        self.total_fuel  = total_fuel
        self.reachable   = reachable

    @staticmethod
    def unreachable() -> PathResult:
        return PathResult([], 0, 0, False)


def find_path(
    grid:         HexGrid,
    terrain:      Dict[int, int],     # cell_id -> terrain type
    traffic:      Dict[int, int],     # road cell_id -> traffic status
    src:          int,
    dst:          int,
    step_budget:  int,
    fuel_budget:  Optional[int] = None,  # None = supply car (no fuel limit)
    fuel_terrain: Optional[Dict[int, int]] = None,  # same as terrain, for clarity
) -> PathResult:
    """
    Find least-step path from src to dst.

    Returns PathResult with the sequence of AgentActions.
    If dst is unreachable within budgets, returns PathResult.unreachable().
    """
    if src == dst:
        return PathResult([], 0, 0, True)

    if terrain.get(dst, C.TERRAIN_LAKE) == C.TERRAIN_LAKE:
        return PathResult.unreachable()

    def step_cost(cell_id: int) -> int:
        t = terrain.get(cell_id, C.TERRAIN_PLAIN)
        if t == C.TERRAIN_ROAD:
            status = traffic.get(cell_id, C.TRAFFIC_CLEAR)
            return C.STEP_COST[C.TERRAIN_ROAD][status]
        cost = C.STEP_COST.get(t)
        return cost if cost is not None else 999999

    def fuel_cost(cell_id: int) -> int:
        t = terrain.get(cell_id, C.TERRAIN_PLAIN)
        return C.FUEL_COST.get(t, 0)

    # h(n) = lower bound on remaining steps to dst
    # Admissible: min step cost per hex = 1 (clear road)
    def heuristic(cell_id: int) -> int:
        return grid.hex_distance(cell_id, dst) * 1

    # State: (f, g_steps, g_fuel, cell, parent_cell, direction_taken)
    # We minimize total steps (g_steps). Fuel is a constraint, not objective.
    INF = float("inf")

    # best_g[cell] = (best_steps_so_far, best_fuel_so_far_at_that_steps)
    best_g: Dict[int, Tuple[int, int]] = {}

    # heap: (f, g_steps, g_fuel, cell)
    heap: List[Tuple] = []
    heapq.heappush(heap, (heuristic(src), 0, 0, src))

    # For path reconstruction
    came_from: Dict[int, Tuple[int, int]] = {}   # cell -> (parent_cell, direction)

    while heap:
        f, g_steps, g_fuel, cur = heapq.heappop(heap)

        if cur == dst:
            return _reconstruct(came_from, dst, g_steps, g_fuel)

        # Prune if we've seen this cell with better cost
        if cur in best_g:
            prev_steps, prev_fuel = best_g[cur]
            if g_steps > prev_steps:
                continue
            if g_steps == prev_steps and g_fuel >= prev_fuel:
                continue
        best_g[cur] = (g_steps, g_fuel)

        sc = step_cost(cur)   # cost of leaving cur
        fc = fuel_cost(cur)   # fuel of leaving cur

        for direction, nbr in grid.neighbors(cur):
            nbr_terrain = terrain.get(nbr, C.TERRAIN_PLAIN)
            if nbr_terrain == C.TERRAIN_LAKE:
                continue   # impassable

            new_steps = g_steps + sc
            new_fuel  = g_fuel  + fc

            if new_steps > step_budget:
                continue
            if fuel_budget is not None and new_fuel > fuel_budget:
                continue

            if nbr in best_g:
                ps, pf = best_g[nbr]
                if new_steps > ps:
                    continue
                if new_steps == ps and new_fuel >= pf:
                    continue

            came_from[nbr] = (cur, direction)
            h = heuristic(nbr)
            heapq.heappush(heap, (new_steps + h, new_steps, new_fuel, nbr))

    return PathResult.unreachable()


def _reconstruct(
    came_from: Dict[int, Tuple[int, int]],
    dst: int,
    total_steps: int,
    total_fuel: int,
) -> PathResult:
    actions: List[AgentAction] = []
    cur = dst
    while cur in came_from:
        parent, direction = came_from[cur]
        actions.append(AgentAction(cmd=CMD_MOVE, direction=direction))
        cur = parent
    actions.reverse()
    return PathResult(actions, total_steps, total_fuel, True)


def multi_waypoint_path(
    grid:        HexGrid,
    terrain:     Dict[int, int],
    traffic:     Dict[int, int],
    start:       int,
    waypoints:   List[int],   # ordered list of targets to visit
    step_budget: int,
    fuel_budget: Optional[int] = None,
) -> List[AgentAction]:
    """
    Chain A* calls through multiple waypoints.
    Returns combined action list, stopping if a waypoint is unreachable.
    """
    all_actions: List[AgentAction] = []
    cur = start
    steps_left = step_budget
    fuel_left  = fuel_budget

    for wp in waypoints:
        result = find_path(
            grid, terrain, traffic, cur, wp,
            step_budget=steps_left,
            fuel_budget=fuel_left,
        )
        if not result.reachable:
            break
        all_actions.extend(result.actions)
        steps_left -= result.total_steps
        if fuel_left is not None:
            fuel_left -= result.total_fuel
        cur = wp

    return all_actions
