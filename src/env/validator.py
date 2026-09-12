"""
Pre-submission validator for DayOrders.

Catches problems before POST to server:
  - Invalid direction value (not 0-5)
  - Direction goes off the map edge
  - Destination is a lake (impassable)
  - Shared step budget exceeded across all agents
  - Patrol fuel exhausted mid-route

Usage:
    ok, errors = validate_orders(orders, state, map_data, grid)
    if not ok:
        orders = fallback_orders  # use safe alternative

Contest strategy: submit greedy orders (~100ms), then try to compute
better orders and resubmit if validate_orders passes and time allows.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, List, Tuple

import config as C
from env.hex_grid import HexGrid
from env.models import CMD_MOVE, CMD_STAY, DayOrder, DayState, MapData


def validate_orders(
    orders:   List[DayOrder],
    state:    DayState,
    map_data: MapData,
    grid:     HexGrid,
) -> Tuple[bool, List[str]]:
    """
    Validate a list of DayOrders against the current game state.

    Simulates the same sequential processing order as the simulator:
    agents are processed in order; step budget is shared and decremented
    after each successful move.

    Returns (is_valid, errors).
    is_valid is True only when no errors are found.
    """
    errors: List[str] = []
    agents_by_id = state.agents_by_id()
    steps_used = 0
    seen_ids = set()

    for order in orders:
        if order.agent_id in seen_ids:
            errors.append(f"agent {order.agent_id}: duplicate order")
            continue
        seen_ids.add(order.agent_id)
        agent = agents_by_id.get(order.agent_id)
        if agent is None:
            errors.append(f"agent {order.agent_id}: not found in state")
            continue

        cur_cell = agent.cell
        fuel_used = 0

        for i, action in enumerate(order.actions):
            if action.cmd == CMD_STAY:
                if action.direction is not None:
                    errors.append(f"agent {order.agent_id}: stay cannot have a direction")
                continue
            if action.cmd != CMD_MOVE:
                errors.append(f"agent {order.agent_id}: unknown command {action.cmd!r}")
                continue

            # --- direction validity ---
            if type(action.direction) is not int or action.direction not in range(C.N_DIRECTIONS):
                errors.append(
                    f"agent {order.agent_id} step {i}: "
                    f"invalid direction {action.direction!r}"
                )
                continue

            # --- destination exists (not off the edge) ---
            dst = grid.neighbor_in_dir(cur_cell, action.direction)
            if dst is None:
                errors.append(
                    f"agent {order.agent_id} step {i}: "
                    f"direction {action.direction} goes off map from cell {cur_cell}"
                )
                continue

            # --- destination is passable ---
            if map_data.cell_map[dst].terrain == C.TERRAIN_LAKE:
                errors.append(
                    f"agent {order.agent_id} step {i}: "
                    f"moving into lake at cell {dst}"
                )
                continue

            # --- costs (source-based, matching simulator) ---
            step_c = _step_cost(cur_cell, state.traffic, map_data)
            fuel_c = _fuel_cost(cur_cell, map_data)

            # --- shared step budget ---
            if steps_used + step_c > state.steps_left:
                errors.append(
                    f"agent {order.agent_id} step {i}: "
                    f"step budget exceeded "
                    f"({steps_used}+{step_c} > {state.steps_left})"
                )
                break   # remaining actions for this agent are also blocked

            # --- patrol fuel ---
            if agent.is_patrol() and fuel_used + fuel_c > agent.fuel:
                errors.append(
                    f"agent {order.agent_id} step {i}: "
                    f"fuel exhausted "
                    f"({fuel_used}+{fuel_c} > {agent.fuel})"
                )
                break

            steps_used += step_c
            fuel_used  += fuel_c
            cur_cell    = dst

    return len(errors) == 0, errors


# ------------------------------------------------------------------ #
# Internal cost helpers — mirror simulator._step_cost / _fuel_cost    #
# ------------------------------------------------------------------ #

def _step_cost(cell_id: int, traffic: Dict[int, int], map_data: MapData) -> int:
    terrain = map_data.cell_map[cell_id].terrain
    if terrain == C.TERRAIN_ROAD:
        status = traffic.get(cell_id, C.TRAFFIC_CLEAR)
        return C.STEP_COST[C.TERRAIN_ROAD][status]
    cost = C.STEP_COST.get(terrain)
    return cost if cost is not None else 0


def _fuel_cost(cell_id: int, map_data: MapData) -> int:
    terrain = map_data.cell_map[cell_id].terrain
    return C.FUEL_COST.get(terrain, 0)
