"""
Traffic predictor: estimates opponent road-step counts for the current day.

Heuristic: assume each opponent agent moves greedily toward the nearest
available spot (by hex distance), using A* to find their path.
Road cells along that path accumulate estimated step counts.

These estimates can be fed into TrafficModel.record_day() so that
day N+1 traffic reflects both our own steps and predicted opponent steps.
This lets the lookahead planner proactively avoid roads that will likely
be congested tomorrow.

Limitations:
  - Assumes greedy opponents (overestimates traffic if opponents are smarter)
  - Ignores opponent fuel (assumes unlimited — conservative estimate)
  - No inter-agent coordination among opponents
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, List, Optional

import config as C
from env.hex_grid import HexGrid
from env.models import DayState, MapData
from pathfinding.astar import find_path


def predict_opponent_road_steps(
    state:        DayState,
    map_data:     MapData,
    grid:         HexGrid,
    steps_budget: int,
) -> Dict[int, float]:
    """
    Estimate road step counts from all known opponent agents for today.

    Returns {road_cell_id: estimated_steps_from_opponents}.
    Returns {} if there are no opponent positions or no spots on the map.
    """
    if not state.opponent_cells or not map_data.spots:
        return {}

    terrain = {c.id: c.terrain for c in map_data.cells}
    road_steps: Dict[int, float] = {}

    for opp_cell in state.opponent_cells:
        target = _nearest_spot(opp_cell, map_data, grid)
        if target is None:
            continue

        result = find_path(
            grid, terrain, state.traffic,
            src=opp_cell,
            dst=target,
            step_budget=steps_budget,
            fuel_budget=None,   # opponents' fuel is unknown; assume no limit
        )
        if not result.reachable:
            continue

        _accumulate_road_steps(result.actions, opp_cell, terrain, state.traffic,
                               grid, road_steps)

    return road_steps


def predict_and_scale(
    state:        DayState,
    map_data:     MapData,
    grid:         HexGrid,
    steps_budget: int,
    n_opponents:  int,
) -> Dict[int, float]:
    """
    Predict traffic and scale by number of opponent teams (when only partial
    opponent positions are visible).

    n_opponents: total number of opponent teams (from MatchConfig.n_teams - 1).
    visible    : len(state.opponent_cells) opponent agents are visible.
    Scaling factor = n_opponents / max(1, n_visible_teams).
    """
    raw = predict_opponent_road_steps(state, map_data, grid, steps_budget)
    if not raw:
        return {}
    n_visible = max(1, len(state.opponent_cells))
    scale = n_opponents / n_visible
    return {cell: steps * scale for cell, steps in raw.items()}


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _nearest_spot(cell: int, map_data: MapData, grid: HexGrid) -> Optional[int]:
    """Return cell_id of the spot nearest to `cell` by hex distance."""
    best_dist = float("inf")
    best_cell = None
    for spot in map_data.spots:
        d = grid.hex_distance(cell, spot.cell_id)
        if d < best_dist:
            best_dist = d
            best_cell = spot.cell_id
    return best_cell


def _accumulate_road_steps(
    actions,
    start_cell:  int,
    terrain:     Dict[int, int],
    traffic:     Dict[int, int],
    grid:        HexGrid,
    road_steps:  Dict[int, float],
) -> None:
    """Walk `actions` from `start_cell` and accumulate steps on road cells."""
    cur = start_cell
    for action in actions:
        src_terrain = terrain.get(cur, C.TERRAIN_PLAIN)
        if src_terrain == C.TERRAIN_ROAD:
            status = traffic.get(cur, C.TRAFFIC_CLEAR)
            step_c = C.STEP_COST[C.TERRAIN_ROAD][status]
            road_steps[cur] = road_steps.get(cur, 0.0) + step_c

        nxt = grid.neighbor_in_dir(cur, action.direction)
        if nxt is None:
            break
        cur = nxt
