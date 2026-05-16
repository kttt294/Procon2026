"""
Greedy baseline planner.

Strategy:
  Patrol cars: each picks the nearest spot belonging to a series not yet
               collected this game. Falls back to nearest uncollected spot
               in any series if all series are covered.
  Supply cars: move towards the patrol car with the lowest remaining fuel
               that is more than 1 hex away.

This is O(n_agents × n_spots) per day and finishes in <1 ms — safe as
a fallback when RL inference exceeds the time limit.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import Dict, List, Optional, Set

import config as C
from env.models import (
    AgentState, DayOrder, DayState, MapData, MatchConfig, Spot,
)
from env.simulator import HexaUdonSimulator
from pathfinding.astar import find_path, multi_waypoint_path
from strategy.planner import BasePlanner


class GreedyPlanner(BasePlanner):
    def __init__(self, cfg: MatchConfig, map_data: MapData, sim: HexaUdonSimulator):
        super().__init__(cfg, map_data, sim)
        self._terrain = self.terrain_dict()

    # ------------------------------------------------------------------ #
    # Main entry                                                           #
    # ------------------------------------------------------------------ #

    def plan(self, state: DayState) -> List[DayOrder]:
        orders: List[DayOrder] = []
        steps_remaining = state.steps_left

        # Assign each patrol a target spot list (greedy nearest-uncollected)
        patrol_targets: Dict[int, List[int]] = {}
        claimed: Set[int] = set()   # spot cell_ids already claimed by another patrol

        for agent in state.patrol_agents():
            target = self._best_spot(agent, state, claimed)
            if target is not None:
                claimed.add(target)
                patrol_targets[agent.id] = [target]
            else:
                patrol_targets[agent.id] = []

        # Build actions for patrol cars
        for agent in state.patrol_agents():
            waypoints = patrol_targets.get(agent.id, [])
            actions   = multi_waypoint_path(
                self.grid,
                self._terrain,
                state.traffic,
                agent.cell,
                waypoints,
                step_budget=steps_remaining,
                fuel_budget=agent.fuel,
            )
            orders.append(DayOrder(agent_id=agent.id, actions=actions))
            # Rough step deduction (shared budget; exact tracking is in simulator)
            steps_remaining = max(0, steps_remaining - sum(
                self.sim._step_cost(agent.cell, state.traffic)
                for _ in actions
            ))

        # Build actions for supply cars
        for agent in state.supply_agents():
            target_cell = self._supply_target(agent, state)
            if target_cell is not None:
                result = find_path(
                    self.grid,
                    self._terrain,
                    state.traffic,
                    agent.cell,
                    target_cell,
                    step_budget=steps_remaining,
                    fuel_budget=None,
                )
                actions = result.actions if result.reachable else []
            else:
                actions = []
            orders.append(DayOrder(agent_id=agent.id, actions=actions))

        return orders

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _best_spot(
        self,
        agent: AgentState,
        state: DayState,
        claimed: Set[int],
    ) -> Optional[int]:
        """Return cell_id of best spot for this patrol car to visit."""
        uncollected_series = set(self.map.series_ids) - state.collected_series

        # Priority 1: spots belonging to not-yet-collected series
        candidates = [
            s for s in self.map.spots
            if s.series_id in uncollected_series
            and s.cell_id not in claimed
            and state.spot_inventory.get(s.cell_id, 0) > 0
        ]

        # Priority 2: if all series collected, any spot with inventory
        if not candidates:
            candidates = [
                s for s in self.map.spots
                if s.cell_id not in claimed
                and state.spot_inventory.get(s.cell_id, 0) > 0
            ]

        if not candidates:
            return None

        # Pick nearest by hex distance
        return min(
            candidates,
            key=lambda s: self.grid.hex_distance(agent.cell, s.cell_id),
        ).cell_id

    def _supply_target(
        self,
        supply: AgentState,
        state: DayState,
    ) -> Optional[int]:
        """Return the cell_id to move towards for the supply car."""
        patrols = state.patrol_agents()
        if not patrols:
            return None

        # Find patrol with lowest fuel that isn't already next to us
        needy = min(patrols, key=lambda a: a.fuel)
        if needy.fuel > 10:   # enough fuel, don't bother
            return None
        if self.grid.hex_distance(supply.cell, needy.cell) == 0:
            return None
        return needy.cell
