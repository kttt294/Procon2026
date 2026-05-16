"""
Game simulator for HEXA UDON.

Cost convention (based on problem statement):
  - Step cost = terrain of SOURCE cell (where agent currently stands).
  - Fuel cost = same (terrain of SOURCE cell).
  - Cannot move INTO lake (terrain=2).
  - Traffic status affects road cost at source.

Traffic model:
  traffic(cell) = sum of steps our agents + opponent agents spent at that road cell
                  over the past 2 days, divided by n_teams.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from copy import deepcopy
from typing import Dict, List, Optional, Set, Tuple

import config as C
from env.hex_grid import HexGrid
from env.models import (
    AgentAction, AgentState, CMD_MOVE, CMD_STAY,
    DayOrder, DayState, MapData, MatchConfig, Spot,
)


class TrafficModel:
    """Tracks road step counts and computes daily traffic status."""

    def __init__(self, n_teams: int, thr_busy: float, thr_congested: float):
        self.n_teams       = n_teams
        self.thr_busy      = thr_busy
        self.thr_congested = thr_congested
        # history[i] = {cell_id: total_steps_all_teams} for day (i+1)
        self._history: List[Dict[int, float]] = []

    def record_day(self, all_team_steps: Dict[int, float]) -> None:
        """Call at end of each day with aggregated step counts."""
        self._history.append(dict(all_team_steps))

    def compute_status(self, for_day: int) -> Dict[int, int]:
        """Return traffic dict {cell_id: status} for the given day number."""
        if for_day <= 1:
            return {}
        past = self._history[max(0, for_day - 3) : for_day - 1]  # up to 2 days
        total: Dict[int, float] = {}
        for day_counts in past:
            for cid, steps in day_counts.items():
                total[cid] = total.get(cid, 0.0) + steps
        result: Dict[int, int] = {}
        for cid, s in total.items():
            val = s / self.n_teams
            if val >= self.thr_congested:
                result[cid] = C.TRAFFIC_CONGESTED
            elif val >= self.thr_busy:
                result[cid] = C.TRAFFIC_BUSY
        return result

    def reset(self) -> None:
        self._history.clear()


class HexaUdonSimulator:
    """
    Simulates one full game of HEXA UDON.

    Usage:
        sim = HexaUdonSimulator(cfg, map_data)
        state = sim.reset(initial_agents)
        while not sim.is_done(state):
            orders = your_strategy(state)
            state, reward = sim.apply_day(state, orders)
    """

    def __init__(self, cfg: MatchConfig, map_data: MapData):
        self.cfg      = cfg
        self.map      = map_data
        self.grid     = HexGrid(cfg.width, cfg.height)
        self.traffic  = TrafficModel(
            cfg.n_teams,
            cfg.traffic_threshold_busy,
            cfg.traffic_threshold_congested,
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def reset(self, initial_agents: List[AgentState], time_limit_ms: int = 5000) -> DayState:
        self.traffic.reset()
        return DayState(
            day=1,
            steps_left=self.cfg.steps_per_day[0],
            time_limit_ms=time_limit_ms,
            traffic={},
            spot_inventory={s.cell_id: s.max_inventory for s in self.map.spots},
            my_agents=deepcopy(initial_agents),
            opponent_cells=[],
            collected_series=set(),
            daily_series=[],
            total_udon=0,
        )

    def apply_day(
        self,
        state: DayState,
        orders: List[DayOrder],
        opponent_step_counts: Optional[Dict[int, float]] = None,
    ) -> Tuple[DayState, float]:
        """
        Execute one full day's orders and return (next_state, reward).

        opponent_step_counts: {road_cell_id: steps} from opponent agents this day.
        If None, opponents are assumed idle (optimistic; update once we have data).
        """
        # Work on mutable copies
        agents   = {a.id: deepcopy(a) for a in state.my_agents}
        inventory = dict(state.spot_inventory)
        order_map = {o.agent_id: o.actions for o in orders}

        steps_used    = 0
        road_steps: Dict[int, float] = {}   # our own road step counts
        collected_today: Set[Tuple[int, int]] = set()   # (agent_id, spot_cell)
        today_series:  Set[int] = set()
        udon_gained    = 0
        fuel_depleted  = 0

        for agent in agents.values():
            actions = order_map.get(agent.id, [])
            prev_fuel = agent.fuel

            for act in actions:
                if act.cmd == CMD_STAY:
                    continue

                # --- resolve destination ---
                dst = self.grid.neighbor_in_dir(agent.cell, act.direction)
                if dst is None:
                    continue   # map edge

                dst_terrain = self.map.cell_map[dst].terrain
                if dst_terrain == C.TERRAIN_LAKE:
                    continue   # cannot enter lake

                # --- costs based on SOURCE cell ---
                src_terrain = self.map.cell_map[agent.cell].terrain
                step_c = self._step_cost(agent.cell, state.traffic)
                fuel_c = self._fuel_cost(agent.cell)

                # --- check budgets ---
                if steps_used + step_c > state.steps_left:
                    break   # shared budget exhausted

                if agent.is_patrol() and agent.fuel < fuel_c:
                    fuel_depleted += 1
                    break   # fuel exhausted for this agent

                # --- apply move ---
                steps_used += step_c

                # Road step tracking (source is a road)
                if src_terrain == C.TERRAIN_ROAD:
                    road_steps[agent.cell] = road_steps.get(agent.cell, 0.0) + step_c

                agent.cell = dst
                if agent.is_patrol():
                    agent.fuel -= fuel_c

                # --- udon collection (patrol only, on arrival) ---
                if agent.is_patrol():
                    spot = self.map.spot_map.get(agent.cell)
                    key  = (agent.id, agent.cell)
                    if spot and key not in collected_today and inventory.get(agent.cell, 0) > 0:
                        inventory[agent.cell] -= 1
                        collected_today.add(key)
                        today_series.add(spot.series_id)
                        udon_gained += 1

        # --- aggregate traffic (our team + opponents) ---
        combined_road_steps: Dict[int, float] = dict(road_steps)
        if opponent_step_counts:
            for cid, s in opponent_step_counts.items():
                combined_road_steps[cid] = combined_road_steps.get(cid, 0.0) + s
        self.traffic.record_day(combined_road_steps)

        # --- build next state ---
        next_day = state.day + 1
        new_collected = set(state.collected_series) | today_series
        new_series_gained = today_series - state.collected_series

        next_state = DayState(
            day=next_day,
            steps_left=(
                self.cfg.steps_per_day[next_day - 1]
                if next_day <= self.cfg.total_days else 0
            ),
            time_limit_ms=state.time_limit_ms,
            traffic=self.traffic.compute_status(next_day),
            # Spots refill to max at start of each new day
            spot_inventory={s.cell_id: s.max_inventory for s in self.map.spots},
            my_agents=list(agents.values()),
            opponent_cells=[],
            collected_series=new_collected,
            daily_series=state.daily_series + [today_series],
            total_udon=state.total_udon + udon_gained,
            _road_step_counts=road_steps,
        )

        reward = self._reward(new_series_gained, today_series, udon_gained, fuel_depleted, steps_used, state.steps_left)
        return next_state, reward

    def is_done(self, state: DayState) -> bool:
        return state.day > self.cfg.total_days

    # ------------------------------------------------------------------ #
    # Costs                                                                #
    # ------------------------------------------------------------------ #

    def _step_cost(self, cell_id: int, traffic: Dict[int, int]) -> int:
        terrain = self.map.cell_map[cell_id].terrain
        if terrain == C.TERRAIN_ROAD:
            status = traffic.get(cell_id, C.TRAFFIC_CLEAR)
            return C.STEP_COST[C.TERRAIN_ROAD][status]
        return C.STEP_COST[terrain]

    def _fuel_cost(self, cell_id: int) -> int:
        terrain = self.map.cell_map[cell_id].terrain
        return C.FUEL_COST.get(terrain, 0)

    def step_cost_of(self, cell_id: int, traffic: Dict[int, int]) -> Optional[int]:
        """Public helper for pathfinder: returns cost or None if impassable."""
        terrain = self.map.cell_map[cell_id].terrain
        if terrain == C.TERRAIN_LAKE:
            return None
        return self._step_cost(cell_id, traffic)

    # ------------------------------------------------------------------ #
    # Reward                                                               #
    # ------------------------------------------------------------------ #

    def _reward(
        self,
        new_series: Set[int],
        daily_series: Set[int],
        udon: int,
        fuel_depleted: int,
        steps_used: int,
        steps_budget: int,
    ) -> float:
        r  = C.RW_NEW_SERIES   * len(new_series)
        r += C.RW_DAILY_SERIES * len(daily_series)
        r += C.RW_UDON         * udon
        r += C.RW_FUEL_EMPTY   * fuel_depleted
        wasted = max(0, steps_budget - steps_used)
        r += C.RW_WASTED_STEP  * wasted
        return r
