"""
Self-play pool for MAPPO training.

Maintains a rolling pool of old model checkpoints used as opponents.
On each game day, the sampled opponent model generates actions, which:
  1. Produces road step counts that influence the traffic model
  2. Updates opponent agent positions visible in DayState.opponent_cells

This forces the RL agent to learn how to handle traffic caused by opponents
and to reason about opponent positions when planning routes.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import random
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

import torch

import config as C
from env.hex_grid import HexGrid
from env.models import AgentState, DayState, MapData, MatchConfig
from pathfinding.astar import find_path


class SelfPlayPool:
    """
    Rolling pool of old ActorCritic checkpoints for opponent simulation.

    Usage in training loop:
        pool = SelfPlayPool()
        ...
        # Each episode: update pool if threshold reached
        pool.step(current_model)

        # Each day: if pool is non-empty, simulate opponent
        if pool.has_opponent():
            opp_model = pool.sample()
            new_opp_cells, road_steps = SelfPlayPool.simulate_day(
                opp_model, opp_agents, state, map_data, cfg, grid
            )
    """

    def __init__(self, pool_size: int = 5, update_every: int = 1000):
        self.pool_size    = pool_size
        self.update_every = update_every
        self._pool:     List = []   # stored as CPU models
        self._episodes: int  = 0

    def step(self, model) -> bool:
        """Call once per episode. Returns True when a new checkpoint is added."""
        self._episodes += 1
        if self._episodes % self.update_every == 0:
            self._add(model)
            return True
        return False

    def _add(self, model) -> None:
        snapshot = copy.deepcopy(model).cpu()
        snapshot.eval()
        if len(self._pool) >= self.pool_size:
            self._pool.pop(0)
        self._pool.append(snapshot)
        print(f"[selfplay] Pool: {len(self._pool)}/{self.pool_size} checkpoints @ ep {self._episodes}")

    def has_opponent(self) -> bool:
        return len(self._pool) > 0

    def sample(self):
        """Return a random checkpoint from the pool."""
        return random.choice(self._pool)

    # ------------------------------------------------------------------ #
    # Opponent state construction                                          #
    # ------------------------------------------------------------------ #

    @staticmethod
    def build_opponent_view(state: DayState, opp_agents: List[AgentState]) -> DayState:
        """
        Construct a DayState from the opponent's perspective:
          - my_agents      = opponent agents (so the model sees them as "mine")
          - opponent_cells = our agents' cells
        Traffic and spot inventory are shared (same physical map).
        """
        our_cells = [a.cell for a in state.my_agents]
        return replace(state, my_agents=opp_agents, opponent_cells=our_cells)

    # ------------------------------------------------------------------ #
    # Opponent simulation                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def simulate_day(
        opp_model,
        opp_agents:  List[AgentState],
        state:       DayState,
        map_data:    MapData,
        cfg:         MatchConfig,
        grid:        HexGrid,
    ) -> Tuple[List[int], Dict[int, float]]:
        """
        Run the opponent model for one day and return:
          new_opp_cells : updated cell IDs for all opponent agents
          road_steps    : {cell_id: steps} of opponent road usage (for traffic)
        """
        opp_view     = SelfPlayPool.build_opponent_view(state, opp_agents)
        patrol_ids   = [a.id for a in opp_agents if a.type == C.AGENT_PATROL]
        terrain      = {c.id: c.terrain for c in map_data.cells}
        agents_by_id = {a.id: a for a in opp_agents}
        n_spots      = len(map_data.spots)

        with torch.no_grad():
            actions, _, _, _ = opp_model.get_action_and_value(
                opp_view, map_data, cfg, patrol_ids, deterministic=True
            )

        road_steps: Dict[int, float] = {}
        new_cells:  Dict[int, int]   = {a.id: a.cell for a in opp_agents}
        steps_left = state.steps_left

        for aid, act in zip(patrol_ids, actions):
            agent = agents_by_id[aid]
            if act >= n_spots:
                continue    # STAY

            target = map_data.spots[act].cell_id
            result = find_path(
                grid, terrain, state.traffic,
                agent.cell, target,
                step_budget=steps_left,
                fuel_budget=agent.fuel,
            )
            if not result.reachable:
                continue

            cur = agent.cell
            for action in result.actions:
                if terrain.get(cur) == C.TERRAIN_ROAD:
                    cost = C.STEP_COST[C.TERRAIN_ROAD][state.traffic.get(cur, C.TRAFFIC_CLEAR)]
                    road_steps[cur] = road_steps.get(cur, 0.0) + cost
                nxt = grid.neighbor_in_dir(cur, action.direction)
                if nxt is not None:
                    cur = nxt
            new_cells[aid] = cur
            agent.fuel -= result.total_fuel
            steps_left -= result.total_steps

        return list(new_cells.values()), road_steps

    # ------------------------------------------------------------------ #
    # Opponent agent generation                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def make_opponent_agents(
        map_data:  MapData,
        cfg:       MatchConfig,
        n_agents:  int,
        n_patrol:  int,
        our_cells: List[int],
        seed:      int,
    ) -> List[AgentState]:
        """
        Place N opponent agents on the map, avoiding cells used by our team.
        Opponent agent IDs start at 1001 to avoid collisions with our IDs.
        """
        occupied   = set(our_cells)
        candidates = [
            c.id for c in map_data.cells
            if c.terrain == C.TERRAIN_PLAIN and c.id not in occupied
            and c.id not in map_data.spot_map
        ]
        rng      = random.Random(seed)
        n_place  = min(n_agents, len(candidates))
        cells    = rng.sample(candidates, n_place)
        fuel_max = cfg.fuel_max or 20

        agents = []
        for i, cell in enumerate(cells):
            atype = C.AGENT_PATROL if i < n_patrol else C.AGENT_SUPPLY
            fuel  = fuel_max if atype == C.AGENT_PATROL else 0
            agents.append(AgentState(id=1001 + i, type=atype, cell=cell, fuel=fuel))
        return agents

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {"pool_len": len(self._pool), "episodes": self._episodes}
