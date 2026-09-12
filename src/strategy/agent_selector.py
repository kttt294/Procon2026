"""
Agent type selector — pre-match decision.

Given N starting positions (fixed by BTC), decides:
  1. How many patrol cars vs supply cars (split ratio)
  2. Which starting position gets which type

Approach: evaluate candidate assignments on the actual match map with full
initial fuel and compare the three contest score criteria lexicographically.
The rollout still has no refuelling until BTC publishes its mechanics.
"""
from __future__ import annotations

import itertools
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import config as C
from env.hex_grid import HexGrid
from env.models import AgentState, Cell, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator
from env.scoring import Score, compute_score
from strategy.lookahead import LookaheadPlanner


@dataclass
class AgentAssignment:
    """Result of the selector: type for each agent ID."""
    types: Dict[int, int]          # agent_id -> AGENT_PATROL or AGENT_SUPPLY
    n_patrol: int
    n_supply: int
    score: Score                   # contest ordering, including both score tie-breaks

    def as_agent_states(
        self,
        agent_ids: List[int],
        start_cells: List[int],
        fuel_max: int,
    ) -> List[AgentState]:
        return [
            AgentState(
                id=aid,
                type=self.types[aid],
                cell=cell,
                fuel=fuel_max if self.types[aid] == C.AGENT_PATROL else 0,
            )
            for aid, cell in zip(agent_ids, start_cells)
        ]


class AgentSelector:
    """
    Selects optimal agent types for a given match config and map.

    Usage:
        selector  = AgentSelector(cfg, map_data, sim)
        result    = selector.select(agent_ids, start_cells, fuel_max=20)
        # result.types → {agent_id: type}
    """

    def __init__(self, cfg: MatchConfig, map_data: MapData, sim: HexaUdonSimulator):
        self.cfg      = cfg
        self.map      = map_data
        self.sim      = sim
        self.grid     = HexGrid(cfg.width, cfg.height)
        self._planner = LookaheadPlanner(cfg, map_data, sim)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def select(
        self,
        agent_ids:   List[int],
        start_cells: List[int],
        fuel_max:    int = 20,
        verbose:     bool = False,
    ) -> AgentAssignment:
        """
        Evaluate all feasible (split, assignment) combos and return the best.

        agent_ids   : list of agent IDs assigned by BTC (order matches start_cells)
        start_cells : list of starting cell IDs, one per agent
        fuel_max    : patrol car fuel capacity (use known value or best estimate)
        """
        n = len(agent_ids)
        candidates = self._generate_candidates(agent_ids, start_cells, n)

        if verbose:
            print(f"[selector] Evaluating {len(candidates)} candidate assignments...")

        best: Optional[AgentAssignment] = None

        for types_map, n_patrol, n_supply in candidates:
            agents = [
                AgentState(
                    id=aid,
                    type=types_map[aid],
                    cell=cell,
                    fuel=fuel_max if types_map[aid] == C.AGENT_PATROL else 0,
                )
                for aid, cell in zip(agent_ids, start_cells)
            ]

            score = self._simulate_score(agents, fuel_max)

            if verbose:
                print(f"  patrol={n_patrol} supply={n_supply} "
                      f"positions={[start_cells[i] for i,a in enumerate(agent_ids) if types_map[a]==C.AGENT_PATROL]} "
                      f"→ {score}")

            if best is None or score > best.score:
                best = AgentAssignment(
                    types=types_map,
                    n_patrol=n_patrol,
                    n_supply=n_supply,
                    score=score,
                )

        assert best is not None
        return best

    # ------------------------------------------------------------------ #
    # Candidate generation                                                 #
    # ------------------------------------------------------------------ #

    def _generate_candidates(
        self,
        agent_ids:   List[int],
        start_cells: List[int],
        n:           int,
    ) -> List[Tuple[Dict[int, int], int, int]]:
        """
        Generate all (type_assignment, n_patrol, n_supply) combos worth trying.

        Rules:
          - At least 1 patrol (otherwise no udon collection).
          - All-patrol is allowed; the published rules do not require a supply.
          - Skip duplicates caused by identical starting positions.
        """
        candidates = []
        seen_position_splits = set()

        for n_patrol in range(1, n + 1):
            n_supply = n - n_patrol

            # Score each position for suitability as patrol vs supply
            patrol_scores  = [self._patrol_score(cell, start_cells) for cell in start_cells]
            supply_scores  = [self._supply_score(cell, start_cells) for cell in start_cells]

            # Try all C(n, n_patrol) position assignments, but prune by score
            all_combos = list(itertools.combinations(range(n), n_patrol))

            # Keep only top-K combos by combined score to limit runtime
            K = min(len(all_combos), 10)
            scored = sorted(
                all_combos,
                key=lambda combo: (
                    sum(patrol_scores[i] for i in combo)
                    + sum(supply_scores[i] for i in range(n) if i not in combo)
                ),
                reverse=True,
            )[:K]

            for combo in scored:
                patrol_indices = set(combo)
                key = (n_patrol, tuple(sorted(start_cells[i] for i in patrol_indices)))
                if key in seen_position_splits:
                    continue
                seen_position_splits.add(key)

                types_map = {
                    agent_ids[i]: (C.AGENT_PATROL if i in patrol_indices else C.AGENT_SUPPLY)
                    for i in range(n)
                }
                candidates.append((types_map, n_patrol, n_supply))

        return candidates

    # ------------------------------------------------------------------ #
    # Position scoring heuristics                                          #
    # ------------------------------------------------------------------ #

    def _patrol_score(self, cell: int, all_starts: List[int]) -> float:
        """
        Higher = better starting position for a patrol car.
        Favors positions close to many high-value spots (uncollected series variety).
        """
        score = 0.0
        series_seen = set()
        for spot in self.map.spots:
            dist = self.grid.hex_distance(cell, spot.cell_id)
            # Bonus for first spot of a new series (diversity value)
            series_bonus = 2.0 if spot.series_id not in series_seen else 1.0
            series_seen.add(spot.series_id)
            score += series_bonus / (dist + 1)
        return score

    def _supply_score(self, cell: int, all_starts: List[int]) -> float:
        """
        Higher = better starting position for a supply car.
        Favors positions central among all other starting positions
        (minimises worst-case travel time to any patrol car).
        """
        if len(all_starts) <= 1:
            return 0.0
        others = [c for c in all_starts if c != cell]
        if not others:
            return 0.0
        max_dist = max(self.grid.hex_distance(cell, other) for other in others)
        return 1.0 / (max_dist + 1)

    # ------------------------------------------------------------------ #
    # Simulation                                                           #
    # ------------------------------------------------------------------ #

    def _simulate_score(self, agents: List[AgentState], fuel_max: int) -> Score:
        """
        Evaluate the actual initial fuel and compare all contest score fields.
        The simulator is deterministic, so repeated identical rollouts add no evidence.
        """
        if self.cfg.fuel_max is None:
            self.cfg.fuel_max = fuel_max

        state = self.sim.reset(agents)
        while not self.sim.is_done(state):
            orders = self._planner.plan(state)
            state, _ = self.sim.apply_day(state, orders)
        return compute_score(state)
