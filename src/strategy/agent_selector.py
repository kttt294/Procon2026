"""
Agent type selector — pre-match decision.

Given N starting positions (fixed by BTC), decides:
  1. How many patrol cars vs supply cars (split ratio)
  2. Which starting position gets which type

Approach: brute-force simulate candidate assignments on the actual match map,
score by avg unique_series over N_SIM quick games, pick the best.

Feasibility: N ≤ 8 agents → at most C(8,4)=70 position assignments × 7 splits
= ~500 combos × N_SIM=20 games = ~10k simulations.
Each simulation is fast (Lookahead on random rollout ≈ milliseconds).
Total runtime < 60s → well within pre-match preparation time.
"""
from __future__ import annotations

import itertools
import random
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import config as C
from env.hex_grid import HexGrid
from env.models import AgentState, Cell, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator
from strategy.lookahead import LookaheadPlanner


# How many simulated games to evaluate each candidate assignment.
# Higher = more accurate but slower. 20 is a good trade-off.
N_SIM = 20


@dataclass
class AgentAssignment:
    """Result of the selector: type for each agent ID."""
    types: Dict[int, int]          # agent_id -> AGENT_PATROL or AGENT_SUPPLY
    n_patrol: int
    n_supply: int
    score: float                   # avg unique_series across N_SIM games

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
            print(f"[selector] Evaluating {len(candidates)} candidate assignments "
                  f"({N_SIM} sims each)...")

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
                      f"→ score={score:.2f}")

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
          - At least 1 supply if n >= 3 (otherwise patrol will always run dry).
          - Skip duplicates caused by identical starting positions.
        """
        candidates = []
        seen_position_splits = set()

        for n_patrol in range(1, n):
            n_supply = n - n_patrol
            if n_supply < 1 and n >= 3:
                continue

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
        max_dist = max(self.grid.hex_distance(cell, other) for other in others)
        return 1.0 / (max_dist + 1)

    # ------------------------------------------------------------------ #
    # Simulation                                                           #
    # ------------------------------------------------------------------ #

    def _simulate_score(self, agents: List[AgentState], fuel_max: int) -> float:
        """
        Run N_SIM quick games with the Lookahead planner.
        Returns average unique_series collected.
        """
        if self.cfg.fuel_max is None:
            self.cfg.fuel_max = fuel_max

        total = 0.0
        for _ in range(N_SIM):
            # Randomise map slightly each sim to get a robust estimate
            state = self.sim.reset(_jitter_agents(agents))
            while not self.sim.is_done(state):
                orders = self._planner.plan(state)
                state, _ = self.sim.apply_day(state, orders)
            total += len(state.collected_series)

        return total / N_SIM


# ------------------------------------------------------------------ #
# Helper                                                               #
# ------------------------------------------------------------------ #

def _jitter_agents(agents: List[AgentState]) -> List[AgentState]:
    """Return a shallow copy with fuel slightly randomised (±10%) for robustness."""
    from copy import deepcopy
    result = deepcopy(agents)
    for a in result:
        if a.is_patrol() and a.fuel > 0:
            jitter = random.uniform(0.9, 1.0)
            a.fuel = max(1, int(a.fuel * jitter))
    return result
