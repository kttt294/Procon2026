"""
Curriculum learning engine for MAPPO training.

5 difficulty levels escalating from 8x8 maps to 32x32.
Level advances when RL win rate vs LookaheadPlanner > ADVANCE_THRESHOLD
over a rolling window of recent episodes.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from collections import deque
from dataclasses import dataclass
from typing import List, Tuple

from env.map_generator import MapGenConfig, MatchGenConfig, generate_scenario
from env.models import AgentState, MapData, MatchConfig
from env.simulator import HexaUdonSimulator
from strategy.lookahead import LookaheadPlanner


# ------------------------------------------------------------------ #
# Level definitions                                                    #
# ------------------------------------------------------------------ #

@dataclass(frozen=True)
class CurriculumLevel:
    level:      int
    width:      int
    height:     int
    total_days: int
    n_agents:   int
    n_patrol:   int
    n_series:   int
    n_spots:    int


LEVELS: List[CurriculumLevel] = [
    CurriculumLevel(1, width=8,  height=8,  total_days=4,  n_agents=3, n_patrol=2, n_series=3, n_spots=6),
    CurriculumLevel(2, width=12, height=12, total_days=5,  n_agents=4, n_patrol=3, n_series=4, n_spots=9),
    CurriculumLevel(3, width=16, height=16, total_days=6,  n_agents=5, n_patrol=3, n_series=5, n_spots=12),
    CurriculumLevel(4, width=24, height=24, total_days=8,  n_agents=6, n_patrol=4, n_series=6, n_spots=15),
    CurriculumLevel(5, width=32, height=32, total_days=10, n_agents=8, n_patrol=5, n_series=8, n_spots=20),
]

ADVANCE_THRESHOLD = 0.60   # RL must beat Lookahead ≥ 60 % of recent episodes
WIN_WINDOW        = 100    # rolling window size
MIN_SAMPLES       = 30     # minimum episodes before advancing is possible


# ------------------------------------------------------------------ #
# Engine                                                               #
# ------------------------------------------------------------------ #

class CurriculumEngine:
    """
    Tracks per-level win rate and promotes to the next level when ready.

    Typical usage inside MAPPOTrainer:
        curriculum = CurriculumEngine()
        for ep in range(n_episodes):
            cfg, map_data, agents = curriculum.generate_scenario(seed=seed + ep)
            ...run episode...
            baseline = curriculum.evaluate_baseline(cfg, map_data, agents)
            curriculum.record(rl_series, baseline)
            curriculum.try_advance()
    """

    def __init__(self, start_level: int = 0):
        self._idx    = max(0, min(start_level, len(LEVELS) - 1))
        self._wins:  deque = deque(maxlen=WIN_WINDOW)
        self._total: int   = 0

    # ------------------------------------------------------------------ #
    # Properties                                                           #
    # ------------------------------------------------------------------ #

    @property
    def level(self) -> CurriculumLevel:
        return LEVELS[self._idx]

    @property
    def level_number(self) -> int:
        return self._idx + 1

    @property
    def win_rate(self) -> float:
        return sum(self._wins) / len(self._wins) if self._wins else 0.0

    @property
    def at_max(self) -> bool:
        return self._idx >= len(LEVELS) - 1

    # ------------------------------------------------------------------ #
    # Scenario generation                                                  #
    # ------------------------------------------------------------------ #

    def generate_scenario(
        self, seed: int
    ) -> Tuple[MatchConfig, MapData, List[AgentState]]:
        lv = self.level
        return generate_scenario(
            seed      = seed,
            map_cfg   = MapGenConfig(
                width    = lv.width,
                height   = lv.height,
                n_series = lv.n_series,
                n_spots  = lv.n_spots,
            ),
            match_cfg = MatchGenConfig(total_days=lv.total_days),
            n_agents  = lv.n_agents,
            n_patrol  = lv.n_patrol,
        )

    # ------------------------------------------------------------------ #
    # Baseline evaluation                                                  #
    # ------------------------------------------------------------------ #

    def evaluate_baseline(
        self,
        cfg:      MatchConfig,
        map_data: MapData,
        agents:   List[AgentState],
    ) -> int:
        """
        Run LookaheadPlanner on the same scenario and return unique_series.
        Used to compare against RL performance.
        """
        import copy
        sim     = HexaUdonSimulator(cfg, map_data)
        state   = sim.reset(copy.deepcopy(agents))
        planner = LookaheadPlanner(cfg, map_data, sim)
        while not sim.is_done(state):
            orders = planner.plan(state)
            state, _ = sim.apply_day(state, orders)
        return len(state.collected_series)

    # ------------------------------------------------------------------ #
    # Win tracking                                                         #
    # ------------------------------------------------------------------ #

    def record(self, rl_series: int, baseline_series: int) -> None:
        """Record whether RL matched or beat the Lookahead baseline."""
        self._wins.append(int(rl_series >= baseline_series))
        self._total += 1

    def try_advance(self) -> bool:
        """Advance to next level if criteria met. Returns True if advanced."""
        if self.at_max or len(self._wins) < MIN_SAMPLES:
            return False
        if self.win_rate >= ADVANCE_THRESHOLD:
            self._idx += 1
            self._wins.clear()
            lv = self.level
            print(
                f"[curriculum] Level {self.level_number} | "
                f"{lv.width}x{lv.height}, {lv.total_days} days, "
                f"{lv.n_agents} agents, {lv.n_series} series"
            )
            return True
        return False

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def state_dict(self) -> dict:
        return {"level_idx": self._idx, "total_episodes": self._total}

    def load_state_dict(self, d: dict) -> None:
        self._idx   = d.get("level_idx", 0)
        self._total = d.get("total_episodes", 0)
