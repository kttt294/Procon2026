"""
Pure scoring functions for HEXA UDON.

Winning priority (contest rules):
  1. unique_series    — most distinct udon types ever collected
  2. daily_series_sum — sum of distinct series collected per day (ties broken here)
  3. total_udon       — raw udon bowl count
  4. submit_time      — earliest submission (not modeled here)

Extracted as pure functions so optimizers can evaluate plans without
running a full simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from env.models import DayState
import config as C


def collection_potential(state, map_data, grid) -> float:
    """Potential used by training; also converts the critic back to raw reward."""
    uncollected = [s for s in map_data.spots if s.series_id not in state.collected_series]
    patrols = state.patrol_agents()
    if not uncollected or not patrols:
        return 0.0
    distance = sum(min(grid.hex_distance(a.cell, s.cell_id) for s in uncollected)
                   for a in patrols)
    return -C.POTENTIAL_SCALE * distance / len(patrols)


@dataclass
class Score:
    unique_series:    int   # number of distinct series ever collected
    daily_series_sum: int   # sum(len(series_on_day_d) for all d)
    total_udon:       int   # total udon bowls

    def __gt__(self, other: Score) -> bool:
        return _cmp(self, other) > 0

    def __lt__(self, other: Score) -> bool:
        return _cmp(self, other) < 0

    def __ge__(self, other: Score) -> bool:
        return _cmp(self, other) >= 0

    def __le__(self, other: Score) -> bool:
        return _cmp(self, other) <= 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Score):
            return False
        return _cmp(self, other) == 0

    def __repr__(self) -> str:
        return (
            f"Score(series={self.unique_series}, "
            f"daily={self.daily_series_sum}, "
            f"udon={self.total_udon})"
        )


def compute_score(state: DayState) -> Score:
    """Extract the current score from a DayState (terminal or intermediate)."""
    return Score(
        unique_series    = len(state.collected_series),
        daily_series_sum = sum(len(s) for s in state.daily_series),
        total_udon       = state.total_udon,
    )


def wins_over(a: Score, b: Score) -> bool:
    """True if score a strictly beats score b by contest rules."""
    return _cmp(a, b) > 0


def _cmp(a: Score, b: Score) -> int:
    if a.unique_series != b.unique_series:
        return a.unique_series - b.unique_series
    if a.daily_series_sum != b.daily_series_sum:
        return a.daily_series_sum - b.daily_series_sum
    return a.total_udon - b.total_udon
