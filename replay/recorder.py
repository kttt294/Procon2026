"""
ReplayRecorder: captures state + orders each day for post-game analysis.

Usage:
    rec = ReplayRecorder()
    while not sim.is_done(state):
        orders = planner.plan(state)
        rec.record(state, orders)          # capture before applying
        state, _ = sim.apply_day(state, orders)
    rec.record_final(state)                # capture final state
    rec.save("replays/game_001.json")

Then load with ReplayPlayer to step through the game.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from env.models import DayOrder, DayState


class ReplayRecorder:
    def __init__(self):
        self._frames: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Recording                                                            #
    # ------------------------------------------------------------------ #

    def record(self, state: DayState, orders: List[DayOrder]) -> None:
        """Capture state at the start of a day, plus the orders being submitted."""
        self._frames.append({
            "day":              state.day,
            "steps_left":       state.steps_left,
            "traffic":          {str(k): v for k, v in state.traffic.items()},
            "spot_inventory":   {str(k): v for k, v in state.spot_inventory.items()},
            "agents": [
                {
                    "id":   a.id,
                    "type": a.type,
                    "cell": a.cell,
                    "fuel": a.fuel,
                }
                for a in state.my_agents
            ],
            "opponent_cells":   list(state.opponent_cells),
            "collected_series": sorted(state.collected_series),
            "daily_series":     [sorted(s) for s in state.daily_series],
            "total_udon":       state.total_udon,
            "orders":           [o.to_dict() for o in orders],
        })

    def record_final(self, state: DayState) -> None:
        """Capture the terminal state (after last apply_day, no orders)."""
        self._frames.append({
            "day":              state.day,
            "steps_left":       0,
            "traffic":          {},
            "spot_inventory":   {},
            "agents": [
                {
                    "id":   a.id,
                    "type": a.type,
                    "cell": a.cell,
                    "fuel": a.fuel,
                }
                for a in state.my_agents
            ],
            "opponent_cells":   [],
            "collected_series": sorted(state.collected_series),
            "daily_series":     [sorted(s) for s in state.daily_series],
            "total_udon":       state.total_udon,
            "orders":           [],
            "final":            True,
        })

    def clear(self) -> None:
        self._frames.clear()

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"frames": self._frames}, f, indent=2, ensure_ascii=False)

    def __len__(self) -> int:
        return len(self._frames)
