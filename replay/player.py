"""
ReplayPlayer: load and iterate over a saved replay file.

Usage:
    player = ReplayPlayer("replays/game_001.json")
    for frame in player:
        print(f"Day {frame['day']}: agents={frame['agents']}")

    # Or access a specific day:
    frame = player.day(3)
    print(frame["orders"])

    # Summary:
    print(player.summary())
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterator, List


class ReplayPlayer:
    def __init__(self, path: str):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self._frames: List[Dict[str, Any]] = data["frames"]

    # ------------------------------------------------------------------ #
    # Access                                                               #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self._frames)

    def day(self, d: int) -> Dict[str, Any]:
        """Return the frame for day d (1-indexed). Raises IndexError if out of range."""
        matches = [f for f in self._frames if f["day"] == d]
        if not matches:
            raise IndexError(f"No frame for day {d}")
        return matches[0]

    def final(self) -> Dict[str, Any]:
        """Return the final state frame."""
        for f in reversed(self._frames):
            if f.get("final"):
                return f
        return self._frames[-1]

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #

    def summary(self) -> str:
        if not self._frames:
            return "Empty replay."
        last = self.final()
        n_days = sum(1 for f in self._frames if not f.get("final"))
        return (
            f"Replay: {n_days} days played | "
            f"series={last['collected_series']} | "
            f"udon={last['total_udon']}"
        )

    def agent_trace(self, agent_id: int) -> List[Dict[str, Any]]:
        """Return per-day position/fuel history for one agent."""
        trace = []
        for frame in self._frames:
            for a in frame["agents"]:
                if a["id"] == agent_id:
                    trace.append({
                        "day":  frame["day"],
                        "cell": a["cell"],
                        "fuel": a["fuel"],
                    })
                    break
        return trace
