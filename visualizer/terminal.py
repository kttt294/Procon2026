"""
Terminal ASCII visualizer for HEXA UDON.

Renders the game state as a compact grid in the terminal.
Useful for debugging strategy behavior without reading raw logs.

Usage:
    viz = TerminalVisualizer()
    viz.render(state, map_data, cfg)

Grid symbols (2 chars per cell):
    P1–P8  patrol agent (number = agent index)
    S1–S8  supply agent
    **     spot with remaining inventory
    ..     spot (inventory = 0)
    ##     lake
    ^^     mountain
    ==     road (clear)
    =B     road (busy)
    =C     road (congested)
    [sp]   plain (two spaces)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from typing import Dict, Optional

import config as C
from env.models import DayState, MapData, MatchConfig


_TRAFFIC_SUFFIX = {
    C.TRAFFIC_CLEAR:     "=",
    C.TRAFFIC_BUSY:      "B",
    C.TRAFFIC_CONGESTED: "C",
}

_FUEL_BAR_WIDTH = 10


class TerminalVisualizer:
    """Print a compact ASCII snapshot of one game state."""

    def render(
        self,
        state:    DayState,
        map_data: MapData,
        cfg:      MatchConfig,
    ) -> None:
        """Print the full state to stdout."""
        lines = self._build(state, map_data, cfg)
        print("\n".join(lines))

    def render_to_string(
        self,
        state:    DayState,
        map_data: MapData,
        cfg:      MatchConfig,
    ) -> str:
        """Return the rendered state as a single string (for logging/replay)."""
        return "\n".join(self._build(state, map_data, cfg))

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _build(self, state: DayState, map_data: MapData, cfg: MatchConfig):
        w, h = cfg.width, cfg.height

        # --- header ---
        total_series = map_data.n_series
        collected    = len(state.collected_series)
        series_bar   = "".join(
            "✓" if sid in state.collected_series else "."
            for sid in map_data.series_ids
        )
        lines = [
            f"Day {state.day - 1}/{cfg.total_days}  "    # day-1 because state.day is NEXT day
            f"Steps: {state.steps_left}/{cfg.steps_per_day[min(state.day - 2, len(cfg.steps_per_day) - 1)]}  "
            f"Series: [{series_bar}] {collected}/{total_series}  "
            f"Udon: {state.total_udon}",
        ]

        # Build cell lookup tables
        agent_at:    Dict[int, str] = {}    # cell_id → symbol
        spot_cells:  Dict[int, int] = {}    # cell_id → remaining inventory

        patrol_agents  = sorted(state.patrol_agents(),  key=lambda a: a.id)
        supply_agents  = sorted(state.supply_agents(), key=lambda a: a.id)

        for i, a in enumerate(patrol_agents):
            agent_at[a.cell] = f"P{i+1}"
        for i, a in enumerate(supply_agents):
            agent_at[a.cell] = f"S{i+1}"

        for cell_id, inv in state.spot_inventory.items():
            spot_cells[cell_id] = inv

        # --- column header ---
        col_nums = "     " + "".join(f"{c:<3}" for c in range(w))
        lines.append(col_nums)

        # --- grid rows ---
        for row in range(h):
            indent = "  " if (row % 2 == 0) else ""   # hex offset for odd rows
            row_str = f"{row:2d} {indent}"
            for col in range(w):
                cell_id = row * w + col
                sym = self._cell_symbol(
                    cell_id, map_data, state.traffic,
                    agent_at, spot_cells
                )
                row_str += sym + " "
            lines.append(row_str)

        # --- agents panel ---
        lines.append("")
        lines.append("Agents:")
        fuel_max = cfg.fuel_max or 20
        for i, a in enumerate(patrol_agents):
            bar  = _fuel_bar(a.fuel, fuel_max)
            lines.append(f"  P{i+1} id={a.id}  cell={a.cell:<4}  fuel {bar} {a.fuel}/{fuel_max}")
        for i, a in enumerate(supply_agents):
            lines.append(f"  S{i+1} id={a.id}  cell={a.cell:<4}  (supply)")

        # --- traffic panel ---
        if state.traffic:
            lines.append("")
            lines.append("Traffic:")
            for cell_id, status in sorted(state.traffic.items()):
                label = {C.TRAFFIC_BUSY: "BUSY", C.TRAFFIC_CONGESTED: "CONGESTED"}.get(status, "")
                if label:
                    lines.append(f"  cell {cell_id}: {label}")

        lines.append("")
        return lines

    def _cell_symbol(
        self,
        cell_id:   int,
        map_data:  MapData,
        traffic:   Dict[int, int],
        agent_at:  Dict[int, str],
        spot_cells: Dict[int, int],
    ) -> str:
        # Agent takes priority
        if cell_id in agent_at:
            return agent_at[cell_id]

        terrain = map_data.cell_map[cell_id].terrain

        if terrain == C.TERRAIN_LAKE:
            return "##"
        if terrain == C.TERRAIN_MOUNTAIN:
            return "^^"

        if cell_id in spot_cells:
            return "**" if spot_cells[cell_id] > 0 else ".."

        if terrain == C.TERRAIN_ROAD:
            status = traffic.get(cell_id, C.TRAFFIC_CLEAR)
            return "=" + _TRAFFIC_SUFFIX.get(status, "=")

        return "  "   # plain


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _fuel_bar(fuel: int, fuel_max: int) -> str:
    if fuel_max <= 0:
        return "[" + "?" * _FUEL_BAR_WIDTH + "]"
    filled = round(fuel / fuel_max * _FUEL_BAR_WIDTH)
    empty  = _FUEL_BAR_WIDTH - filled
    return "[" + "█" * filled + "░" * empty + "]"
