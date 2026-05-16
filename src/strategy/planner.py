"""
Abstract interface for strategic planners.

A planner takes the current DayState and returns DayOrders for all agents.
Different implementations (greedy, RL, hybrid) conform to this interface.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from env.models import DayOrder, DayState, MapData, MatchConfig
from env.hex_grid import HexGrid
from env.simulator import HexaUdonSimulator


class BasePlanner(ABC):
    def __init__(self, cfg: MatchConfig, map_data: MapData, sim: HexaUdonSimulator):
        self.cfg  = cfg
        self.map  = map_data
        self.grid = sim.grid
        self.sim  = sim

    @abstractmethod
    def plan(self, state: DayState) -> List[DayOrder]:
        """
        Given the current day state, return orders for all agents.
        Must complete within state.time_limit_ms (enforced externally).
        """
        ...

    def terrain_dict(self) -> dict:
        return {c.id: c.terrain for c in self.map.cells}
