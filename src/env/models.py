from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class MatchConfig:
    width:                      int
    height:                     int
    total_days:                 int
    steps_per_day:              List[int]
    n_teams:                    int
    traffic_threshold_busy:     float
    traffic_threshold_congested: float
    # fuel_max: BTC has not published this value yet (as of preparation time).
    # It will be known before the real match starts (in match config or announcement).
    # Equal for all patrol cars within a match; may differ across matches.
    # Set to None until BTC publishes or match config is received.
    fuel_max: Optional[int] = None

    @staticmethod
    def from_dict(d: dict) -> MatchConfig:
        cfg = d["match_config"]
        return MatchConfig(
            width=cfg["width"],
            height=cfg["height"],
            total_days=cfg["total_days"],
            steps_per_day=cfg["steps_per_day"],
            n_teams=cfg["n_teams"],
            traffic_threshold_busy=cfg["traffic_threshold_busy"],
            traffic_threshold_congested=cfg["traffic_threshold_congested"],
            fuel_max=cfg.get("fuel_max"),
        )

    def infer_fuel_max(self, day1_agents: List) -> None:
        """Fallback: infer fuel_max from day-1 patrol agents (all start with full tank)."""
        fuels = [a.fuel for a in day1_agents if a.is_patrol()]
        if fuels:
            self.fuel_max = max(fuels)


@dataclass
class Cell:
    id:      int
    terrain: int   # 0=plain, 1=mountain, 2=lake, 3=road


@dataclass
class Spot:
    cell_id:       int
    series_id:     int
    max_inventory: int


@dataclass
class MapData:
    cells: List[Cell]
    spots: List[Spot]

    def __post_init__(self):
        self.cell_map:  Dict[int, Cell] = {c.id: c for c in self.cells}
        self.spot_map:  Dict[int, Spot] = {s.cell_id: s for s in self.spots}
        self.n_series:  int             = len({s.series_id for s in self.spots})
        self.series_ids: List[int]      = sorted({s.series_id for s in self.spots})

    @staticmethod
    def from_dict(d: dict) -> MapData:
        cells = [Cell(id=c["id"], terrain=c["terrain"]) for c in d["map"]["cells"]]
        spots = [
            Spot(
                cell_id=s["cell_id"],
                series_id=s["series_id"],
                max_inventory=s["max_inventory"],
            )
            for s in d["map"]["spots"]
        ]
        return MapData(cells=cells, spots=spots)


@dataclass
class AgentState:
    id:   int
    type: int   # 0=patrol, 1=supply
    cell: int
    fuel: int = 0   # meaningful only for patrol cars

    def is_patrol(self) -> bool:
        return self.type == 0

    def is_supply(self) -> bool:
        return self.type == 1


@dataclass
class DayState:
    """Full state snapshot at the beginning of a day."""
    day:             int
    steps_left:      int
    time_limit_ms:   int
    traffic:         Dict[int, int]       # road cell_id -> {0,1,2}
    spot_inventory:  Dict[int, int]       # spot cell_id -> remaining
    my_agents:       List[AgentState]
    opponent_cells:  List[int]            # cell IDs of all opponent agents
    collected_series: Set[int]            # series IDs collected so far (this game)
    daily_series:    List[Set[int]]       # series collected per day (history)
    total_udon:      int

    # Internal: step counts our agents spent on each road cell today
    # Used to feed into traffic model.
    _road_step_counts: Dict[int, float] = field(default_factory=dict, repr=False)

    def agents_by_id(self) -> Dict[int, AgentState]:
        return {a.id: a for a in self.my_agents}

    def patrol_agents(self) -> List[AgentState]:
        return [a for a in self.my_agents if a.is_patrol()]

    def supply_agents(self) -> List[AgentState]:
        return [a for a in self.my_agents if a.is_supply()]


# Action types
CMD_MOVE = "move"
CMD_STAY = "stay"


@dataclass
class AgentAction:
    cmd:       str            # CMD_MOVE or CMD_STAY
    direction: Optional[int] = None   # 0-5, only for CMD_MOVE


@dataclass
class DayOrder:
    """Orders for one agent for one day."""
    agent_id: int
    actions:  List[AgentAction]

    def to_dict(self) -> dict:
        acts = []
        for a in self.actions:
            if a.cmd == CMD_MOVE:
                acts.append({"cmd": CMD_MOVE, "dir": a.direction})
            else:
                acts.append({"cmd": CMD_STAY})
        return {"agent_id": self.agent_id, "actions": acts}
