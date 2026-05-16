"""
Random map generator for HEXA UDON.

Generates reproducible random scenarios (map + config + agents) given a seed.

Usage:
    from env.map_generator import generate_scenario, generate_random_scenario

    # Fixed size:
    cfg, map_data, agents = generate_scenario(seed=42)

    # Fully random size/days/agents:
    cfg, map_data, agents = generate_random_scenario(seed=42)
"""
from __future__ import annotations

import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass
from typing import List, Optional, Tuple

import config as C
from env.models import AgentState, Cell, MapData, MatchConfig, Spot


# ------------------------------------------------------------------ #
# Config dataclasses                                                   #
# ------------------------------------------------------------------ #

@dataclass
class MapGenConfig:
    width:          int   = 12
    height:         int   = 12
    plain_ratio:    float = 0.55
    mountain_ratio: float = 0.10
    lake_ratio:     float = 0.10
    road_ratio:     float = 0.25
    n_series:       int   = 4
    n_spots:        int   = 8    # total spots across all series
    max_inventory:  int   = 3    # per-spot max (upper bound; actual is random 1..max)


@dataclass
class MatchGenConfig:
    total_days:     int   = 6
    steps_base:     int   = 120   # steps on day 1
    steps_decay:    float = 0.85  # multiply steps each successive day
    n_teams:        int   = 4
    thr_busy:       float = 3.0
    thr_congested:  float = 7.0
    fuel_max:       int   = 20


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def generate_scenario(
    seed:      int,
    map_cfg:   Optional[MapGenConfig]   = None,
    match_cfg: Optional[MatchGenConfig] = None,
    n_agents:  int = 3,
    n_patrol:  int = 2,
) -> Tuple[MatchConfig, MapData, List[AgentState]]:
    """
    Generate a reproducible scenario with given (or default) config.

    seed    : random seed for full reproducibility
    n_agents: total number of agents
    n_patrol: how many are patrol (rest are supply)
    """
    rng       = random.Random(seed)
    map_cfg   = map_cfg   or MapGenConfig()
    match_cfg = match_cfg or MatchGenConfig()

    map_data = _generate_map(rng, map_cfg)
    cfg      = _generate_config(map_cfg, match_cfg)
    agents   = _generate_agents(rng, map_data, cfg, n_agents, n_patrol)
    return cfg, map_data, agents


def generate_random_scenario(
    seed:          int,
    width_range:   Tuple[int, int] = (8,  24),
    height_range:  Tuple[int, int] = (8,  24),
    days_range:    Tuple[int, int] = (4,  8),
    agents_range:  Tuple[int, int] = (3,  6),
) -> Tuple[MatchConfig, MapData, List[AgentState]]:
    """
    Generate a fully random scenario — size, days, and agent count all randomised.
    Useful for training and benchmarking over diverse conditions.
    """
    rng = random.Random(seed)

    w        = rng.randrange(width_range[0],  width_range[1]  + 1, 2)
    h        = rng.randrange(height_range[0], height_range[1] + 1, 2)
    n_days   = rng.randint(*days_range)
    n_agents = rng.randint(*agents_range)
    n_patrol = max(1, rng.randint(1, max(1, n_agents - 1)))
    n_series = rng.randint(2, min(n_agents + 2, 8))
    n_spots  = rng.randint(n_series, n_series * 3)

    plain_r  = rng.uniform(0.40, 0.70)
    mtn_r    = rng.uniform(0.05, 0.15)
    lake_r   = rng.uniform(0.05, 0.15)
    road_r   = rng.uniform(0.10, 0.25)
    total    = plain_r + mtn_r + lake_r + road_r

    map_cfg = MapGenConfig(
        width          = w,
        height         = h,
        plain_ratio    = plain_r  / total,
        mountain_ratio = mtn_r   / total,
        lake_ratio     = lake_r  / total,
        road_ratio     = road_r  / total,
        n_series       = n_series,
        n_spots        = n_spots,
    )
    match_cfg = MatchGenConfig(
        total_days = n_days,
        fuel_max   = rng.randint(15, 30),
    )

    map_data = _generate_map(rng, map_cfg)
    cfg      = _generate_config(map_cfg, match_cfg)
    agents   = _generate_agents(rng, map_data, cfg, n_agents, n_patrol)
    return cfg, map_data, agents


# ------------------------------------------------------------------ #
# Internal builders                                                    #
# ------------------------------------------------------------------ #

def _generate_map(rng: random.Random, cfg: MapGenConfig) -> MapData:
    n = cfg.width * cfg.height
    terrain_types   = [C.TERRAIN_PLAIN, C.TERRAIN_MOUNTAIN, C.TERRAIN_LAKE, C.TERRAIN_ROAD]
    terrain_weights = [cfg.plain_ratio, cfg.mountain_ratio, cfg.lake_ratio, cfg.road_ratio]

    terrain = rng.choices(terrain_types, weights=terrain_weights, k=n)
    cells   = [Cell(id=i, terrain=terrain[i]) for i in range(n)]

    # Spots placed only on non-lake cells
    non_lake = [i for i in range(n) if terrain[i] != C.TERRAIN_LAKE]
    n_spots  = min(cfg.n_spots, len(non_lake))
    spot_cells = rng.sample(non_lake, n_spots)

    # Round-robin series assignment so each series appears at least once
    spots = []
    for i, cell_id in enumerate(spot_cells):
        series_id     = (i % cfg.n_series) + 1
        max_inv       = rng.randint(1, cfg.max_inventory)
        spots.append(Spot(cell_id=cell_id, series_id=series_id, max_inventory=max_inv))

    return MapData(cells=cells, spots=spots)


def _generate_config(map_cfg: MapGenConfig, match_cfg: MatchGenConfig) -> MatchConfig:
    steps = []
    s = float(match_cfg.steps_base)
    for _ in range(match_cfg.total_days):
        steps.append(max(40, int(s)))
        s *= match_cfg.steps_decay

    return MatchConfig(
        width                    = map_cfg.width,
        height                   = map_cfg.height,
        total_days               = match_cfg.total_days,
        steps_per_day            = steps,
        n_teams                  = match_cfg.n_teams,
        traffic_threshold_busy   = match_cfg.thr_busy,
        traffic_threshold_congested = match_cfg.thr_congested,
        fuel_max                 = match_cfg.fuel_max,
    )


def _generate_agents(
    rng:      random.Random,
    map_data: MapData,
    cfg:      MatchConfig,
    n_agents: int,
    n_patrol: int,
) -> List[AgentState]:
    spot_cells = {s.cell_id for s in map_data.spots}
    candidates = [
        c.id for c in map_data.cells
        if c.terrain != C.TERRAIN_LAKE and c.id not in spot_cells
    ]
    # Fallback: if not enough non-spot cells, allow spot cells
    if len(candidates) < n_agents:
        candidates = [c.id for c in map_data.cells if c.terrain != C.TERRAIN_LAKE]

    n_agents    = min(n_agents, len(candidates))
    start_cells = rng.sample(candidates, n_agents)
    fuel_max    = cfg.fuel_max or 20

    agents = []
    for i, cell in enumerate(start_cells):
        aid = i + 1
        if i < n_patrol:
            agents.append(AgentState(id=aid, type=C.AGENT_PATROL, cell=cell, fuel=fuel_max))
        else:
            agents.append(AgentState(id=aid, type=C.AGENT_SUPPLY, cell=cell, fuel=0))
    return agents
