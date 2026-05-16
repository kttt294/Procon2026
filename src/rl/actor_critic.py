"""
Actor-Critic network for HEXA UDON.

Architecture:
  Map encoder : CNN over 2D hex grid feature planes → spatial embedding
  Agent encoder: MLP over (agent_fuel, agent_type, spatial_embed_at_agent_cell)
  Global encoder: concat(all_agent_features, collected_mask, day_info) → MLP
  Actor head : softmax over (n_spots + 1) choices per agent
                 (last = STAY / no target for this agent)
  Critic head: scalar V(state)

Input channels per cell (C_in = 9):
  0  terrain_plain    (binary)
  1  terrain_mountain (binary)
  2  terrain_road     (binary)
  3  traffic_clear    (binary, road only)
  4  traffic_busy     (binary, road only)
  5  traffic_congested(binary, road only)
  6  has_spot         (binary)
  7  series_collected (binary, 1 if spot here belongs to already-collected series)
  8  spot_inventory_norm (float 0..1)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from env.models import DayState, MapData, MatchConfig
from env.hex_grid import HexGrid


C_IN = 9   # feature channels per cell

# Fuel features fed to the agent MLP.
# We use relative/categorical features instead of raw fuel/fuel_max to avoid
# sensitivity to fuel_max, which BTC has not yet published.
#
# Features (4 values):
#   [0] can_reach_nearest_spot   : binary — can agent reach ≥1 spot with current fuel?
#   [1] fuel_tier_low            : binary — fuel ≤ 25% of observed session max
#   [2] fuel_tier_mid            : binary — fuel in (25%, 75%] of observed session max
#   [3] fuel_tier_high           : binary — fuel > 75% of observed session max
#
# "observed session max" = highest fuel seen for any patrol in day 1 of this episode.
# This is self-calibrating: once day-1 state is seen, tier thresholds auto-adjust.
FUEL_FEAT_DIM = 4


class MapEncoder(nn.Module):
    """CNN that encodes the 2D hex grid into per-cell embeddings."""

    def __init__(self, hidden: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C_IN, hidden // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden // 2, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_IN, H, W) → out: (B, hidden, H, W)
        return self.conv(x)


class ActorCritic(nn.Module):
    """
    Centralized Critic, per-agent Actor (shared weights across agents).

    n_spots: number of spot choices (action 0..n_spots-1 = target spot index,
             action n_spots = STAY/no-op for this agent).
    """

    def __init__(
        self,
        cfg: MatchConfig,
        map_data: MapData,
        hidden: int = C.HIDDEN_DIM,
    ):
        super().__init__()
        self.cfg      = cfg
        self.map      = map_data
        self.hidden   = hidden
        self.n_spots  = len(map_data.spots)
        self.n_series = map_data.n_series
        self.grid     = HexGrid(cfg.width, cfg.height)
        # _fuel_max: used only to compute tier thresholds for fuel features.
        # Self-calibrates each episode from the max patrol fuel seen on day 1.
        # Falls back to 20 until first episode data arrives.
        self._fuel_max: int = cfg.fuel_max if cfg.fuel_max else 20

        self.map_encoder = MapEncoder(hidden)

        # Agent MLP: spatial_embed + fuel_features (FUEL_FEAT_DIM) + agent_type (1)
        # Using categorical fuel features instead of raw fuel/fuel_max ratio so
        # the model is robust to unknown fuel_max (see FUEL_FEAT_DIM comment above).
        self.agent_mlp = nn.Sequential(
            nn.Linear(hidden + FUEL_FEAT_DIM + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )

        # Global MLP: concat all per-agent features + collected_mask + day_info
        # We handle variable n_agents by pooling agent features.
        # global_in = hidden//2 (mean agent pool) + n_series + 3 (day, steps, days_left)
        global_in = hidden // 2 + self.n_series + 3
        self.global_mlp = nn.Sequential(
            nn.Linear(global_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Actor head: for each agent, score each spot + STAY
        self.actor_head = nn.Linear(hidden // 2 + hidden, self.n_spots + 1)

    def set_fuel_max(self, fuel_max: int) -> None:
        self._fuel_max = fuel_max

    def _fuel_features(self, agent, state: DayState) -> torch.Tensor:
        """
        Categorical fuel features that don't depend on knowing fuel_max exactly.

        For supply cars, returns all zeros (fuel is irrelevant).
        For patrol cars:
          [0] can_reach_any_spot : 1 if fuel ≥ min fuel cost to move at all
          [1] tier_low           : fuel ≤ 25% of _fuel_max
          [2] tier_mid           : 25% < fuel ≤ 75% of _fuel_max
          [3] tier_high          : fuel > 75% of _fuel_max
        """
        feats = torch.zeros(FUEL_FEAT_DIM)
        if not agent.is_patrol():
            return feats

        fuel = agent.fuel
        fm   = max(self._fuel_max, 1)

        # tier thresholds are relative → robust even if _fuel_max is off
        lo = fm * 0.25
        hi = fm * 0.75
        feats[0] = 1.0 if fuel >= 1 else 0.0   # can move at all
        feats[1] = 1.0 if fuel <= lo else 0.0
        feats[2] = 1.0 if lo < fuel <= hi else 0.0
        feats[3] = 1.0 if fuel > hi else 0.0
        return feats

        # Critic head
        self.critic_head = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------ #
    # Encoding                                                             #
    # ------------------------------------------------------------------ #

    def encode_map(self, state: DayState) -> torch.Tensor:
        """Build (1, C_IN, H, W) feature tensor from current state."""
        H, W = self.cfg.height, self.cfg.width
        feat = torch.zeros(1, C_IN, H, W)

        for cell in self.map.cells:
            r, c = divmod(cell.id, W)
            t = cell.terrain
            if t == C.TERRAIN_PLAIN:
                feat[0, 0, r, c] = 1.0
            elif t == C.TERRAIN_MOUNTAIN:
                feat[0, 1, r, c] = 1.0
            elif t == C.TERRAIN_ROAD:
                feat[0, 2, r, c] = 1.0
                status = state.traffic.get(cell.id, C.TRAFFIC_CLEAR)
                feat[0, 3 + status, r, c] = 1.0

        for spot in self.map.spots:
            r, c = divmod(spot.cell_id, W)
            feat[0, 6, r, c] = 1.0
            if spot.series_id in state.collected_series:
                feat[0, 7, r, c] = 1.0
            inv = state.spot_inventory.get(spot.cell_id, 0)
            feat[0, 8, r, c] = inv / max(spot.max_inventory, 1)

        return feat

    def forward(
        self,
        state: DayState,
        agent_indices: List[int],   # which agents to compute actor for
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns:
          logits_list: list of (1, n_spots+1) logit tensors, one per agent in agent_indices
          value:       (1, 1) scalar
        """
        H, W = self.cfg.height, self.cfg.width

        # --- map encoding ---
        map_feat = self.encode_map(state)            # (1, C_IN, H, W)
        spatial  = self.map_encoder(map_feat)         # (1, hidden, H, W)

        agents_by_id = state.agents_by_id()

        # --- per-agent encoding ---
        agent_feats: List[torch.Tensor] = []
        for agent in state.my_agents:
            r, c = divmod(agent.cell, W)
            cell_embed  = spatial[0, :, r, c]                          # (hidden,)
            fuel_feats  = self._fuel_features(agent, state)            # (FUEL_FEAT_DIM,)
            atype       = torch.tensor([float(agent.type)])
            x = torch.cat([cell_embed, fuel_feats, atype])             # (hidden+FUEL_FEAT_DIM+1,)
            af = self.agent_mlp(x.unsqueeze(0))                        # (1, hidden//2)
            agent_feats.append(af)

        # Pool agent features (mean) for global context
        pooled = torch.stack(agent_feats).mean(0)             # (1, hidden//2)

        # --- global context ---
        collected_vec = torch.zeros(self.n_series)
        for i, sid in enumerate(self.map.series_ids):
            if sid in state.collected_series:
                collected_vec[i] = 1.0

        day_info = torch.tensor([
            state.day / self.cfg.total_days,
            state.steps_left / max(self.cfg.steps_per_day),
            (self.cfg.total_days - state.day) / self.cfg.total_days,
        ])
        global_in = torch.cat([pooled.squeeze(0), collected_vec, day_info])
        global_feat = self.global_mlp(global_in.unsqueeze(0))  # (1, hidden)

        # --- actor: one set of logits per requested agent ---
        logits_list: List[torch.Tensor] = []
        for aid in agent_indices:
            # Find agent feature
            idx = next(i for i, a in enumerate(state.my_agents) if a.id == aid)
            af  = agent_feats[idx]                              # (1, hidden//2)
            combined = torch.cat([af, global_feat], dim=-1)    # (1, hidden//2+hidden)
            logits = self.actor_head(combined)                  # (1, n_spots+1)
            logits_list.append(logits)

        # --- critic ---
        value = self.critic_head(global_feat)                   # (1, 1)

        return logits_list, value

    def get_action_and_value(
        self,
        state: DayState,
        agent_indices: List[int],
        deterministic: bool = False,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample or argmax actions. Returns:
          actions     : list of int (spot index or n_spots=STAY) per agent
          log_probs   : (n_agents,) tensor
          entropy     : (n_agents,) tensor
          value       : scalar tensor
        """
        logits_list, value = self.forward(state, agent_indices)

        actions: List[int] = []
        log_probs_list: List[torch.Tensor] = []
        entropy_list:   List[torch.Tensor] = []

        for logits in logits_list:
            dist = torch.distributions.Categorical(logits=logits.squeeze(0))
            a    = dist.mode if deterministic else dist.sample()
            actions.append(a.item())
            log_probs_list.append(dist.log_prob(a))
            entropy_list.append(dist.entropy())

        return (
            actions,
            torch.stack(log_probs_list),
            torch.stack(entropy_list),
            value.squeeze(),
        )
