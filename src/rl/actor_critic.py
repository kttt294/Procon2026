"""
Actor-Critic network for HEXA UDON.

Architecture:
  Map encoder : CNN over 2D hex grid feature planes → spatial embedding
  Agent MLP   : per-agent (spatial_embed + fuel_features + type) → agent_feat
  Global MLP  : mean_pool(agent_feats) + collected_mask + day_info → global_feat
  Actor head  : (agent_feat + global_feat) → logits over (max_spots + 1) choices
  Critic head : global_feat → scalar V(state)

Variable-map support:
  Network dimensions are fixed to (max_spots, max_series, max_width, max_height).
  Maps smaller than the max are zero-padded. Invalid spot slots (index ≥ actual
  n_spots on this map) are masked to -inf before sampling so they're never chosen.
  The STAY action always lives at index max_spots (never masked).

Input channels per cell (C_IN = 10):
  0  terrain_plain      binary
  1  terrain_mountain   binary
  2  terrain_road       binary
  3  traffic_clear      binary (road only)
  4  traffic_busy       binary (road only)
  5  traffic_congested  binary (road only)
  6  has_spot           binary
  7  series_collected   binary (spot's series already in collected_series)
  8  spot_inventory_norm float 0..1
  9  opponent_present   binary (opponent agent at this cell)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from typing import List, Tuple

import torch
import torch.nn as nn

import config as C
from env.hex_grid import HexGrid
from env.models import DayState, MapData, MatchConfig


C_IN          = 10   # feature channels per cell (9 map + 1 opponent presence)
FUEL_FEAT_DIM = 4    # categorical fuel features (see _fuel_features)


class MapEncoder(nn.Module):
    """CNN: (B, C_IN, H, W) → (B, hidden, H, W). Padding=1 keeps spatial size."""

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
        return self.conv(x)


class ActorCritic(nn.Module):
    """
    Centralized Critic, per-agent Actor (weights shared across agents).

    Initialised with max dimensions so the same network trains across
    maps of different sizes and spot counts.

    Args:
        max_spots  : maximum number of spots any training map will have
        max_series : maximum number of series any training map will have
        max_width  : maximum map width
        max_height : maximum map height
        hidden     : hidden dimension for all MLPs
    """

    def __init__(
        self,
        max_spots:  int = 30,
        max_series: int = 10,
        max_width:  int = 32,
        max_height: int = 32,
        hidden:     int = C.HIDDEN_DIM,
    ):
        super().__init__()
        self.max_spots  = max_spots
        self.max_series = max_series
        self.max_width  = max_width
        self.max_height = max_height
        self.hidden     = hidden

        # Observed fuel_max for tier thresholds; updated each episode via set_fuel_max().
        self._fuel_max: int = 20

        self.map_encoder = MapEncoder(hidden)

        # Agent MLP: cell_embed(hidden) + fuel_feats(4) + agent_type(1)
        self.agent_mlp = nn.Sequential(
            nn.Linear(hidden + FUEL_FEAT_DIM + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )

        # Global MLP: mean_pool_agents(hidden//2) + collected_mask(max_series) + day_info(3)
        global_in = hidden // 2 + max_series + 3
        self.global_mlp = nn.Sequential(
            nn.Linear(global_in, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

        # Actor: outputs max_spots + 1 logits (slots 0..max_spots-1 = spots, max_spots = STAY)
        self.actor_head  = nn.Linear(hidden // 2 + hidden, max_spots + 1)

        # Critic: scalar value estimate
        self.critic_head = nn.Linear(hidden, 1)

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    @property
    def device(self) -> torch.device:
        """Return the device of the model parameters."""
        return next(self.parameters()).device

    def set_fuel_max(self, fuel_max: int) -> None:
        self._fuel_max = fuel_max

    def _fuel_features(self, agent, state: DayState) -> torch.Tensor:
        """
        4 categorical fuel features robust to unknown fuel_max.
          [0] can_move    : fuel ≥ 1 (can move at all)
          [1] tier_low    : fuel ≤ 25 % of _fuel_max
          [2] tier_mid    : 25 % < fuel ≤ 75 %
          [3] tier_high   : fuel > 75 %
        Supply cars return all zeros (fuel irrelevant for them).
        """
        feats = torch.zeros(FUEL_FEAT_DIM, device=self.device)
        if not agent.is_patrol():
            return feats
        fuel = agent.fuel
        fm   = max(self._fuel_max, 1)
        lo, hi = fm * 0.25, fm * 0.75
        feats[0] = 1.0 if fuel >= 1      else 0.0
        feats[1] = 1.0 if fuel <= lo     else 0.0
        feats[2] = 1.0 if lo < fuel <= hi else 0.0
        feats[3] = 1.0 if fuel > hi      else 0.0
        return feats

    # ------------------------------------------------------------------ #
    # Map encoding (variable size → padded to max)                        #
    # ------------------------------------------------------------------ #

    def encode_map(
        self,
        state:    DayState,
        map_data: MapData,
        cfg:      MatchConfig,
    ) -> torch.Tensor:
        """
        Build (1, C_IN, max_height, max_width) feature tensor.
        Cells beyond the actual map dimensions stay zero-padded.
        """
        feat = torch.zeros(1, C_IN, self.max_height, self.max_width, device=self.device)
        W = cfg.width

        for cell in map_data.cells:
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

        for spot in map_data.spots:
            r, c = divmod(spot.cell_id, W)
            feat[0, 6, r, c] = 1.0
            if spot.series_id in state.collected_series:
                feat[0, 7, r, c] = 1.0
            inv = state.spot_inventory.get(spot.cell_id, 0)
            feat[0, 8, r, c] = inv / max(spot.max_inventory, 1)

        # Channel 9: opponent agent presence
        for cell_id in state.opponent_cells:
            r, c = divmod(cell_id, W)
            if 0 <= r < self.max_height and 0 <= c < self.max_width:
                feat[0, 9, r, c] = 1.0

        return feat

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        state:         DayState,
        map_data:      MapData,
        cfg:           MatchConfig,
        agent_indices: List[int],
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Returns:
          logits_list : one (1, max_spots+1) tensor per agent in agent_indices
                        (invalid spots already masked to -inf)
          value       : (1,) scalar
        """
        W = cfg.width
        n_spots  = len(map_data.spots)
        series_ids = map_data.series_ids

        # --- map encoding ---
        map_feat = self.encode_map(state, map_data, cfg)   # (1, C_IN, maxH, maxW)
        spatial  = self.map_encoder(map_feat)               # (1, hidden, maxH, maxW)

        # --- per-agent encoding ---
        agent_feats: List[torch.Tensor] = []
        for agent in state.my_agents:
            r, c       = divmod(agent.cell, W)
            cell_embed = spatial[0, :, r, c]                          # (hidden,)
            fuel_feats = self._fuel_features(agent, state)            # (4,)
            atype      = torch.tensor([float(agent.type)], device=self.device)
            x  = torch.cat([cell_embed, fuel_feats, atype])           # (hidden+5,)
            af = self.agent_mlp(x.unsqueeze(0))                       # (1, hidden//2)
            agent_feats.append(af)

        pooled = torch.stack(agent_feats).mean(0)   # (1, hidden//2)

        # --- global context (padded to max_series) ---
        collected_vec = torch.zeros(self.max_series, device=self.device)
        for i, sid in enumerate(series_ids[:self.max_series]):
            if sid in state.collected_series:
                collected_vec[i] = 1.0

        day_info = torch.tensor([
            state.day / max(cfg.total_days, 1),
            state.steps_left / max(max(cfg.steps_per_day), 1),
            (cfg.total_days - state.day) / max(cfg.total_days, 1),
        ], device=self.device)
        global_in   = torch.cat([pooled.squeeze(0), collected_vec, day_info])
        global_feat = self.global_mlp(global_in.unsqueeze(0))   # (1, hidden)

        # --- actor logits with invalid-slot masking ---
        # Slots n_spots .. max_spots-1 are padding (not real spots) → mask to -inf
        # Slot max_spots is always STAY → never masked
        logits_list: List[torch.Tensor] = []
        for aid in agent_indices:
            idx      = next(i for i, a in enumerate(state.my_agents) if a.id == aid)
            af       = agent_feats[idx]                             # (1, hidden//2)
            combined = torch.cat([af, global_feat], dim=-1)        # (1, hidden//2+hidden)
            logits   = self.actor_head(combined).clone()           # (1, max_spots+1)

            # Mask padding slots (keep real spots 0..n_spots-1 and STAY at max_spots)
            if n_spots < self.max_spots:
                logits[0, n_spots:self.max_spots] = float("-inf")

            logits_list.append(logits)

        # --- critic ---
        value = self.critic_head(global_feat)   # (1, 1)
        return logits_list, value

    def get_action_and_value(
        self,
        state:         DayState,
        map_data:      MapData,
        cfg:           MatchConfig,
        agent_indices: List[int],
        deterministic: bool = False,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample (or argmax) one action per patrol agent.

        Returns:
          actions   : list of int — spot index (0..n_spots-1) or max_spots (STAY)
          log_probs : (n_agents,)
          entropy   : (n_agents,)
          value     : scalar tensor
        """
        logits_list, value = self.forward(state, map_data, cfg, agent_indices)

        actions, log_probs_list, entropy_list = [], [], []
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
