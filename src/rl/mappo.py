"""
MAPPO trainer for HEXA UDON.

Episode = 1 game on a randomly generated map.
Step    = 1 day (strategic level; A* handles tactical execution).

Key design decisions:
  - CurriculumEngine controls map difficulty (8x8 -> 32x32).
    Use random maps when no curriculum is provided.
  - SelfPlayPool provides opponent checkpoints. When the pool is non-empty,
    opponent agents are simulated each day to produce realistic traffic.
  - Reward shaping: potential-based phi(s) = -POTENTIAL_SCALE * mean_min_hex_dist
    to nearest uncollected spot (Ng 1999 — provably policy-invariant).
  - TensorBoard logging when available.
  - fuel_max randomised each episode (BTC value unknown; trains robustness).
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import random
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config as C
from env.hex_grid import HexGrid
from env.map_generator import generate_random_scenario
from env.models import (
    AgentState, DayOrder, DayState, MapData, MatchConfig,
)
from env.simulator import HexaUdonSimulator
from pathfinding.astar import multi_waypoint_path, find_path
from rl.actor_critic import ActorCritic
from rl.curriculum import CurriculumEngine
from rl.selfplay import SelfPlayPool
from strategy.greedy import GreedyPlanner

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False


# ------------------------------------------------------------------ #
# Rollout buffer                                                       #
# ------------------------------------------------------------------ #

@dataclass
class Transition:
    state:     DayState
    map_data:  MapData      # episode-level
    cfg:       MatchConfig  # episode-level
    actions:   List[int]
    log_probs: torch.Tensor
    value:     torch.Tensor
    reward:    float        # shaped reward
    done:      bool


class RolloutBuffer:
    def __init__(self):
        self.transitions: List[Transition] = []

    def add(self, t: Transition) -> None:
        self.transitions.append(t)

    def clear(self) -> None:
        self.transitions.clear()

    def __len__(self) -> int:
        return len(self.transitions)

    def compute_returns(
        self,
        gamma: float = C.GAMMA,
        lam:   float = C.GAE_LAMBDA,
    ) -> Tuple[List[float], List[float]]:
        """GAE-Lambda returns and advantages."""
        T          = len(self.transitions)
        returns    = [0.0] * T
        advantages = [0.0] * T
        gae        = 0.0
        next_val   = 0.0

        for t in reversed(range(T)):
            tr    = self.transitions[t]
            v     = tr.value.item()
            delta = tr.reward + gamma * next_val * (1 - tr.done) - v
            gae   = delta + gamma * lam * (1 - tr.done) * gae
            advantages[t] = gae
            returns[t]    = gae + v
            next_val      = 0.0 if tr.done else v

        return returns, advantages


# ------------------------------------------------------------------ #
# Trainer                                                              #
# ------------------------------------------------------------------ #

class MAPPOTrainer:
    """
    MAPPO trainer with optional curriculum learning and self-play.

    Args:
        max_spots, max_series, max_width, max_height:
            Fixed network dimensions; maps smaller than max are padded.
        device:     "cpu" or "cuda"
        log_dir:    TensorBoard log directory (None disables logging)
        curriculum: CurriculumEngine instance (None = fully random maps)
        selfplay:   SelfPlayPool instance (None = no opponent simulation)
    """

    FUEL_MAX_CANDIDATES = [10, 15, 20, 25, 30]

    def __init__(
        self,
        max_spots:  int = 30,
        max_series: int = 10,
        max_width:  int = 32,
        max_height: int = 32,
        device:     str = "cpu",
        log_dir:    str = "runs/mappo",
        curriculum: Optional[CurriculumEngine] = None,
        selfplay:   Optional[SelfPlayPool]     = None,
    ):
        self.max_spots  = max_spots
        self.max_series = max_series
        self.device     = torch.device(device)
        self.curriculum = curriculum
        self.selfplay   = selfplay

        self.model = ActorCritic(
            max_spots  = max_spots,
            max_series = max_series,
            max_width  = max_width,
            max_height = max_height,
        ).to(self.device)

        self.optimizer = optim.Adam([
            {"params": self.model.map_encoder.parameters(),  "lr": C.LR_ACTOR},
            {"params": self.model.agent_mlp.parameters(),    "lr": C.LR_ACTOR},
            {"params": self.model.global_mlp.parameters(),   "lr": C.LR_ACTOR},
            {"params": self.model.actor_head.parameters(),   "lr": C.LR_ACTOR},
            {"params": self.model.critic_head.parameters(),  "lr": C.LR_CRITIC},
        ])

        self.buffer = RolloutBuffer()

        if _TB_AVAILABLE and log_dir:
            self.writer = SummaryWriter(log_dir=log_dir)
        else:
            self.writer = None
            if log_dir and not _TB_AVAILABLE:
                print("[mappo] TensorBoard not available; install tensorboard for logging.")

    # ------------------------------------------------------------------ #
    # Training loop                                                        #
    # ------------------------------------------------------------------ #

    def train(
        self,
        n_episodes:          int = 1000,
        seed:                int = 42,
        log_every:           int = 50,
        eval_baseline_every: int = 10,   # how often to evaluate vs Lookahead (curriculum)
    ) -> None:
        ep_shaped:  List[float] = []
        ep_raw:     List[float] = []
        ep_series:  List[int]   = []
        ep_udon:    List[int]   = []

        for ep in range(n_episodes):
            # --- Generate scenario ---
            if self.curriculum:
                cfg, map_data, agents = self.curriculum.generate_scenario(seed=seed + ep)
            else:
                cfg, map_data, agents = generate_random_scenario(seed=seed + ep)

            # --- Collect episode ---
            ep_info = self._collect_episode(cfg, map_data, agents, seed=seed + ep)
            ep_shaped.append(ep_info["shaped_return"])
            ep_raw.append(ep_info["raw_return"])
            ep_series.append(ep_info["unique_series"])
            ep_udon.append(ep_info["total_udon"])

            # --- PPO update ---
            losses: Dict[str, float] = {}
            if len(self.buffer) > 0:
                losses = self._update()
                self.buffer.clear()

            # --- Curriculum: compare vs Lookahead baseline ---
            if self.curriculum and (ep + 1) % eval_baseline_every == 0:
                import copy
                baseline = self.curriculum.evaluate_baseline(
                    cfg, map_data, copy.deepcopy(agents)
                )
                self.curriculum.record(ep_info["unique_series"], baseline)
                advanced = self.curriculum.try_advance()
                if advanced and self.writer:
                    self.writer.add_scalar(
                        "curriculum/level", self.curriculum.level_number, ep + 1
                    )

            # --- Self-play: update pool ---
            if self.selfplay:
                self.selfplay.step(self.model)

            # --- TensorBoard ---
            if self.writer is not None:
                g = ep + 1
                self.writer.add_scalar("episode/shaped_return", ep_info["shaped_return"], g)
                self.writer.add_scalar("episode/raw_return",    ep_info["raw_return"],    g)
                self.writer.add_scalar("episode/unique_series", ep_info["unique_series"], g)
                self.writer.add_scalar("episode/total_udon",    ep_info["total_udon"],    g)
                if self.curriculum:
                    self.writer.add_scalar("curriculum/win_rate",    self.curriculum.win_rate,    g)
                    self.writer.add_scalar("curriculum/level_number", self.curriculum.level_number, g)
                for k, v in losses.items():
                    self.writer.add_scalar(f"train/{k}", v, g)

            # --- Console ---
            if (ep + 1) % log_every == 0:
                n   = log_every
                lvl = f" lv={self.curriculum.level_number}" if self.curriculum else ""
                print(
                    f"Ep {ep+1:5d}{lvl} | "
                    f"shaped={np.mean(ep_shaped[-n:]):.1f}  "
                    f"raw={np.mean(ep_raw[-n:]):.1f}  "
                    f"series={np.mean(ep_series[-n:]):.2f}  "
                    f"udon={np.mean(ep_udon[-n:]):.1f}"
                    + (f"  loss={losses.get('total', 0):.4f}" if losses else "")
                )

        if self.writer is not None:
            self.writer.flush()

    # ------------------------------------------------------------------ #
    # Episode collection                                                   #
    # ------------------------------------------------------------------ #

    def _collect_episode(
        self,
        cfg:      MatchConfig,
        map_data: MapData,
        agents:   List[AgentState],
        seed:     int = 0,
    ) -> Dict[str, float]:
        fuel_max = random.choice(self.FUEL_MAX_CANDIDATES)
        self.model.set_fuel_max(fuel_max)
        for a in agents:
            if a.is_patrol():
                a.fuel = fuel_max

        sim = HexaUdonSimulator(cfg, map_data)

        # --- Set up opponent agents (self-play) ---
        opp_agents: Optional[List[AgentState]] = None
        if self.selfplay and self.selfplay.has_opponent():
            opp_model  = self.selfplay.sample()
            our_cells  = [a.cell for a in agents]
            n_opp      = max(1, len(agents) // 2)
            n_opp_pat  = max(1, n_opp // 2)
            opp_agents = SelfPlayPool.make_opponent_agents(
                map_data, cfg,
                n_agents   = n_opp,
                n_patrol   = n_opp_pat,
                our_cells  = our_cells,
                seed       = seed + 9999,
            )
            for a in opp_agents:
                if a.is_patrol():
                    a.fuel = fuel_max

        # --- Initial state ---
        state     = sim.reset(agents)
        if opp_agents:
            state = replace(state, opponent_cells=[a.cell for a in opp_agents])

        raw_total    = 0.0
        shaped_total = 0.0
        phi_s        = self._compute_potential(state, map_data, sim.grid)

        while not sim.is_done(state):
            patrol_ids = [a.id for a in state.patrol_agents()]

            with torch.no_grad():
                actions, log_probs, entropy, value = self.model.get_action_and_value(
                    state, map_data, cfg, patrol_ids
                )

            orders = self._actions_to_orders(state, map_data, cfg, sim, patrol_ids, actions)

            # --- Opponent simulation (self-play) ---
            opp_road_steps: Optional[Dict[int, float]] = None
            if opp_agents and self.selfplay:
                new_opp_cells, opp_road_steps = SelfPlayPool.simulate_day(
                    opp_model, opp_agents, state, map_data, cfg, sim.grid
                )
                # Update opponent agent positions
                cell_map = {a.id: c for a, c in zip(opp_agents, new_opp_cells)}
                for a in opp_agents:
                    a.cell = cell_map.get(a.id, a.cell)

            next_state, reward = sim.apply_day(
                state, orders,
                opponent_step_counts=opp_road_steps,
            )

            # Update opponent cells in next state
            if opp_agents:
                next_state = replace(
                    next_state,
                    opponent_cells=[a.cell for a in opp_agents],
                )

            done     = sim.is_done(next_state)
            phi_next = self._compute_potential(next_state, map_data, sim.grid)
            shaped_r = reward + C.GAMMA * phi_next - phi_s
            phi_s    = phi_next

            self.buffer.add(Transition(
                state     = state,
                map_data  = map_data,
                cfg       = cfg,
                actions   = actions,
                log_probs = log_probs.detach(),
                value     = value.detach(),
                reward    = shaped_r,
                done      = done,
            ))

            state         = next_state
            raw_total    += reward
            shaped_total += shaped_r

        return {
            "raw_return":    raw_total,
            "shaped_return": shaped_total,
            "unique_series": len(state.collected_series),
            "total_udon":    state.total_udon,
        }

    # ------------------------------------------------------------------ #
    # PPO update                                                           #
    # ------------------------------------------------------------------ #

    def _update(self) -> Dict[str, float]:
        returns, advantages = self.buffer.compute_returns()
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.tensor(returns,    dtype=torch.float32)

        total_loss   = 0.0
        total_actor  = 0.0
        total_critic = 0.0
        total_ent    = 0.0
        count        = 0

        for _ in range(C.N_EPOCHS):
            for i, tr in enumerate(self.buffer.transitions):
                patrol_ids = [a.id for a in tr.state.patrol_agents()]
                _, new_log_probs, new_entropy, new_value = \
                    self.model.get_action_and_value(
                        tr.state, tr.map_data, tr.cfg, patrol_ids
                    )

                ratio        = (new_log_probs.mean() - tr.log_probs.mean()).exp()
                adv          = adv_t[i]
                surr1        = ratio * adv
                surr2        = torch.clamp(ratio, 1 - C.CLIP_EPS, 1 + C.CLIP_EPS) * adv
                actor_loss   = -torch.min(surr1, surr2)
                critic_loss  = (new_value - ret_t[i]).pow(2)
                entropy_loss = -new_entropy.mean()

                loss = actor_loss + 0.5 * critic_loss + C.ENTROPY_COEF * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

                total_loss   += loss.item()
                total_actor  += actor_loss.item()
                total_critic += critic_loss.item()
                total_ent    += entropy_loss.item()
                count        += 1

        denom = max(count, 1)
        return {
            "total":   total_loss   / denom,
            "actor":   total_actor  / denom,
            "critic":  total_critic / denom,
            "entropy": total_ent    / denom,
        }

    # ------------------------------------------------------------------ #
    # Reward shaping                                                       #
    # ------------------------------------------------------------------ #

    def _compute_potential(
        self,
        state:    DayState,
        map_data: MapData,
        grid:     HexGrid,
    ) -> float:
        """
        phi(s) = -POTENTIAL_SCALE * mean over patrol agents of
                  min hex_distance to any spot in an uncollected series.
        Returns 0 when all series collected or no patrol agents exist.
        """
        uncollected = [
            s for s in map_data.spots
            if s.series_id not in state.collected_series
        ]
        patrol = list(state.patrol_agents())
        if not uncollected or not patrol:
            return 0.0

        total = 0.0
        for agent in patrol:
            total += min(grid.hex_distance(agent.cell, s.cell_id) for s in uncollected)
        return -C.POTENTIAL_SCALE * (total / len(patrol))

    # ------------------------------------------------------------------ #
    # Action -> Orders                                                     #
    # ------------------------------------------------------------------ #

    def _actions_to_orders(
        self,
        state:      DayState,
        map_data:   MapData,
        cfg:        MatchConfig,
        sim:        HexaUdonSimulator,
        patrol_ids: List[int],
        actions:    List[int],
    ) -> List[DayOrder]:
        """Convert spot-index actions -> DayOrders via A*."""
        orders:      List[DayOrder] = []
        terrain      = {c.id: c.terrain for c in map_data.cells}
        agents_by_id = state.agents_by_id()
        n_spots      = len(map_data.spots)

        for aid, act in zip(patrol_ids, actions):
            agent = agents_by_id[aid]
            if act >= n_spots:
                orders.append(DayOrder(agent_id=aid, actions=[]))
                continue
            target_cell  = map_data.spots[act].cell_id
            path_actions = multi_waypoint_path(
                sim.grid, terrain, state.traffic,
                agent.cell, [target_cell],
                step_budget = state.steps_left,
                fuel_budget = agent.fuel if agent.is_patrol() else None,
            )
            orders.append(DayOrder(agent_id=aid, actions=path_actions))

        greedy = GreedyPlanner(cfg, map_data, sim)
        for agent in state.supply_agents():
            target = greedy._supply_target(agent, state)
            if target is not None:
                result = find_path(
                    sim.grid, terrain, state.traffic,
                    agent.cell, target, step_budget=state.steps_left,
                )
                supply_actions = result.actions if result.reachable else []
            else:
                supply_actions = []
            orders.append(DayOrder(agent_id=agent.id, actions=supply_actions))

        return orders

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        payload: dict = {
            "model":      self.model.state_dict(),
            "max_spots":  self.max_spots,
            "max_series": self.max_series,
        }
        if self.curriculum:
            payload["curriculum"] = self.curriculum.state_dict()
        if self.selfplay:
            payload["selfplay"] = self.selfplay.state_dict()
        torch.save(payload, path)
        print(f"[mappo] Model saved -> {path}")

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        if self.curriculum and "curriculum" in ckpt:
            self.curriculum.load_state_dict(ckpt["curriculum"])
        print(f"[mappo] Model loaded <- {path}")
