"""
MAPPO trainer for HEXA UDON.

Episode = 1 game (4–10 days).
Step    = 1 day (strategic level, A* handles tactical execution).

High-level loop:
  for each episode:
    state = sim.reset(initial_agents)
    while not done:
      actions, log_probs, entropy, value = model.get_action_and_value(state, patrol_ids)
      targets = [spots[a] if a < n_spots else STAY for a in actions]
      orders  = build_orders(targets, supply_rules)
      next_state, reward = sim.apply_day(state, orders)
      buffer.add(state, actions, log_probs, reward, value, done)
      state = next_state
  update(buffer)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config as C
from env.models import (
    AgentAction, AgentState, CMD_MOVE, CMD_STAY,
    DayOrder, DayState, MapData, MatchConfig,
)
from env.simulator import HexaUdonSimulator
from pathfinding.astar import multi_waypoint_path
from rl.actor_critic import ActorCritic
from strategy.greedy import GreedyPlanner


@dataclass
class Transition:
    state:     DayState
    actions:   List[int]          # spot indices per patrol agent
    log_probs: torch.Tensor       # (n_patrol,)
    value:     torch.Tensor       # scalar
    reward:    float
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

    def compute_returns(self, gamma: float = C.GAMMA, lam: float = C.GAE_LAMBDA) -> Tuple[List[float], List[float]]:
        """GAE-Lambda return and advantage estimates."""
        T = len(self.transitions)
        returns    = [0.0] * T
        advantages = [0.0] * T

        gae = 0.0
        next_val = 0.0
        for t in reversed(range(T)):
            tr    = self.transitions[t]
            r     = tr.reward
            v     = tr.value.item()
            done  = tr.done
            delta = r + gamma * next_val * (1 - done) - v
            gae   = delta + gamma * lam * (1 - done) * gae
            advantages[t] = gae
            returns[t]    = gae + v
            next_val = 0.0 if done else v
        return returns, advantages


class MAPPOTrainer:
    def __init__(
        self,
        cfg:      MatchConfig,
        map_data: MapData,
        sim:      HexaUdonSimulator,
        initial_agents_fn,   # callable() -> List[AgentState]
        device:   str = "cpu",
    ):
        self.cfg       = cfg
        self.map       = map_data
        self.sim       = sim
        self.initial_fn = initial_agents_fn
        self.device    = torch.device(device)

        self.model = ActorCritic(cfg, map_data).to(self.device)
        self.optimizer = optim.Adam([
            {"params": self.model.map_encoder.parameters(),  "lr": C.LR_ACTOR},
            {"params": self.model.agent_mlp.parameters(),    "lr": C.LR_ACTOR},
            {"params": self.model.global_mlp.parameters(),   "lr": C.LR_ACTOR},
            {"params": self.model.actor_head.parameters(),   "lr": C.LR_ACTOR},
            {"params": self.model.critic_head.parameters(),  "lr": C.LR_CRITIC},
        ])

        self.greedy_fallback = GreedyPlanner(cfg, map_data, sim)
        self.buffer = RolloutBuffer()

        # BTC has not published fuel_max yet. We train across a range of plausible
        # values so the model is robust until the real value is announced.
        # TODO: once BTC publishes fuel_max, set FUEL_MAX_CANDIDATES = [<real_value>]
        #       and retrain for best performance.
        self.FUEL_MAX_CANDIDATES = [10, 15, 20, 25, 30]

    def set_fuel_max(self, fuel_max: int) -> None:
        """Lock fuel normalization to the real match value before playing."""
        self.model.set_fuel_max(fuel_max)

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(self, n_episodes: int = 1000, log_every: int = 50) -> None:
        ep_returns = []

        for ep in range(n_episodes):
            ep_return = self._collect_episode()
            ep_returns.append(ep_return)

            if len(self.buffer) > 0:
                loss = self._update()
                self.buffer.clear()

            if (ep + 1) % log_every == 0:
                mean_ret = np.mean(ep_returns[-log_every:])
                print(f"Episode {ep+1:5d} | mean_return={mean_ret:.1f}")

    def _collect_episode(self) -> float:
        # Sample a random fuel_max each episode so the model learns to work
        # across different match configurations (fuel_max is unknown at train time).
        fuel_max = random.choice(self.FUEL_MAX_CANDIDATES)
        self.model.set_fuel_max(fuel_max)

        agents = self.initial_fn()
        for a in agents:
            if a.is_patrol():
                a.fuel = fuel_max   # start day-1 with full tank

        state = self.sim.reset(agents)
        total  = 0.0

        while not self.sim.is_done(state):
            patrol_ids = [a.id for a in state.patrol_agents()]

            with torch.no_grad():
                actions, log_probs, entropy, value = self.model.get_action_and_value(
                    state, patrol_ids
                )

            orders = self._actions_to_orders(state, patrol_ids, actions)
            next_state, reward = self.sim.apply_day(state, orders)
            done = self.sim.is_done(next_state)

            self.buffer.add(Transition(
                state=state,
                actions=actions,
                log_probs=log_probs.detach(),
                value=value.detach(),
                reward=reward,
                done=done,
            ))
            state  = next_state
            total += reward

        return total

    def _update(self) -> float:
        returns, advantages = self.buffer.compute_returns()
        adv_t = torch.tensor(advantages, dtype=torch.float32)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.tensor(returns, dtype=torch.float32)

        total_loss = 0.0
        for _ in range(C.N_EPOCHS):
            for i, tr in enumerate(self.buffer.transitions):
                patrol_ids = [a.id for a in tr.state.patrol_agents()]
                new_actions, new_log_probs, new_entropy, new_value = \
                    self.model.get_action_and_value(tr.state, patrol_ids)

                old_lp = tr.log_probs
                # Use mean over agents
                ratio  = (new_log_probs.mean() - old_lp.mean()).exp()
                adv    = adv_t[i]

                surr1  = ratio * adv
                surr2  = torch.clamp(ratio, 1 - C.CLIP_EPS, 1 + C.CLIP_EPS) * adv
                actor_loss  = -torch.min(surr1, surr2)
                critic_loss = (new_value - ret_t[i]).pow(2)
                entropy_loss = -new_entropy.mean()

                loss = actor_loss + 0.5 * critic_loss + C.ENTROPY_COEF * entropy_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()

        return total_loss

    # ------------------------------------------------------------------ #
    # Action → Orders translation                                          #
    # ------------------------------------------------------------------ #

    def _actions_to_orders(
        self,
        state:      DayState,
        patrol_ids: List[int],
        actions:    List[int],
    ) -> List[DayOrder]:
        """Convert spot-index actions to DayOrders using A* pathfinding."""
        orders: List[DayOrder] = []
        terrain = {c.id: c.terrain for c in self.map.cells}
        agents  = state.agents_by_id()

        steps_left  = state.steps_left
        patrol_step_share = steps_left  # simplification; real budget is shared

        for aid, act in zip(patrol_ids, actions):
            agent = agents[aid]
            if act >= self.model.n_spots:
                # STAY
                orders.append(DayOrder(agent_id=aid, actions=[]))
                continue

            target_cell = self.map.spots[act].cell_id
            path_actions = multi_waypoint_path(
                self.sim.grid,
                terrain,
                state.traffic,
                agent.cell,
                [target_cell],
                step_budget=patrol_step_share,
                fuel_budget=agent.fuel if agent.is_patrol() else None,
            )
            orders.append(DayOrder(agent_id=aid, actions=path_actions))

        # Supply cars: use greedy rule
        for agent in state.supply_agents():
            target = self.greedy_fallback._supply_target(agent, state)
            if target is not None:
                from pathfinding.astar import find_path
                result = find_path(
                    self.sim.grid, terrain, state.traffic,
                    agent.cell, target, step_budget=steps_left,
                )
                actions_supply = result.actions if result.reachable else []
            else:
                actions_supply = []
            orders.append(DayOrder(agent_id=agent.id, actions=actions_supply))

        return orders

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Model loaded from {path}")
