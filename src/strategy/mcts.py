"""
AlphaZero-style MCTS for HEXA UDON.

Each node = game state at the START of a day.
Each edge = a full set of DayOrders for that day.

Search uses:
  - Actor network logits  to generate K candidate joint actions per node
  - Critic network value  to evaluate leaf nodes (no random rollout)
  - UCB1                  to balance exploration vs exploitation
  - Time budget           checked between inference, expansion and simulation;
                          the caller reserves time for submission

Usage:
    planner = MCTSPlanner(cfg, map_data, sim, model, time_budget_ms=2500)
    orders  = planner.plan(state)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import copy
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch

import config as C
from env.models import DayOrder, DayState, MapData, MatchConfig
from env.simulator import HexaUdonSimulator, TrafficModel
from env.scoring import collection_potential
from pathfinding.astar import multi_waypoint_path, find_path
from strategy.greedy import GreedyPlanner
from strategy.planner import BasePlanner


# ------------------------------------------------------------------ #
# Tree node                                                            #
# ------------------------------------------------------------------ #

@dataclass
class MCTSNode:
    state:      DayState
    parent:     Optional["MCTSNode"]
    orders:     List[DayOrder]          # orders that led TO this node
    reward:     float                   # immediate reward received on arrival
    depth:      int

    children:   List["MCTSNode"] = field(default_factory=list, repr=False)
    visits:     int   = 0
    value_sum:  float = 0.0
    traffic: Optional[TrafficModel] = field(default=None, repr=False)

    @property
    def is_terminal(self) -> bool:
        return not self.children and self.visits > 0  # expanded but no children = terminal

    @property
    def q(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def ucb1(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        assert self.parent is not None
        return self.q + c * math.sqrt(math.log(self.parent.visits) / self.visits)


# ------------------------------------------------------------------ #
# Planner                                                              #
# ------------------------------------------------------------------ #

class MCTSPlanner(BasePlanner):
    """
    MCTS planner that wraps a trained ActorCritic model.

    Args:
        cfg, map_data, sim : standard planner args
        model              : trained ActorCritic (must match current map dims)
        time_budget_ms     : wall-clock ms to spend searching (default 2500)
        beam_width         : candidate actions per node expansion (default 5)
        c_puct             : UCB1 exploration constant (default 1.5)
        max_depth          : max MCTS tree depth from root (default 3)
    """

    def __init__(
        self,
        cfg:            MatchConfig,
        map_data:       MapData,
        sim:            HexaUdonSimulator,
        model,
        time_budget_ms: int   = 2500,
        beam_width:     int   = 5,
        c_puct:         float = 1.5,
        max_depth:      int   = 3,
    ):
        super().__init__(cfg, map_data, sim)
        self._model          = model
        self._model.eval()
        self._time_budget_ms = time_budget_ms
        self._beam           = beam_width
        self._c              = c_puct
        self._max_depth      = max_depth
        self._terrain        = self.terrain_dict()
        self._greedy         = GreedyPlanner(cfg, map_data, sim)

    # ------------------------------------------------------------------ #
    # Public entry                                                         #
    # ------------------------------------------------------------------ #

    def plan(self, state: DayState) -> List[DayOrder]:
        """
        Run MCTS from state until time budget is spent.
        Returns the DayOrders of the most-visited child of the root.
        Falls back to greedy if tree has no children (e.g., terminal state).
        """
        deadline = time.monotonic() + self._time_budget_ms / 1000.0
        root     = MCTSNode(state=state, parent=None, orders=[], reward=0.0, depth=0)
        root.traffic = copy.deepcopy(self.sim.traffic)

        iterations = 0
        while time.monotonic() < deadline:
            node = self._select(root)
            if not self.sim.is_done(node.state) and node.depth < self._max_depth:
                children = self._expand(node, deadline)
                if children:
                    node = children[0]   # evaluate first new child
            if time.monotonic() >= deadline:
                break
            value = self._evaluate(node)
            self._backup(node, value)
            iterations += 1

        if not root.children:
            return self._greedy.plan(state)

        best = max(root.children, key=lambda n: n.visits)
        return best.orders

    # ------------------------------------------------------------------ #
    # MCTS phases                                                          #
    # ------------------------------------------------------------------ #

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Descend using UCB1 until an unexpanded or terminal node."""
        while node.children:
            if self.sim.is_done(node.state):
                break
            node = max(node.children, key=lambda n: n.ucb1(self._c))
        return node

    def _expand(self, node: MCTSNode, deadline: float = float('inf')) -> List[MCTSNode]:
        """
        Generate up to beam_width child nodes by sampling diverse joint actions
        from the actor network, then executing them through the simulator.
        """
        if time.monotonic() >= deadline:
            return []
        state      = node.state
        patrol_ids = [a.id for a in state.patrol_agents()]
        n_spots    = len(self.map.spots)

        # Get actor logits for all patrol agents
        with torch.no_grad():
            logits_list, _ = self._model.forward(state, self.map, self.cfg, patrol_ids)

        # Sample K diverse joint actions (with deduplication)
        seen:     set             = set()
        children: List[MCTSNode] = []

        for _ in range(self._beam * 6):   # over-sample to find K distinct ones
            if len(children) >= self._beam or time.monotonic() >= deadline:
                break
            actions = []
            for logits in logits_list:
                dist = torch.distributions.Categorical(logits=logits.squeeze(0))
                actions.append(dist.sample().item())

            key = tuple(actions)
            if key in seen:
                continue
            seen.add(key)

            orders              = self._build_orders(state, patrol_ids, actions)
            if time.monotonic() >= deadline:
                break
            branch_sim = HexaUdonSimulator(self.cfg, self.map)
            branch_sim.traffic = copy.deepcopy(node.traffic if node.traffic is not None else self.sim.traffic)
            next_state, reward = branch_sim.apply_day(state, orders)

            children.append(MCTSNode(
                state   = next_state,
                parent  = node,
                orders  = orders,
                reward  = reward,
                depth   = node.depth + 1,
                traffic = branch_sim.traffic,
            ))

        node.children = children
        return children

    def _evaluate(self, node: MCTSNode) -> float:
        """
        Continuation value at this state; edge rewards are added in backup.
        A terminal state has no future reward.
        """
        if self.sim.is_done(node.state):
            return 0.0

        with torch.no_grad():
            patrol_ids = [a.id for a in node.state.patrol_agents()]
            _, value   = self._model.forward(node.state, self.map, self.cfg, patrol_ids)
        # Training uses r + gamma*phi(next) - phi(now), so V_raw = V_shaped + phi.
        return value.item() + collection_potential(node.state, self.map, self.grid)

    def _backup(self, node: MCTSNode, value: float) -> None:
        """Propagate value up to the root."""
        cur = node
        while cur is not None:
            value          = cur.reward + C.GAMMA * value
            cur.visits    += 1
            cur.value_sum += value
            cur            = cur.parent

    # ------------------------------------------------------------------ #
    # Action -> Orders                                                     #
    # ------------------------------------------------------------------ #

    def _build_orders(
        self,
        state:      DayState,
        patrol_ids: List[int],
        actions:    List[int],
    ) -> List[DayOrder]:
        """Convert actor actions (spot indices) to DayOrders via A* using sequential step allocation."""
        orders:      List[DayOrder] = []
        agents_by_id = state.agents_by_id()
        n_spots      = len(self.map.spots)
        remaining_shared_steps = state.steps_left

        for aid, act in zip(patrol_ids, actions):
            agent = agents_by_id[aid]
            if act >= n_spots:
                orders.append(DayOrder(agent_id=aid, actions=[]))
                continue
            target       = self.map.spots[act].cell_id
            path_actions = multi_waypoint_path(
                self.grid, self._terrain, state.traffic,
                agent.cell, [target],
                step_budget = remaining_shared_steps,
                fuel_budget = agent.fuel if agent.is_patrol() else None,
            )
            orders.append(DayOrder(agent_id=aid, actions=path_actions))

            # Calculate actual steps used by this agent's actions
            actual_steps_used = 0
            cur_cell = agent.cell
            for a in path_actions:
                if a.cmd == "move":
                    actual_steps_used += self.sim._step_cost(cur_cell, state.traffic)
                    dst = self.grid.neighbor_in_dir(cur_cell, a.direction)
                    if dst is not None:
                        cur_cell = dst
            remaining_shared_steps = max(0, remaining_shared_steps - actual_steps_used)

        for agent in state.supply_agents():
            target = self._greedy._supply_target(agent, state)
            if target is not None:
                result = find_path(
                    self.grid, self._terrain, state.traffic,
                    agent.cell, target, step_budget=remaining_shared_steps,
                )
                supply_actions = result.actions if result.reachable else []
            else:
                supply_actions = []
            orders.append(DayOrder(agent_id=agent.id, actions=supply_actions))

            # Calculate actual steps used by this supply car
            actual_steps_used = 0
            cur_cell = agent.cell
            for a in supply_actions:
                if a.cmd == "move":
                    actual_steps_used += self.sim._step_cost(cur_cell, state.traffic)
                    dst = self.grid.neighbor_in_dir(cur_cell, a.direction)
                    if dst is not None:
                        cur_cell = dst
            remaining_shared_steps = max(0, remaining_shared_steps - actual_steps_used)

        return orders
