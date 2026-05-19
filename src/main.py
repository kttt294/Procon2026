"""
Entry point for HEXA UDON bot.

Modes:
  train   -- run MAPPO training on the simulator
  play    -- connect to contest server and play a real match
  sim     -- run greedy baseline on simulator (quick sanity check)

Usage:
  python main.py train --episodes 2000 --save model.pt
  python main.py play  --url http://192.168.1.100:8080 --model model.pt
  python main.py sim
"""
from __future__ import annotations

import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from env.models import AgentState, DayOrder, MatchConfig, MapData
from env.simulator import HexaUdonSimulator
from strategy.greedy import GreedyPlanner
from strategy.lookahead import LookaheadPlanner


# ------------------------------------------------------------------ #
# Demo / sanity-check data                                            #
# ------------------------------------------------------------------ #

def _demo_config() -> MatchConfig:
    return MatchConfig(
        width=10, height=8,
        total_days=6,
        steps_per_day=[120, 120, 100, 100, 80, 80],
        n_teams=4,
        traffic_threshold_busy=3.0,
        traffic_threshold_congested=7.0,
    )


def _demo_map(cfg: MatchConfig):
    from env.models import Cell, MapData, Spot
    cells = []
    for i in range(cfg.width * cfg.height):
        # Simple: all plain except a few roads and a lake
        if i in (5, 6, 7, 15, 16, 17):
            terrain = 3   # road
        elif i in (22, 32):
            terrain = 2   # lake
        elif i in (11, 21, 31):
            terrain = 1   # mountain
        else:
            terrain = 0   # plain
        cells.append(Cell(id=i, terrain=terrain))

    spots = [
        Spot(cell_id=12, series_id=1, max_inventory=3),
        Spot(cell_id=24, series_id=2, max_inventory=2),
        Spot(cell_id=35, series_id=1, max_inventory=1),
        Spot(cell_id=47, series_id=3, max_inventory=2),
        Spot(cell_id=58, series_id=2, max_inventory=3),
    ]
    return MapData(cells=cells, spots=spots)


def _demo_agents() -> list[AgentState]:
    return [
        AgentState(id=1, type=0, cell=0,  fuel=20),   # patrol
        AgentState(id=2, type=0, cell=70, fuel=20),   # patrol
        AgentState(id=3, type=1, cell=40, fuel=0),    # supply
    ]


# ------------------------------------------------------------------ #
# Modes                                                               #
# ------------------------------------------------------------------ #

def _run_one(label: str, planner, sim, agents_fn):
    state = sim.reset(agents_fn())
    total_reward = 0.0
    print(f"\n=== {label} ===")
    while not sim.is_done(state):
        orders = planner.plan(state)
        state, reward = sim.apply_day(state, orders)
        total_reward += reward
        print(
            f"  Day {state.day-1}: "
            f"series={sorted(state.collected_series)}  "
            f"udon={state.total_udon}  "
            f"reward={reward:.1f}"
        )
    print(f"  Final: unique_series={len(state.collected_series)}  "
          f"total_udon={state.total_udon}  cumulative_reward={total_reward:.1f}")
    return len(state.collected_series), state.total_udon


def run_sim(args):
    """Compare Greedy vs Lookahead on demo map."""
    cfg      = _demo_config()
    map_data = _demo_map(cfg)
    sim      = HexaUdonSimulator(cfg, map_data)

    greedy    = GreedyPlanner(cfg, map_data, sim)
    lookahead = LookaheadPlanner(cfg, map_data, sim)

    g_series, g_udon = _run_one("Greedy",    greedy,    sim, _demo_agents)
    l_series, l_udon = _run_one("Lookahead", lookahead, sim, _demo_agents)

    print(f"\n{'='*40}")
    print(f"Greedy   : {g_series} series, {g_udon} udon")
    print(f"Lookahead: {l_series} series, {l_udon} udon")


def run_train(args):
    """Train MAPPO with optional curriculum and self-play."""
    from rl.mappo import MAPPOTrainer
    from rl.curriculum import CurriculumEngine
    from rl.selfplay import SelfPlayPool

    curriculum = CurriculumEngine(start_level=args.start_level) if args.curriculum else None
    selfplay   = SelfPlayPool(update_every=args.selfplay_every)  if args.selfplay   else None

    trainer = MAPPOTrainer(
        device     = args.device,
        log_dir    = args.log_dir,
        curriculum = curriculum,
        selfplay   = selfplay,
    )

    if args.load and os.path.exists(args.load):
        trainer.load(args.load)

    trainer.train(
        n_episodes = args.episodes,
        seed       = args.seed,
        log_every  = args.log_every,
    )

    if args.save:
        trainer.save(args.save)


def run_play(args):
    """
    Connect to contest server and play.

    Fallback chain (each tier used if the previous raises or times out):
      MCTS  (if --mcts and model loaded, ~2500ms)
       -> RL policy (deterministic, ~50ms)
       -> Lookahead (~5ms)
       -> Greedy (~1ms, never crashes)
    """
    from client.http_client import ContestClient
    from rl.mappo import MAPPOTrainer

    client = ContestClient(base_url=args.url)

    print("[play] Fetching match config...")
    cfg, map_data, initial_agents = client.get_match_config()
    sim = HexaUdonSimulator(cfg, map_data)

    greedy    = GreedyPlanner(cfg, map_data, sim)
    lookahead = LookaheadPlanner(cfg, map_data, sim)

    use_rl = args.model and os.path.exists(args.model)
    rl_model = None
    if use_rl:
        trainer = MAPPOTrainer(device="cpu")
        trainer.load(args.model)
        rl_model = trainer.model
        rl_model.eval()
        print("[play] RL model loaded.")

    mcts_planner = None
    if use_rl and getattr(args, "mcts", False):
        from strategy.mcts import MCTSPlanner
        mcts_planner = MCTSPlanner(cfg, map_data, sim, rl_model, time_budget_ms=2500)
        print("[play] MCTS enabled.")

    def _rl_plan(state):
        patrol_ids = [a.id for a in state.patrol_agents()]
        actions, _, _, _ = rl_model.get_action_and_value(
            state, map_data, cfg, patrol_ids, deterministic=True,
        )
        return trainer._actions_to_orders(state, map_data, cfg, sim, patrol_ids, actions)

    def safe_plan(state, time_limit_ms):
        """Try MCTS -> RL -> Lookahead -> Greedy."""
        # Tier 1: MCTS
        if mcts_planner is not None:
            try:
                mcts_planner._time_budget_ms = max(500, time_limit_ms - 800)
                return mcts_planner.plan(state)
            except Exception as e:
                print(f"[play] MCTS error: {e}")

        # Tier 2: RL policy
        if rl_model is not None:
            try:
                return _rl_plan(state)
            except Exception as e:
                print(f"[play] RL error: {e}")

        # Tier 3: Lookahead
        try:
            return lookahead.plan(state)
        except Exception as e:
            print(f"[play] Lookahead error: {e}")

        # Tier 4: Greedy (must not crash)
        return greedy.plan(state)

    prev_state = None
    fuel_max_locked = False
    for day in range(1, cfg.total_days + 1):
        print(f"\n[play] Day {day}")
        start = time.time()

        state = client.get_day_state(day, cfg, map_data, prev_state)
        time_limit_ms = state.time_limit_ms

        # Day 1: read fuel_max from initial agent states and set model normalization.
        if day == 1 and not fuel_max_locked and rl_model is not None:
            observed = [a.fuel for a in state.my_agents if a.is_patrol()]
            if observed:
                rl_model.set_fuel_max(max(observed))
                print(f"[play] fuel_max inferred = {max(observed)}")
            fuel_max_locked = True

        # Pre-submit greedy immediately (~1ms) as a safe placeholder
        greedy_orders = greedy.plan(state)

        # Main planning (MCTS -> RL -> Lookahead -> Greedy)
        orders = safe_plan(state, time_limit_ms)

        resp = client.submit_with_retry(
            day             = day,
            orders          = orders,
            fallback_orders = greedy_orders,
            deadline_ms     = time_limit_ms,
            start_ms        = start,
        )
        elapsed = (time.time() - start) * 1000
        print(f"[play] Submitted — status={resp.get('status')}  elapsed={elapsed:.0f}ms")
        prev_state = state

    print("\n[play] Match complete.")


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="HEXA UDON Bot")
    sub    = parser.add_subparsers(dest="mode", required=True)

    # sim
    sub.add_parser("sim", help="Greedy sanity check on demo map")

    # train
    tr = sub.add_parser("train", help="Train MAPPO")
    tr.add_argument("--episodes",      type=int,   default=1000)
    tr.add_argument("--save",          type=str,   default="model.pt")
    tr.add_argument("--load",          type=str,   default=None)
    tr.add_argument("--device",        type=str,   default="cpu")
    tr.add_argument("--seed",          type=int,   default=42)
    tr.add_argument("--log-every",     type=int,   default=50,   dest="log_every")
    tr.add_argument("--log-dir",       type=str,   default="runs/mappo", dest="log_dir")
    tr.add_argument("--curriculum",    action="store_true", help="Enable curriculum learning")
    tr.add_argument("--start-level",   type=int,   default=0,    dest="start_level",
                    help="Curriculum start level 0-4")
    tr.add_argument("--selfplay",      action="store_true", help="Enable self-play opponent pool")
    tr.add_argument("--selfplay-every",type=int,   default=1000, dest="selfplay_every",
                    help="Add checkpoint to self-play pool every N episodes")

    # play
    pl = sub.add_parser("play", help="Connect to contest server")
    pl.add_argument("--url",   type=str, required=True)
    pl.add_argument("--model", type=str, default="model.pt")
    pl.add_argument("--mcts",  action="store_true", help="Use MCTS on top of RL model")

    args = parser.parse_args()

    if args.mode == "sim":
        run_sim(args)
    elif args.mode == "train":
        run_train(args)
    elif args.mode == "play":
        run_play(args)


if __name__ == "__main__":
    main()
