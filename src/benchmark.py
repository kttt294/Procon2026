"""
Benchmark runner: compare strategies across random maps.

Usage:
    python src/benchmark.py                        # 100 games, greedy vs lookahead
    python src/benchmark.py --games 500 --seed 0
    python src/benchmark.py --games 50 --verbose

Output table:
    Strategy   Avg Series  Avg Daily   Avg Udon     Avg ms   Win Rate
    greedy          2.31       4.12      18.4        1.2    baseline
    lookahead       3.78       6.90      31.1        8.4      73.0%
"""
from __future__ import annotations

import argparse
import sys
import os
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from env.map_generator import generate_random_scenario
from env.models import AgentState, MapData, MatchConfig
from env.scoring import Score, compute_score, wins_over
from env.simulator import HexaUdonSimulator
from env.validator import validate_orders
from strategy.greedy import GreedyPlanner
from strategy.lookahead import LookaheadPlanner
from strategy.planner import BasePlanner


# ------------------------------------------------------------------ #
# Data                                                                 #
# ------------------------------------------------------------------ #

@dataclass
class GameResult:
    strategy:  str
    score:     Score
    time_ms:   float
    day_times: List[float] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    strategy:          str
    n_games:           int
    avg_unique_series: float
    avg_daily_series:  float
    avg_udon:          float
    avg_time_ms:       float
    win_rate:          Optional[float] = None   # vs the first (baseline) strategy
    tie_rate:          Optional[float] = None
    avg_day_ms:        float = 0.0
    max_day_ms:        float = 0.0


class _PolicyPlanner(BasePlanner):
    def __init__(self, cfg, map_data, sim, trainer):
        super().__init__(cfg, map_data, sim)
        self.trainer = trainer

    def plan(self, state):
        import torch
        ids = [a.id for a in state.patrol_agents()]
        with torch.no_grad():
            actions, _, _, _ = self.trainer.model.get_action_and_value(
                state, self.map, self.cfg, ids, deterministic=True)
        return self.trainer._actions_to_orders(state, self.map, self.cfg, self.sim, ids, actions)


# ------------------------------------------------------------------ #
# Public API                                                           #
# ------------------------------------------------------------------ #

def run_benchmark(
    n_games:    int  = 100,
    seed:       int  = 42,
    strategies: Optional[List[str]] = None,
    verbose:    bool = False,
    model_path: Optional[str] = None,
    mcts_ms:    int = 100,
) -> Dict[str, BenchmarkResult]:
    """
    Run n_games random-map games for each strategy.

    strategies: ["greedy", "lookahead"] (default). The first entry is the baseline
                used to compute win_rate for subsequent strategies.
    Returns a dict mapping strategy name → BenchmarkResult.
    """
    if n_games < 1:
        raise ValueError("n_games must be positive")
    strategies = strategies or ["greedy", "lookahead"]
    trainer = None
    if any(name in ("rl", "mcts") for name in strategies):
        if not model_path:
            raise ValueError("RL/MCTS benchmark requires --model")
        from rl.mappo import MAPPOTrainer
        trainer = MAPPOTrainer(log_dir=None)
        trainer.load(model_path)
        trainer.model.eval()
    all_results: Dict[str, List[GameResult]] = {s: [] for s in strategies}

    for game_idx in range(n_games):
        game_seed = seed + game_idx
        cfg, map_data, agents = generate_random_scenario(seed=game_seed)
        sim = HexaUdonSimulator(cfg, map_data)

        for strat_name in strategies:
            planner = _make_planner(strat_name, cfg, map_data, sim, trainer, mcts_ms)
            if trainer is not None:
                trainer.model.set_fuel_max(cfg.fuel_max)

            t0    = time.perf_counter()
            state = sim.reset(deepcopy(agents))
            day_times = []
            while not sim.is_done(state):
                day_start = time.perf_counter()
                orders = planner.plan(state)
                day_times.append((time.perf_counter() - day_start) * 1000)
                ok, errors = validate_orders(orders, state, map_data, sim.grid)
                if not ok:
                    raise ValueError(f"{strat_name}, seed={game_seed}, day={state.day}: {errors}")
                state, _ = sim.apply_day(state, orders)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            all_results[strat_name].append(
                GameResult(strat_name, compute_score(state), elapsed_ms, day_times)
            )

        if verbose and (game_idx + 1) % 10 == 0:
            print(f"  {game_idx + 1}/{n_games} games done", flush=True)

    return _summarise(all_results, strategies)


# ------------------------------------------------------------------ #
# Printing                                                             #
# ------------------------------------------------------------------ #

def print_results(results: Dict[str, BenchmarkResult]) -> None:
    header = f"{'Strategy':<12} {'Avg Series':>10} {'Avg Daily':>10} {'Avg Udon':>10} {'Game ms':>8} {'Day ms':>8} {'Max day':>8} {'Win':>9} {'Tie':>9}"
    print()
    print(header)
    print("-" * len(header))
    for r in results.values():
        win_str = f"{r.win_rate:.1%}" if r.win_rate is not None else "baseline"
        tie_str = f"{r.tie_rate:.1%}" if r.tie_rate is not None else "-"
        print(
            f"{r.strategy:<12} {r.avg_unique_series:>10.2f} "
            f"{r.avg_daily_series:>10.2f} {r.avg_udon:>10.1f} "
            f"{r.avg_time_ms:>8.1f} {r.avg_day_ms:>8.1f} {r.max_day_ms:>8.1f} {win_str:>9} {tie_str:>9}"
        )
    print()


# ------------------------------------------------------------------ #
# Internals                                                            #
# ------------------------------------------------------------------ #

def _make_planner(name: str, cfg: MatchConfig, map_data: MapData, sim: HexaUdonSimulator,
                  trainer=None, mcts_ms: int = 100):
    if name == "greedy":
        return GreedyPlanner(cfg, map_data, sim)
    if name == "lookahead":
        return LookaheadPlanner(cfg, map_data, sim)
    if name == "rl" and trainer is not None:
        return _PolicyPlanner(cfg, map_data, sim, trainer)
    if name == "mcts" and trainer is not None:
        from strategy.mcts import MCTSPlanner
        return MCTSPlanner(cfg, map_data, sim, trainer.model, time_budget_ms=mcts_ms)
    raise ValueError(f"Unknown strategy or missing model: {name!r}")


def _summarise(
    all_results: Dict[str, List[GameResult]],
    strategies:  List[str],
) -> Dict[str, BenchmarkResult]:
    summary: Dict[str, BenchmarkResult] = {}
    baseline: Optional[List[Score]] = None

    for i, name in enumerate(strategies):
        results = all_results[name]
        scores  = [r.score for r in results]
        n       = len(scores)

        avg_unique = sum(s.unique_series    for s in scores) / n
        avg_daily  = sum(s.daily_series_sum for s in scores) / n
        avg_udon   = sum(s.total_udon       for s in scores) / n
        avg_time   = sum(r.time_ms          for r in results) / n

        win_rate = None
        tie_rate = None
        day_times = [t for r in results for t in r.day_times]
        if baseline is not None:
            wins     = sum(1 for s, b in zip(scores, baseline) if wins_over(s, b))
            win_rate = wins / n
            tie_rate = sum(s == b for s, b in zip(scores, baseline)) / n

        if i == 0:
            baseline = scores

        summary[name] = BenchmarkResult(
            strategy          = name,
            n_games           = n,
            avg_unique_series = avg_unique,
            avg_daily_series  = avg_daily,
            avg_udon          = avg_udon,
            avg_time_ms       = avg_time,
            win_rate          = win_rate,
            tie_rate          = tie_rate,
            avg_day_ms        = sum(day_times) / len(day_times) if day_times else 0.0,
            max_day_ms        = max(day_times, default=0.0),
        )

    return summary


# ------------------------------------------------------------------ #
# CLI                                                                  #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark HEXA UDON strategies")
    parser.add_argument("--games",      type=int,   default=100)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--strategies", nargs="+",  default=["greedy", "lookahead"])
    parser.add_argument("--verbose",    action="store_true")
    parser.add_argument("--model", type=str, help="Checkpoint required for rl or mcts")
    parser.add_argument("--mcts-ms", type=int, default=100, help="Search budget per day")
    args = parser.parse_args()

    print(f"Benchmarking {args.strategies} over {args.games} games (seed={args.seed})...")
    results = run_benchmark(
        n_games    = args.games,
        seed       = args.seed,
        strategies = args.strategies,
        verbose    = args.verbose,
        model_path = args.model,
        mcts_ms    = args.mcts_ms,
    )
    print_results(results)
