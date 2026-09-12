import random
import heapq

import pytest

from env.hex_grid import HexGrid
from env.models import AgentAction, AgentState, Cell, DayOrder, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator
from env.validator import validate_orders
from env.map_generator import MapGenConfig, generate_scenario, generate_random_scenario
from pathfinding.astar import find_path
from strategy.greedy import GreedyPlanner
from strategy.lookahead import LookaheadPlanner
from strategy.agent_selector import AgentSelector
from env.scoring import Score


def scenario(spots=None, steps=20):
    cfg = MatchConfig(8, 8, 4, [steps] * 4, 2, 3, 7, 20)
    mp = MapData([Cell(i, 0) for i in range(64)], spots or [Spot(1, 1, 3)])
    sim = HexaUdonSimulator(cfg, mp)
    state = sim.reset([AgentState(0, 0, 0, 20), AgentState(1, 1, 8), AgentState(2, 1, 9)])
    return cfg, mp, sim, state


def test_astar_reconstruction_matches_cost_and_budget():
    result = find_path(HexGrid(8, 8), dict.fromkeys(range(64), 0), {}, 57, 52, 6, 3)
    assert result.reachable
    assert result.total_steps == len(result.actions) * 2 <= 6
    assert result.total_fuel == len(result.actions) <= 3


def test_astar_keeps_slower_fuel_saving_paths():
    terrain = dict.fromkeys(range(64), 0)
    terrain[53] = 3
    result = find_path(HexGrid(8, 8), terrain, {}, 58, 46, 30, 5)
    assert result.reachable
    assert (result.total_steps, result.total_fuel) == (10, 5)


@pytest.mark.parametrize('planner_type', [GreedyPlanner, LookaheadPlanner])
def test_adjacent_spot_is_collected_each_day(planner_type):
    cfg, mp, sim, state = scenario()
    planner = planner_type(cfg, mp, sim)
    for day in range(1, 5):
        orders = planner.plan(state)
        assert validate_orders(orders, state, mp, sim.grid)[0]
        state, _ = sim.apply_day(state, orders)
        assert state.total_udon >= day


def test_reposition_respects_remaining_fuel():
    cfg, mp, sim, state = scenario([Spot(1, 1, 3), Spot(2, 2, 3)])
    patrol = state.my_agents[0]
    patrol.fuel = 1
    planner = LookaheadPlanner(cfg, mp, sim)
    actions = planner._append_reposition(patrol, [1], [AgentAction('move', 2)], 20, state)
    assert validate_orders([DayOrder(0, actions)], state, mp, sim.grid)[0]


@pytest.mark.parametrize('orders', [
    [DayOrder(0, [AgentAction('typo')])],
    [DayOrder(0, []), DayOrder(0, [])],
    [DayOrder(0, [AgentAction('move', 2.0)])],
])
def test_validator_rejects_malformed_orders(orders):
    _, mp, sim, state = scenario()
    assert not validate_orders(orders, state, mp, sim.grid)[0]


def test_generated_maps_follow_start_spot_and_inventory_rules():
    cfg, mp, agents = generate_scenario(4, MapGenConfig(8, 8, n_spots=40, n_series=40,
                                                       max_inventory=8), n_agents=3)
    assert len(mp.spots) == 40
    assert all(mp.cell_map[s.cell_id].terrain == 0 and 1 <= s.max_inventory <= 3 for s in mp.spots)
    assert all(mp.cell_map[a.cell].terrain == 0 and a.cell not in mp.spot_map for a in agents)


@pytest.mark.parametrize('seed', range(10))
def test_planners_return_valid_orders_on_contest_maps(seed):
    cfg, mp, agents = generate_random_scenario(seed)
    for planner_type in (GreedyPlanner, LookaheadPlanner):
        sim = HexaUdonSimulator(cfg, mp)
        state = sim.reset(agents)
        planner = planner_type(cfg, mp, sim)
        while not sim.is_done(state):
            orders = planner.plan(state)
            ok, errors = validate_orders(orders, state, mp, sim.grid)
            assert ok, (planner_type.__name__, seed, state.day, errors)
            state, _ = sim.apply_day(state, orders)


def test_astar_matches_resource_state_oracle():
    rng = random.Random(17)
    grid = HexGrid(8, 8)
    for _ in range(60):
        terrain = {i: rng.choice([0, 0, 1, 2, 3]) for i in range(64)}
        src, dst = rng.sample(range(64), 2)
        terrain[src] = terrain[dst] = 0
        traffic = {i: rng.randrange(3) for i, t in terrain.items() if t == 3}
        budget, fuel = rng.randint(2, 30), rng.randint(1, 12)
        cfg = MatchConfig(8, 8, 4, [budget] * 4, 2, 3, 7, fuel)
        mp = MapData([Cell(i, t) for i, t in terrain.items()], [])
        sim = HexaUdonSimulator(cfg, mp)
        queue, visited, optimum = [(0, 0, src)], set(), None
        while queue:
            steps, used_fuel, cell = heapq.heappop(queue)
            if (cell, used_fuel) in visited:
                continue
            visited.add((cell, used_fuel))
            if cell == dst:
                optimum = steps
                break
            ns, nf = steps + sim._step_cost(cell, traffic), used_fuel + sim._fuel_cost(cell)
            if ns <= budget and nf <= fuel:
                for _, nbr in grid.neighbors(cell):
                    if terrain[nbr] != 2:
                        heapq.heappush(queue, (ns, nf, nbr))
        result = find_path(grid, terrain, traffic, src, dst, budget, fuel)
        assert result.reachable == (optimum is not None)
        if result.reachable:
            cell, steps, used_fuel = src, 0, 0
            for action in result.actions:
                steps += sim._step_cost(cell, traffic)
                used_fuel += sim._fuel_cost(cell)
                cell = grid.neighbor_in_dir(cell, action.direction)
                assert cell is not None and terrain[cell] != 2
            assert cell == dst
            assert result.total_steps == steps == optimum
            assert result.total_fuel == used_fuel <= fuel


def test_selector_includes_all_patrol_and_uses_score_tiebreaks():
    cfg, mp, sim, state = scenario()
    selector = AgentSelector(cfg, mp, sim)
    candidates = selector._generate_candidates([0, 1, 2], [0, 8, 9], 3)
    assert any(patrols == 3 and supplies == 0 for _, patrols, supplies in candidates)
    result = selector.select([0, 1, 2], [0, 8, 9], 20)
    assert isinstance(result.score, Score)
