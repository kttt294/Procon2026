from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from env.models import AgentState, Cell, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator
from rl.actor_critic import ActorCritic
from rl.mappo import RolloutBuffer, MAPPOTrainer
from rl.curriculum import CurriculumEngine
from env.scoring import Score
from env.validator import validate_orders
from env.map_generator import generate_scenario, MapGenConfig, MatchGenConfig
from rl.selfplay import SelfPlayPool
from strategy.mcts import MCTSNode, MCTSPlanner


def setup_game(n_spots=1):
    cfg = MatchConfig(8, 8, 4, [20] * 4, 2, 3, 7, 20)
    mp = MapData([Cell(i, 0) for i in range(64)], [Spot(i + 1, i, 3) for i in range(n_spots)])
    sim = HexaUdonSimulator(cfg, mp)
    state = sim.reset([AgentState(0, 0, 0, 20), AgentState(1, 1, 62), AgentState(2, 1, 63)])
    return cfg, mp, sim, state


def test_gae_terminal_return():
    buffer = RolloutBuffer()
    for v, done in [(1., False), (10., True)]:
        buffer.add(SimpleNamespace(value=torch.tensor(v), reward=0., done=done))
    assert buffer.compute_returns(1., 1.)[0] == [0., 0.]


def test_ppo_evaluates_saved_action_without_sampling():
    cfg, mp, _, state = setup_game()
    model = ActorCritic(hidden=16, max_width=8, max_height=8)
    actions, old_logs, _, _ = model.get_action_and_value(state, mp, cfg, [0])
    with patch('torch.distributions.Categorical.sample', side_effect=AssertionError('must not sample')):
        chosen, new_logs, _, _ = model.get_action_and_value(state, mp, cfg, [0], actions=actions)
    assert chosen == actions
    assert torch.allclose((new_logs - old_logs).exp(), torch.ones_like(old_logs))


def test_mcts_backup_rewards_once():
    _, _, _, state = setup_game()
    root = MCTSNode(state, None, [], 0, 0)
    parent = MCTSNode(state, root, [], 100, 1)
    leaf = MCTSNode(state, parent, [], 10, 2)
    planner = object.__new__(MCTSPlanner)
    planner._backup(leaf, 0.)  # terminal continuation value
    assert leaf.q == 10
    assert parent.q == pytest.approx(109.9)


def test_mcts_expansion_does_not_change_simulator_history():
    cfg, mp, sim, state = setup_game()
    model = ActorCritic(hidden=16, max_width=8, max_height=8)
    planner = MCTSPlanner(cfg, mp, sim, model, beam_width=2)
    root = MCTSNode(state, None, [], 0, 0)
    planner._expand(root)
    assert sim.traffic._history == []
    assert root.children
    assert all(len(child.traffic._history) == 1 for child in root.children)
    assert all(child.traffic is not sim.traffic for child in root.children)
    with patch.object(model, 'forward', side_effect=AssertionError('deadline expired')):
        assert planner._expand(root, deadline=0) == []


def test_selfplay_moves_and_consumes_shared_steps_and_fuel():
    cfg, mp, sim, state = setup_game()
    state.steps_left = 2
    opponents = [AgentState(100, 0, 0, 20), AgentState(101, 0, 0, 20)]
    model = SimpleNamespace(get_action_and_value=lambda *a, **kw: ([0, 0], None, None, None))
    cells, _ = SelfPlayPool.simulate_day(model, opponents, state, mp, cfg, sim.grid)
    assert cells == [1, 0]
    assert [a.fuel for a in opponents] == [19, 20]


def test_more_than_30_spots_preserves_stay_and_all_targets():
    cfg, mp, sim, state = setup_game(40)
    model = ActorCritic(hidden=16, max_width=8, max_height=8)
    logits, _ = model.forward(state, mp, cfg, [0])
    assert logits[0].shape == (1, 41)
    assert torch.isfinite(logits[0]).all()
    planner = MCTSPlanner(cfg, mp, sim, model)
    orders = planner._build_orders(state, [0], [40])
    assert orders[0].actions == []


def test_curriculum_uses_tiebreaks_and_does_not_count_ties_as_wins():
    curriculum = CurriculumEngine()
    curriculum.record(Score(2, 3, 5), Score(2, 3, 5))
    curriculum.record(Score(2, 2, 100), Score(2, 3, 5))
    curriculum.record(Score(2, 4, 5), Score(2, 3, 100))
    assert curriculum.win_rate == pytest.approx(1 / 3)


def test_training_update_uses_rollout_actions_and_stays_finite():
    torch.manual_seed(9)
    cfg, mp, sim, state = setup_game()
    trainer = MAPPOTrainer(max_width=8, max_height=8, log_dir=None)
    result = trainer._collect_episode(cfg, mp, state.my_agents)
    assert result['score'].unique_series >= 0
    for tr in trainer.buffer.transitions:
        ids = [a.id for a in tr.state.patrol_agents()]
        orders = trainer._actions_to_orders(tr.state, mp, cfg, sim, ids, tr.actions)
        assert validate_orders(orders, tr.state, mp, sim.grid)[0]
    before = trainer.model.critic_head.weight.detach().clone()
    evaluate = trainer.model.get_action_and_value
    def saved_only(*args, **kwargs):
        assert 'actions' in kwargs
        return evaluate(*args, **kwargs)
    with patch.object(trainer.model, 'get_action_and_value', side_effect=saved_only):
        losses = trainer._update()
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert not torch.equal(before, trainer.model.critic_head.weight)


def test_mcts_converts_shaped_critic_to_raw_continuation():
    cfg, mp, sim, state = setup_game()
    model = ActorCritic(hidden=16, max_width=8, max_height=8)
    planner = MCTSPlanner(cfg, mp, sim, model)
    node = MCTSNode(state, None, [], 0, 0)
    with patch.object(model, 'forward', return_value=([], torch.tensor(50.))):
        assert planner._evaluate(node) == 40.  # one hex away: phi=-10


@pytest.mark.parametrize('width,n_agents,days,n_spots', [(8, 3, 4, 40), (32, 8, 10, 80)])
def test_rl_and_mcts_orders_at_contest_boundaries(width, n_agents, days, n_spots):
    cfg, mp, agents = generate_scenario(
        91, MapGenConfig(width, width, n_spots=n_spots, n_series=n_spots),
        MatchGenConfig(total_days=days), n_agents=n_agents, n_patrol=n_agents-1)
    sim = HexaUdonSimulator(cfg, mp)
    state = sim.reset(agents)
    model = ActorCritic(hidden=16)
    planner = MCTSPlanner(cfg, mp, sim, model, time_budget_ms=20, beam_width=2)
    first_orders = planner.plan(state)
    assert validate_orders(first_orders, state, mp, sim.grid)[0]
    assert sim.traffic._history == []
    while not sim.is_done(state):
        ids = [a.id for a in state.patrol_agents()]
        with torch.no_grad():
            actions, _, _, _ = model.get_action_and_value(state, mp, cfg, ids, deterministic=True)
        orders = planner._build_orders(state, ids, actions)
        assert validate_orders(orders, state, mp, sim.grid)[0]
        state, _ = sim.apply_day(state, orders)
