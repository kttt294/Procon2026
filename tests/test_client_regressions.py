from types import SimpleNamespace
from unittest.mock import Mock, patch

from client.http_client import ContestClient
from env.models import AgentAction, AgentState, Cell, DayOrder, MapData, MatchConfig, Spot
from env.simulator import HexaUdonSimulator
from main import run_play


def test_retry_caps_http_timeout_to_remaining_deadline():
    client = ContestClient('http://test')
    response = Mock()
    response.json.return_value = {'status': 'valid'}
    client.session.post = Mock(return_value=response)
    with patch('client.http_client.time.monotonic', return_value=10.4):
        result = client.submit_with_retry(1, [], [], 1000, 10.)
    assert result['status'] == 'valid'
    assert 0 < client.session.post.call_args.kwargs['timeout'] <= .6


def test_retry_is_bounded_and_reports_the_actual_accepted_fallback():
    client = ContestClient('http://test')
    original = [DayOrder(0, [AgentAction('move', 2)])]
    client.submit_orders = Mock(side_effect=[{'status': 'invalid'}, {'status': 'valid'}])
    with patch('client.http_client.time.monotonic', return_value=10.):
        result = client.submit_with_retry(1, original, [], 1000, 10.)
    assert result['_accepted_orders'] == []
    assert client.submit_orders.call_count == 2
    client.submit_orders.reset_mock()
    with patch('client.http_client.time.monotonic', return_value=12.):
        assert client.submit_with_retry(1, original, [], 1000, 10.)['status'] == 'failed'
    client.submit_orders.assert_not_called()


def test_play_submits_before_advanced_planner_and_survives_failure():
    cfg = MatchConfig(8, 8, 1, [20], 2, 3, 7, 20)
    mp = MapData([Cell(i, 0) for i in range(64)], [Spot(1, 1, 3)])
    sim = HexaUdonSimulator(cfg, mp)
    state = sim.reset([AgentState(0, 0, 0, 20), AgentState(1, 1, 8), AgentState(2, 1, 9)])
    events = []
    client = Mock()
    client.get_match_config.return_value = (cfg, mp, state.my_agents)
    client.get_day_state.return_value = state
    def submit(*args, **kwargs):
        events.append('submit')
        return {'status': 'valid'}
    client.submit_with_retry.side_effect = submit
    client.submit_orders.side_effect = submit
    def advanced(*args):
        assert events == ['submit']
        events.append('plan')
        raise RuntimeError('planner failed')
    with patch('client.http_client.ContestClient', return_value=client), \
         patch('main.LookaheadPlanner') as planner:
        planner.return_value.plan.side_effect = advanced
        run_play(SimpleNamespace(url='http://test', model=None, mcts=False))
    assert events[:2] == ['submit', 'plan']


def test_play_does_not_submit_invalid_advanced_orders():
    cfg = MatchConfig(8, 8, 1, [20], 2, 3, 7, 20)
    mp = MapData([Cell(i, 0) for i in range(64)], [Spot(1, 1, 3)])
    state = HexaUdonSimulator(cfg, mp).reset([AgentState(0, 0, 0, 20)])
    client = Mock()
    client.get_match_config.return_value = (cfg, mp, state.my_agents)
    client.get_day_state.return_value = state
    client.submit_with_retry.return_value = {'status': 'valid'}
    with patch('client.http_client.ContestClient', return_value=client), \
         patch('main.LookaheadPlanner') as planner:
        planner.return_value.plan.return_value = [DayOrder(0, [AgentAction('move', 99)])]
        run_play(SimpleNamespace(url='http://test', model=None, mcts=False))
    assert client.submit_with_retry.call_args.kwargs['orders'][0].actions[0].direction == 2
    client.submit_orders.assert_not_called()
