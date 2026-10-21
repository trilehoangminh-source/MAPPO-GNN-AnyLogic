"""Offline tests for V3.1 read-only route trace. No AnyLogic/Alpyne needed.
Run from project root: python tests/test_uoe_v3_trace_offline.py
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np
from thesis.mappo.routing_trace_uoe_v3 import DirectedRouteDistances, RoutingDecisionTrace


def graph():
    # GraphSpec.ids are REAL IDs, while arcs use 0-based sorted node INDICES.
    return SimpleNamespace(ids=(10, 20, 30, 40), xy=((0.,0.),(1.,0.),(2.,0.),(1.,1.)),
                           arcs=((0,1), (1,2), (1,3), (3,2)))


def synthetic_obs():
    agvs = np.zeros((3, 20), dtype=np.float32)
    agvs[0, 4:6] = (0., 0.)
    agvs[0, 6:8] = (2., 0.)
    agvs[0, 8:16] = (1.,0., 1.,1., 5.,5., -5.,-5.)
    agvs[0, 0] = 1.0
    agvs[1, 0] = 1.0
    agvs[1, 4:6] = (1., 0.)
    return {'agvs':agvs, 'action_mask':np.array([1,1,0,0,0],np.float32),
            'routing_guide_bias':np.array([0,1.5,0,0,0],np.float32)}


class StubEnv:
    def __init__(self):
        self.statistics = {'block':0, 'liveLock':0, 'jobDone':0, 'wrongTarget':0, 'targetReached':0}
        self.time = 0.
    def get_mappo_time(self):
        return self.time


class StubModel:
    def __init__(self):
        self.actions = []
    def act(self, role, obs, state, greedy=False):
        self.actions.append(role)
        return (1, -0.25, 3.0) if role == 'routing' else (0, -0.5, 0.2)


def test_distances():
    g = DirectedRouteDistances(graph())
    assert g.match((0., 0.)) == 10
    assert g.match((1., 1.)) == 40
    assert g.match((0.005, 0.)) is None
    assert g.distance_map(30)[10] == 2.0
    assert g.distance_map(10) == {10:0.0}  # directed! no backwards move
    assert (10, 20) in g.edges and (20, 10) not in g.edges
    assert g.distance_map(999) == {999:0.0}


def test_noninvasive():
    env, model = StubEnv(), StubModel()
    tracer = RoutingDecisionTrace(graph(), local_radius=1.1)
    obs = synthetic_obs()
    result = model.act('routing', obs, {'ego':7}, greedy=True)
    with tempfile.TemporaryDirectory() as work:
        outfile = Path(work)/'trace.jsonl'
        with tracer.attach(model, env, outfile):
            assert model.act('routing', obs, {'ego':7}, greedy=True) == result
            assert model.act('dispatch_front', obs, {'ego':7}) == (0, -.5, .2)
            env.time = 1.5
            env.statistics['block'] += 2
            assert model.act('routing', obs, {'ego':7}) == result
        rows = [json.loads(x) for x in outfile.read_text().splitlines()]
        assert len(rows) == 2 and rows[0]['action'] == 1
        assert rows[0]['current_node'] == 10
        assert rows[0]['target_node'] == 30
        assert rows[0]['next_node'] == 20
        assert rows[0]['selected_is_guided'] is True
        assert rows[0]['remaining_change'] == -1.0
        assert rows[0]['other_agvs_last_node_nearby'] == 1
        assert rows[0]['stats_increase_since_previous_routing_observation']['block'] is None
        assert rows[1]['stats_increase_since_previous_routing_observation']['block'] == 2
        assert tracer.report()['routing_decisions'] == 2
        assert tracer.report()['guide_follow_rate'] == 1.0
        # More importantly: callback restored, and new act is still unchanged.
        assert model.act('routing', obs, {'ego':7}) == result


def test_restore_even_if_error():
    env, model = StubEnv(), StubModel()
    tracer = RoutingDecisionTrace(graph())
    original = model.act
    with tempfile.TemporaryDirectory() as tmp:
        try:
            with tracer.attach(model, env, Path(tmp)/'one.jsonl'):
                raise RuntimeError('fake failure')
        except RuntimeError:
            pass
        assert model.act == original


def test_comparator():
    sys.path.insert(0, str(ROOT/'scripts'))
    from verify_B3_UOE_v3_noninvasive import compare
    x = {1001:{k: (5 if k != 'delivered_reward' else 10.5)
               for k in ('delivered_reward','jobDone','targetReached','wrongTarget','block','liveLock')}}
    assert all(r[-1] for r in compare(x, x))
    y = {1001:dict(x[1001], block=6)}
    assert sum(not r[-1] for r in compare(x, y)) == 1


def test_analyzer():
    sys.path.insert(0, str(ROOT/'scripts'))
    from analyze_B3_UOE_v3_trace import summarize
    env, model = StubEnv(), StubModel()
    tracer = RoutingDecisionTrace(graph())
    obs = synthetic_obs()
    with tempfile.TemporaryDirectory() as d:
        outfile = Path(d)/'trace.jsonl'
        with tracer.attach(model, env, outfile):
            model.act('routing', obs, {'ego':2})
            env.statistics['block'] += 1
            model.act('routing', obs, {'ego':2})
        summary = summarize(json.loads(s) for s in outfile.read_text().splitlines())
        assert summary['total_decisions'] == 2
        assert summary['guide_follow_rate'] == 1.0
        assert summary['intervals_with_block_increase'] == 1
        assert summary['hotspot_nodes'][0]['node'] == '10'


if __name__ == '__main__':
    test_distances()
    test_noninvasive()
    test_restore_even_if_error()
    test_analyzer()
    test_comparator()
    print('PASS V3.1 offline tests (directed routes, trace transparency, counters, restoration, summary)')
