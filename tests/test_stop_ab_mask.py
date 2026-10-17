"""No AnyLogic required. Run: python tests/test_stop_ab_mask.py"""
from pathlib import Path
import importlib.util
from types import SimpleNamespace
import numpy as np

COLLECTOR = Path(__file__).resolve().parents[1] / 'thesis' / 'mappo' / 'collector_stop_ab.py'
spec = importlib.util.spec_from_file_location('collector_stop_ab', COLLECTOR)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
EventCollector = module.EventCollector

class Policy:
    def __init__(self, n=5):
        self.roles = {
            'routing': SimpleNamespace(actor=SimpleNamespace(action_dim=n)),
            'dispatch_front': SimpleNamespace(actor=SimpleNamespace(action_dim=22)),
        }
        self.cfg = SimpleNamespace(two_fleets=False)
    def role_for(self, agent_id, physical, two_fleets):
        return 'routing'
    def act(self, role, obs, state, greedy=False):
        # Prefers Stop to test whether the SAME mask is used for sampling.
        mask = obs['action_mask']
        action = next(i for i in [0, 1, 2, 3, 4] if mask[i] > 0.5)
        return action, -0.01, 0.5
    def value(self, role, state):
        return 0.4

class Env:
    def __init__(self, allowed):
        self.allowed = np.asarray(allowed, dtype=np.float32)
        self.routing_agent_death = False
        self.dispatching_agent_death = False
        self.statistics = {'taken_invalid': 0}
    def reset(self, seed=None):
        self.agents = ['0']
        self.agent_selection = '0'
        self.idx = 0
        self.action = None
    def observe_dict(self):
        return dict(agvs=np.ones((8, 19)), stat=np.ones((22, 19)), action_mask=self.allowed.copy())
    def last(self):
        return self.observe_dict(), (1.0 if self.idx == 1 else 0.0), self.idx == 1, {}
    def get_mappo_state(self):
        return dict(ego=0, agvs=np.ones((8, 19)), stat=np.ones((22, 19)))
    def get_mappo_time(self):
        return self.idx * 2.0
    def step(self, action):
        self.action = action
        self.idx = 1

base = EventCollector(Policy(), Env([1, 1, 0, 0, 1]), 'baseline')
B = EventCollector(Policy(), Env([1, 1, 0, 0, 1]), 'forbid_when_move')
obs = base.env.observe_dict()
assert base._prepare(obs, 'routing')['action_mask'].tolist() == [1, 1, 0, 0, 1]
assert B._prepare(obs, 'routing')['action_mask'].tolist() == [0, 1, 0, 0, 1]
assert obs['action_mask'][0] == 1.0, 'Must not mutate source observation'
assert B._prepare(base.env.observe_dict(), 'routing')['action_mask'].sum() == 2
assert B._prepare(dict(obs, action_mask=np.array([1, 0, 0, 0, 0])), 'routing')['action_mask'].tolist() == [1, 0, 0, 0, 0]
assert B._prepare(dict(obs, action_mask=np.array([0, 1, 0, 0, 0])), 'routing')['action_mask'].tolist() == [0, 1, 0, 0, 0]
assert B._prepare(dict(obs, action_mask=np.array([1, 0, 0, 0, 0] + [0] * 17)), 'dispatch_front')['action_mask'][0] == 1

for mode, original, selected, expected_masked, stopped_with_move in [
    ('baseline', [1, 1, 0, 0, 1], 0, 0, 1),
    ('forbid_when_move', [1, 1, 0, 0, 1], 1, 1, 0),
    ('forbid_when_move', [1, 0, 0, 0, 0], 0, 0, 0),
]:
    env = Env(original)
    collector = EventCollector(Policy(), env, mode)
    segments, meta = collector.episode(seed=43)
    assert env.action == selected, (mode, selected, env.action)
    assert meta['closed_transitions'] == 1
    assert segments[0][0]['action'] == selected
    assert segments[0][0]['observation']['action_mask'][selected] > 0.5
    assert meta['action_diagnostics']['routing_stop_masked_decisions'] == expected_masked
    assert meta['action_diagnostics']['routing_stop_with_move_available'] == stopped_with_move
    assert meta['action_diagnostics']['selected_mask_violations'] == 0
print('PASS: A/B Stop mask (baseline, movement available, no movement, dispatch unaffected, rollout storage)')
