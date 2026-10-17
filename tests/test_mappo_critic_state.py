"""Small, dependency-light offline checks for centralized-critic input.

Copy into thesis/tests and run from project root:
    python tests/test_mappo_critic_state.py
Uses AST to test Matrix methods without starting AnyLogic or importing Ray.
"""
from __future__ import annotations
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if (ROOT / 'thesis' / 'envs' / 'matrix.py').exists():
    MATRIX = ROOT / 'thesis' / 'envs' / 'matrix.py'
    NETWORKS = ROOT / 'thesis' / 'mappo' / 'networks.py'
else:
    # Allows running this file directly in the same directory as the supplied files.
    MATRIX = Path(__file__).parent / 'matrix_critic_fresh.py'
    NETWORKS = Path(__file__).parent / 'networks.py'

source = ast.parse(MATRIX.read_text('utf-8'))
klass = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == 'Matrix')
methods = [n for n in klass.body if isinstance(n, ast.FunctionDef)
           and n.name in {'_capture_mappo_state', 'get_mappo_state', 'get_mappo_station_source'}]
wrapper = ast.Module(body=[ast.ClassDef(name='MatrixUnderTest', bases=[], keywords=[],
                                    body=methods, decorator_list=[])], type_ignores=[])
ns = {}
exec(compile(ast.fix_missing_locations(wrapper), str(MATRIX), 'exec'), ns)
MatrixUnderTest = ns['MatrixUnderTest']

spec = importlib.util.spec_from_file_location('network_for_critic_state_test', NETWORKS)
network_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(network_module)
CentralCritic = network_module.CentralCritic

class FakeRun:
    def __init__(self): self.time = 1.0
    def get_state(self): return None, {'model_time': self.time}

class FakeObs:
    def __init__(self, caller, obs): self.caller = caller; self.obs = obs

m = MatrixUnderTest()
m.fleetsize, m.max_fleetsize = 8, 20
m.agent_hive = SimpleNamespace(stations={i: None for i in range(27)},
    statistics={'n_agv':8, 'n_stat':22, 'rout_len':18, 'disp_len':53})
m.run = FakeRun()
m._mappo_station_cache = None
routing = np.zeros((8,18), dtype=np.float32)
routing[:,0] = 1
routing[:,3] = np.linspace(0.1,0.8,8)

# Original Tobias Routing has NO station rows.
m._capture_mappo_state(FakeObs(3,routing))
missing = m.get_mappo_state()
assert m.get_mappo_station_source() == 'missing'
assert missing['agvs'].shape == (20,53) and missing['stat'].shape == (27,53)
assert missing['ego'] == 3 and np.count_nonzero(missing['stat']) == 0

# Original Tobias Dispatching contains 22 registered station rows.
m.run.time = 360.1
dispatch = np.zeros((30,53),dtype=np.float32)
dispatch[:8,:18] = routing
dispatch[8:,0] = np.arange(1,23)
m._capture_mappo_state(FakeObs(1007,dispatch))
fresh_dispatch = m.get_mappo_state()
assert m.get_mappo_station_source() == 'fresh_dispatch'
assert fresh_dispatch['ego']==7 and np.count_nonzero(fresh_dispatch['stat'])==22

# Legacy Routing afterwards uses the stale station cache.
m.run.time = 700.0
m._capture_mappo_state(FakeObs(4,routing))
assert m.get_mappo_station_source() == 'cached'
assert np.array_equal(m.get_mappo_state()['stat'], fresh_dispatch['stat'])

# New Routing export includes station rows at EACH routing event.
new_routing = np.zeros((30,18),dtype=np.float32)
new_routing[:8] = routing
new_routing[8:,0] = np.arange(1,23)*10
m.run.time = 720.0
m._capture_mappo_state(FakeObs(5,new_routing))
fresh_routing = m.get_mappo_state()
assert m.get_mappo_station_source() == 'fresh_routing'
assert np.count_nonzero(fresh_routing['stat']) == 22
assert fresh_routing['ego'] == 5
assert not np.array_equal(fresh_routing['stat'], fresh_dispatch['stat'])

# The ORIGINAL CentralCritic architecture works unchanged with both state modes.
critic = CentralCritic(53,hidden=32)
batch = {
    'agvs':torch.tensor(np.stack([missing['agvs'],fresh_dispatch['agvs'],fresh_routing['agvs']])),
    'stat':torch.tensor(np.stack([missing['stat'],fresh_dispatch['stat'],fresh_routing['stat']])),
    'ego':torch.tensor([missing['ego'],fresh_dispatch['ego'],fresh_routing['ego']]),
}
out = critic(batch)
assert out.shape == (3,) and torch.isfinite(out).all()
out.sum().backward()
assert all(p.grad is not None for p in critic.parameters())
print('PASS — legacy routing, dispatch, routing cache, fresh routing, physical ego, critic forward/backward')
print('Ready to use the new Matrix Python parser; AnyLogic Java export must also be updated for fresh station data.')
