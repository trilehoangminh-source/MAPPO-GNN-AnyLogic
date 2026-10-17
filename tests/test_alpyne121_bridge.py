"""Offline tests. No installed Alpyne/Java required; fake public API only."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

BRIDGE_PATH = Path(__file__).resolve().parents[1] / 'thesis' / 'envs' / 'alpyne121_bridge.py'

class FakeEngineState:
    ERROR = 'error'
    FINISHED = 'finished'
    PAUSED = 'paused'

class Status:
    def __init__(self, time=0.1, caller=1007, stop=False, state='paused'):
        self.state = state
        self.time = time
        self.stop = stop
        self.observation = {'caller': caller, 'obs': [[1, 0]], 'rew': 0, 'networkcontext': None}

class Sim:
    created = []
    def __init__(self, model_loc, **kwargs):
        self.created.append((model_loc, kwargs))
        self.status_ = Status()
        self.configs = []
        self.actions = []
        self.closed = False
    def reset(self, **kwargs):
        self.configs.append(kwargs)
        self.status_ = Status()
        return self.status_
    def take_action(self, **kwargs):
        self.actions.append(kwargs)
        self.status_ = Status(time=2.0, caller=7)
        return self.status_
    def lock(self): return self.status_
    def close(self): self.closed = True

sim_mod = types.ModuleType('alpyne.sim'); sim_mod.AnyLogicSim = Sim
const_mod = types.ModuleType('alpyne.constants'); const_mod.EngineState = FakeEngineState
base_mod = types.ModuleType('alpyne'); base_mod.__path__ = []
with patch.dict(sys.modules, {'alpyne':base_mod, 'alpyne.sim':sim_mod, 'alpyne.constants':const_mod}):
    spec = importlib.util.spec_from_file_location('bridge', BRIDGE_PATH)
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)

    c = bridge.Configuration(runmode=1, fleetsize=8, fixed_fleets=True)
    assert c.names() == ['runmode','fleetsize','fixed_fleets']
    assert c.fleetsize == 8
    client = bridge.AlpyneClient('some.zip',port=51700)
    run = client.create_reinforcement_learning(c)
    assert run.id is None
    assert run.run() is run
    assert run.id == 1
    assert run.get_state()[1]['model_time'] == 0.1
    assert run.get_observation().names() == ['caller','obs','rew','networkcontext']
    assert run.get_observation().caller == 1007
    run.reset(c)
    a = bridge.Action(data=[{'name':'receiver','value':1007},{'name':'actions','value':[4]}])
    run.take_action(a)
    assert client._sim.actions[-1] == dict(receiver=1007,actions=[4])
    assert run.get_observation().caller == 7
    assert run.get_state()[1]['model_time'] == 2.0
    assert not run.is_terminal()
    run.stop()
    assert client._sim.closed
    print('PASS: v1.2 bridge config, schema facade, reset, step, time, terminal, cleanup')
