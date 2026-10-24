"""Run without AnyLogic/torch: syntax+Pypeline Observation and Java action contract.

python tests/test_v7_desktop_hotfix_offline.py
"""
import importlib.util
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "thesis" / "pypeline_connectors" / "matrix_mappo_v7.py"

# Avoid loading torch or launching a Java server in this isolated contract test.
for name in ("thesis", "thesis.envs", "thesis.pypeline_connectors"):
    package = types.ModuleType(name)
    package.__path__ = []
    sys.modules[name] = package

bridge = types.ModuleType("thesis.envs.alpyne121_bridge")
class Observation:
    def __init__(self, data):
        assert isinstance(data, dict), "new Alpyne Observation expects ONE dict"
        self.__dict__.update(data)
bridge.Observation = Observation
sys.modules[bridge.__name__] = bridge

inference = types.ModuleType("thesis.pypeline_connectors.v7_inference")
inference.verify_checkpoint = lambda ckpt, cfg: ({"stop_mask_mode": "forbid_when_move"}, object())
inference.make_inference_policy = lambda *a, **k: object()
inference.greedy_action = lambda model, role, obs: 3 if role == "routing" else 12
sys.modules[inference.__name__] = inference

spec = importlib.util.spec_from_file_location("thesis.pypeline_connectors.matrix_mappo_v7", SOURCE)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class FakeAction:
    def __init__(self, action, receiver):
        self.fields = {"actions": [int(action)], "receiver": int(receiver)}

class FakeBehavior:
    def __init__(self): self.ready = False
    def is_for_learning(self, packet):
        self.ready = True
        assert isinstance(packet, Observation)
        return True
    def convert_from_observation(self, packet):
        assert self.ready, "HiveContext must be loaded first"
        return {"obs": packet.obs}, 0.0, False, {}
    def convert_to_action(self, action, caller):
        assert self.ready
        return FakeAction(action, caller)

class FakeConfig:
    def names(self): return ["fleetsize", "dispatch"]
    def values(self): return [8, True]

class FakeEnv:
    config = FakeConfig()
    def __init__(self):
        self.agent_behaviors = {str(x): FakeBehavior() for x in (0, 1001)}
    def _agent_selection_fn(self, packet): return str(packet.caller)
    def _save_observation(self, *args): pass

env = FakeEnv()
m._env_instance = lambda: env
m._ensure_model = lambda _: None
m._model = object()
m._collector = types.SimpleNamespace(_prepare=lambda obs, role: obs)

assert m.get_config("fleetsize") == 8
assert m.get_config("dispatch") is True
for caller, expected_action in [(0, 3), (1001, 12)]:
    output = m.get_action([[0] * 30] * 8, caller, 105, [], [], [], 0.0)
    assert output == f"Action(actions=[{expected_action}], receiver={caller})", output
    # Minimal faithful decoding contract from Tobias's Java source.
    match = re.fullmatch(r"Action\(actions=\[(\d+)\], receiver=(\d+)\)", output)
    assert match and tuple(map(int, match.groups())) == (expected_action, caller)
print("V7 DESKTOP HOTFIX OFFLINE: PASS (Observation, HiveContext, Java action payload)")
