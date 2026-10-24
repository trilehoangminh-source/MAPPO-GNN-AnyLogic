"""AnyLogic Desktop Pypeline entrypoint for V8 15 AGV: get_config + get_action.

Experimental integration, fail-closed. Runs from the SAME Python 3.11 virtualenv
as the V7 PyTorch training source. Does not use RLlib or launch Alpyne server.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

os.environ["MAPPO_DISPATCH_MASK_MODE"] = "operation_first"

from thesis.envs.alpyne121_bridge import Observation
from .v7_inference import verify_checkpoint, make_inference_policy, greedy_action

ROOT = Path(__file__).resolve().parents[2]
CKPT_DIR = ROOT / "mappo_runs" / "V8_15AGV_fromV7_5h"
CHECKPOINT = Path(os.environ.get("V8_DESKTOP_CHECKPOINT", str(CKPT_DIR / "best_mappo.pt")))
RUN_CONFIG = Path(os.environ.get("V8_DESKTOP_RUN_CONFIG", str(CKPT_DIR / "run_config.json")))

_config, _ckpt = verify_checkpoint(CHECKPOINT, RUN_CONFIG)
if int(_config["fleetsize"]) != 15 or int(_config["max_fleetsize"]) != 20:
    raise RuntimeError("V8 Desktop requires 15 AGVs and max_fleetsize=20 in run_config.json")
_env = None
_model = None
_collector = None
_call_count = 0


def _make_model_free_environment():
    """Create Tobias observation/behavior converter without launching another AnyLogic."""
    from thesis.envs.matrix_v121 import Matrix
    from scripts.train_mappo_long_B3_v121 import SIM_CONFIG

    # Matrix.__init__ has an optional reset at the very end that normally starts
    # an Alpyne Java server. Suppress ONLY that initialization reset.
    # The external AnyLogic Desktop process owns the simulation instead.
    original_reset = Matrix.reset
    Matrix.reset = lambda self, *args, **kwargs: None
    try:
        env = Matrix(
            model_path=None, fleetsize=int(_config['fleetsize']),
            max_fleetsize=int(_config['max_fleetsize']),
            sim_config=SIM_CONFIG.copy(), max_seconds=None,
            pseudo_routing=False, pseudo_dispatcher=False,
            routing_agent_death=True, death_on_target=False,
            transform_dispatching_partobs=True, direction_reward=-0.1,
        )
    finally:
        Matrix.reset = original_reset
    # Required because no Alpyne reset() occurred.
    env._mappo_station_cache = None
    env._mappo_station_source = 'missing'
    return env


def _env_instance():
    global _env
    if _env is None:
        _env = _make_model_free_environment()
    return _env


def get_config(key: str):
    conf = _env_instance().config
    if key in conf.names():
        return conf.values()[conf.names().index(key)]
    raise KeyError(f"Missing Pypeline model configuration: {key}")


def _ensure_model(env):
    global _model, _collector
    if _model is not None:
        return
    from thesis.mappo.routing_guidance_v121 import GuidedEventCollector
    _model = make_inference_policy(CHECKPOINT, RUN_CONFIG, env.agent_hive)
    _collector = GuidedEventCollector(
        _model, env, stop_mask_mode=_config['stop_mask_mode'],
        routing_guide_strength=float(_config['final_used_guide_strength']),
    )
    print(f'[V8 DESKTOP] Loaded {CHECKPOINT.name} with 15 AGVs; graph fingerprint verified.',flush=True)


def get_action(observation, caller, n_nodes, context, statTitles, statValues, time):
    global _call_count
    env = _env_instance()
    # The V1.2 bridge expects a single mapping (unlike legacy Alpyne 0.2).
    packet = Observation({
        "obs": observation,
        "caller": int(caller),
        "n_nodes": n_nodes,
        "networkcontext": context,
        "rew": 0,
        "statTitles": statTitles,
        "statValues": statValues,
    })
    agent = env._agent_selection_fn(packet)
    if agent == 'dummy':
        raise RuntimeError('Desktop Pypeline demo received a dummy (centralized) agent; expected MARL caller')
    behavior = env.agent_behaviors[agent]
    # Essential: _catch_context populates HiveContext.nodes/paths/stations
    # and statistics. Without this, GraphSpec and observations cannot be built.
    behavior.is_for_learning(packet)
    obs, reward, done, info = behavior.convert_from_observation(packet)
    env.agent_selection = agent
    env._save_observation(packet, agent, obs, reward, done, info)
    _ensure_model(env)
    physical = int(caller) - 1000 if int(caller) >= 1000 else int(caller)
    role = 'routing' if int(caller) < 1000 else ('dispatch_front' if physical % 2 == 0 else 'dispatch_rear')
    # Reuse EXACT prepare(mask + Dijkstra bias) path from simulation training.
    obs_prepared = _collector._prepare(obs, role)
    action = greedy_action(_model, role, obs_prepared)
    result = behavior.convert_to_action(action, int(caller))
    # AnyLogic's existing take_action_pypeline() uses String parsing, NOT
    # the bridge Action object. Return its expected legacy textual shape:
    #   Action(actions=[3], receiver=1001)
    # Never send a Python object repr (which would be <... at 0x...>).
    fields = result.fields
    receiver = int(fields["receiver"])
    actions = [int(x) for x in fields["actions"]]
    if len(actions) != 1 or receiver != int(caller) or any(x < 0 for x in actions):
        raise RuntimeError(f"Invalid Pypeline action payload: {fields!r}")
    java_text = f"Action(actions={actions}, receiver={receiver})"
    _call_count += 1
    if _call_count <= 15 or _call_count % 300 == 0:
        print(f'[V8 DESKTOP] caller={caller} role={role} action={action} count={_call_count}',flush=True)
    return java_text
