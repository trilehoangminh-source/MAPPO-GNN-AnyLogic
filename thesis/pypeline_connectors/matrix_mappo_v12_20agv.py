"""Desktop AnyLogic Simulation animation with V12 FULL No-Dijkstra checkpoint.

Compatible with existing AnyLogic Main.runmode=3 Pypeline code, using
get_config(key) and get_action(observation,caller,n_nodes,context,statTitles,
statValues,time). DOES NOT start a separate AnyLogic/Alpyne Java server.

Designed against repository's existing, working V8 20-AGV animation connector.
Only changes reconstruction of the policy & checkpoint selection.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Required before importing environment/dispatch mask modules.
os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'operation_first'

from thesis.envs.alpyne121_bridge import Observation
from .v7_inference import make_inference_policy, greedy_action

ROOT = Path(__file__).resolve().parents[2]
V12_RUN = ROOT / 'mappo_runs' / 'V12_FULL_NO_DIJKSTRA_FROM_V11'
V12_MANIFEST = V12_RUN / 'experiment.json'
V12_CHECKPOINT = Path(os.environ.get(
    'V12_20AGV_DESKTOP_CHECKPOINT',
    str(V12_RUN / 'snapshot_ep_0830.pt')
)).resolve()

if not V12_MANIFEST.is_file():
    raise FileNotFoundError(f'V12 manifest not found: {V12_MANIFEST}')
if not V12_CHECKPOINT.is_file():
    raise FileNotFoundError(f'V12 checkpoint not found: {V12_CHECKPOINT}')
if V12_CHECKPOINT.parent != V12_RUN.resolve():
    raise RuntimeError('V12 checkpoint must reside in V12 run folder')

_manifest = json.loads(V12_MANIFEST.read_text(encoding='utf-8-sig'))
if _manifest.get('version') != 'V12_FULL_NO_DIJKSTRA_FROM_V11':
    raise RuntimeError('This connector requires the V12 FULL No-Dijkstra run')
V8_CHECKPOINT = Path(_manifest['v8_source_checkpoint']).resolve()
V8_CONFIG = Path(_manifest['v8_config_file']).resolve()
if not V8_CHECKPOINT.is_file() or not V8_CONFIG.is_file():
    raise FileNotFoundError('Missing historical V8 checkpoint/run_config required to rebuild V12 actor')

_env = None
_model = None
_collector = None
_call_count = 0


def _make_model_free_environment():
    """Construct observation/action converters without launching AnyLogic headless."""
    from thesis.envs.matrix_v121 import Matrix
    from scripts.train_mappo_long_B3_v121 import SIM_CONFIG

    sim_config = SIM_CONFIG.copy()
    sim_config['reward_part_completed'] = 2.0
    original_reset = Matrix.reset
    Matrix.reset = lambda self, *args, **kwargs: None
    try:
        env = Matrix(
            model_path=None,
            fleetsize=20,
            max_fleetsize=20,
            sim_config=sim_config,
            max_seconds=None,
            pseudo_routing=False,
            pseudo_dispatcher=False,
            routing_agent_death=True,
            death_on_target=False,
            transform_dispatching_partobs=True,
            direction_reward=-0.1,
        )
    finally:
        Matrix.reset = original_reset
    env._mappo_station_cache = None
    env._mappo_station_source = 'missing'
    return env


def _env_instance():
    global _env
    if _env is None:
        _env = _make_model_free_environment()
    return _env


def get_config(key: str):
    config = _env_instance().config
    if key in config.names():
        return config.values()[config.names().index(key)]
    raise KeyError(f'Unknown Pypeline configuration: {key}')


def _ensure_model(env):
    global _model, _collector
    if _model is not None:
        return

    import torch
    from thesis.mappo.algorithm import MAPPO
    from thesis.mappo.congestion_edgegat_v12_clean import install_v12
    from scripts.v12_clean_collector import RawRoutingEventCollector

    # Reconstruct the V8 actor exactly using LIVE AnyLogic networkcontext,
    # then upgrade to V12 attention WITHOUT either shortest-path guide.
    model = make_inference_policy(V8_CHECKPOINT, V8_CONFIG, env.agent_hive)
    info = install_v12(model)
    actor = model.roles['routing'].actor
    if (actor.dist.__func__ is not type(actor).dist
            or model.act.__func__ is not MAPPO.act
            or info['source_shortest_path_flow'] is not False):
        raise RuntimeError('V12 No-Dijkstra installation failed: guided action remains')

    # Only ACTOR weights needed for display/inference. Ignore Adam and Critics.
    # weights_only=True keeps checkpoint loading restricted to tensors/basic data.
    checkpoint = torch.load(str(V12_CHECKPOINT), map_location='cpu', weights_only=True)
    expected = {'routing', 'dispatch_front', 'dispatch_rear'}
    if set(checkpoint['roles']) != expected:
        raise RuntimeError('V12 checkpoint roles do not match the expected 3 actors')
    for role_name in sorted(expected):
        role_actor = model.roles[role_name].actor
        state = checkpoint['roles'][role_name]['actor']
        if role_name == 'routing' and any('next_hop' in k for k in state):
            raise RuntimeError('V12 checkpoint still contains shortest path buffer')
        role_actor.load_state_dict(state, strict=True)
        role_actor.eval()

    collector = RawRoutingEventCollector(model, env, stop_mask_mode=json.loads(
        V8_CONFIG.read_text(encoding='utf-8-sig'))['stop_mask_mode'])
    _model, _collector = model, collector
    print(f'[V12 EP830 ANIMATION] LOADED {V12_CHECKPOINT} '
          f'| 20 AGVs | observed-only Traffic GAT | no Dijkstra', flush=True)


def get_action(observation, caller, n_nodes, context, statTitles, statValues, time):
    global _call_count
    env = _env_instance()
    packet = Observation({
        'obs': observation,
        'caller': int(caller),
        'n_nodes': n_nodes,
        'networkcontext': context,
        'rew': 0,
        'statTitles': statTitles,
        'statValues': statValues,
    })
    agent = env._agent_selection_fn(packet)
    if agent == 'dummy':
        raise RuntimeError('Expected individual agent, got central dummy agent')
    behavior = env.agent_behaviors[agent]
    behavior.is_for_learning(packet)  # updates HiveContext from packet
    obs, reward, done, info = behavior.convert_from_observation(packet)
    env.agent_selection = agent
    env._save_observation(packet, agent, obs, reward, done, info)
    _ensure_model(env)

    physical = int(caller) - 1000 if int(caller) >= 1000 else int(caller)
    role = ('routing' if int(caller) < 1000
            else ('dispatch_front' if physical % 2 == 0 else 'dispatch_rear'))
    prepared = _collector._prepare(obs, role)
    if 'routing_guide_bias' in prepared:
        raise RuntimeError('Unexpected Dijkstra guidance in V12 actor input')
    action = greedy_action(_model, role, prepared)
    result = behavior.convert_to_action(action, int(caller))
    fields = result.fields
    receiver = int(fields['receiver'])
    actions = [int(x) for x in fields['actions']]
    if len(actions) != 1 or receiver != int(caller) or any(x < 0 for x in actions):
        raise RuntimeError(f'Invalid V12 inference action: {fields!r}')

    # AnyLogic Main.take_action_pypeline() parses this exact format.
    java_text = f'Action(actions={actions}, receiver={receiver})'
    _call_count += 1
    if _call_count <= 15 or _call_count % 300 == 0:
        print(f'[V12 EP830 ANIMATION] t={time} caller={caller} '
              f'role={role} action={action} count={_call_count}', flush=True)
    return java_text
