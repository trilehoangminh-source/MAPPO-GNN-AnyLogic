"""Run the held-out Joint16 checkpoint in an AnyLogic Pypeline animation."""
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'legacy'

from thesis.envs.alpyne_bridge import Observation
from thesis.mappo.algorithm import as_actor, batchify
from thesis.mappo.model_architecture import TrafficRoutingActor, TobiasLinearDispatchActor
from thesis.mappo.routing_graph import GraphSpec

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'mappo_runs' / 'V17B_EVAL_EP0490' / 'joint16'
BLUEPRINT = ROOT / 'mappo_runs' / 'V15_FULLROLLOUT_GA512' / 'joint16' / 'run_config_blueprint.json'
_env = None
_actors = None


def _environment():
    global _env
    if _env is None:
        from thesis.envs.matrix import Matrix
        from scripts.model_builder import TOBIAS_JOINT_REWARDS
        reset = Matrix.reset
        Matrix.reset = lambda self, *args, **kwargs: None
        try:
            _env = Matrix(model_path=None, fleetsize=16, max_fleetsize=20,
                          sim_config=dict(TOBIAS_JOINT_REWARDS), max_seconds=None,
                          pseudo_routing=False, pseudo_dispatcher=False,
                          routing_agent_death=True, death_on_target=False,
                          transform_dispatching_partobs=True, direction_reward=-0.1)
        finally:
            Matrix.reset = reset
        _env._mappo_station_cache = None
        _env._mappo_station_source = 'missing'
    return _env


def get_config(key):
    config = _environment().config
    if key not in config.names():
        raise KeyError(key)
    return config.values()[config.names().index(key)]


def _load_actors(env, obs):
    global _actors
    if _actors is not None:
        return
    state = json.loads((RUN / 'state.json').read_text(encoding='utf-8-sig'))
    checkpoint = RUN / 'snapshot_ep_0490.pt'
    if int(state['last_episode']) != 490 or hashlib.sha256(checkpoint.read_bytes()).hexdigest() != state['checkpoint_sha256']:
        raise RuntimeError('EP490 checkpoint hash or episode mismatch')
    graph = GraphSpec.from_hive(env.agent_hive)
    cfg = json.loads(BLUEPRINT.read_text(encoding='utf-8-sig'))
    if graph.fingerprint != cfg['graph_fingerprint']:
        raise RuntimeError('AnyLogic topology differs from training blueprint')
    space = {key: SimpleNamespace(shape=np.asarray(value).shape) for key, value in obs.items()}
    station_count = space['stat'].shape[0]
    actors = {'routing': TrafficRoutingActor(graph),
              'dispatch_front': TobiasLinearDispatchActor(space, station_count),
              'dispatch_rear': TobiasLinearDispatchActor(space, station_count)}
    saved = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
    if set(saved['roles']) != set(actors):
        raise RuntimeError('Checkpoint roles mismatch')
    for name, actor in actors.items():
        actor.load_state_dict(saved['roles'][name]['actor'], strict=True)
        actor.eval()
    _actors = actors
    print('[ANIMATION] EP490 loaded; 16 AGVs, greedy policy', flush=True)


def get_action(observation, caller, n_nodes, context, statTitles, statValues, time):
    env = _environment()
    caller = int(caller)
    packet = Observation(dict(obs=observation, caller=caller, n_nodes=n_nodes,
                              networkcontext=context, rew=0,
                              statTitles=statTitles, statValues=statValues))
    agent = env._agent_selection_fn(packet)
    if agent == 'dummy':
        raise RuntimeError('Expected individual agent')
    behavior = env.agent_behaviors[agent]
    behavior.is_for_learning(packet)
    obs, reward, done, info = behavior.convert_from_observation(packet)
    env.agent_selection = agent
    env._save_observation(packet, agent, obs, reward, done, info)
    _load_actors(env, obs)
    physical = caller - 1000 if caller >= 1000 else caller
    role = 'routing' if caller < 1000 else ('dispatch_front' if physical % 2 == 0 else 'dispatch_rear')
    actor = _actors[role]
    mask = np.asarray(obs['action_mask'], dtype=np.float32).reshape(-1)[:actor.action_dim]
    prepared = {'agvs': np.asarray(obs['agvs'], dtype=np.float32),
                'stat': np.asarray(obs['stat'], dtype=np.float32), 'action_mask': mask}
    with torch.inference_mode():
        action = int(actor.dist(batchify(as_actor(prepared, torch.device('cpu')))).probs.argmax(-1).item())
    if role != 'routing' and mask[action] <= 0.5:
        raise RuntimeError('Dispatch selected an invalid action')
    fields = behavior.convert_to_action(action, caller).fields
    actions = [int(x) for x in fields['actions']]
    if int(fields['receiver']) != caller or len(actions) != 1:
        raise RuntimeError('Invalid action payload')
    return f'Action(actions={actions}, receiver={caller})'
