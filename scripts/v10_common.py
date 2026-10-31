"""Shared V10 wiring. Uses the exact V8 checkpoint and existing AnyLogic bridge."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json_atomic(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def source_config(source_run):
    source_run = Path(source_run).resolve()
    cfg = read_json(source_run / 'run_config.json')
    if (int(cfg['fleetsize']), int(cfg['max_fleetsize'])) != (20, 20):
        raise ValueError('V10 only supports the original V8 20-AGV run')
    if cfg['routing_encoder'] != 'residual_edge_gat':
        raise ValueError('V10 requires V8 Residual Edge-GAT')
    if cfg['dispatch_backbone'] != 'pooled' or cfg['dispatch_mask_mode'] != 'operation_first':
        raise ValueError('V10 requires original pooled MLP Dispatch + operation-first mask')
    if cfg.get('ppo_v7_mode') != 'stable' or cfg['final_used_guide_strength'] != 1.5:
        raise ValueError('V10 requires stable V7 PPO and fixed Dijkstra guidance 1.5')
    return cfg


def make_run_args(cfg, device, port, sim_seconds, max_events):
    """Reconstruct exact V8 architecture and simulation flags without modifying it."""
    import train_B3_MAPPO_v8_overnight as v8
    a = SimpleNamespace(
        model=Path(cfg['model_file']).resolve(), port_train=int(port),
        max_additional_episodes=5000,
        train_sim_seconds=int(sim_seconds), max_events=int(max_events),
        fleetsize=20, device=str(device), gamma=float(cfg['gamma'])
    )
    v7, args = v8.train_args(a, cfg)
    if not a.model.is_file():
        raise FileNotFoundError('AnyLogic export missing: ' + str(a.model))
    return v7, args


def setup_model(cfg, checkpoint, variant, device, port, sim_seconds, max_events,
                hidden, heads, layers, v10_checkpoint=None):
    """V8 initialize/load -> V10 adapter -> optional V10 full-state restore."""
    import torch
    import train_mappo_long_B3_v121 as baseline
    from thesis.mappo.mappo_stability_v7 import V7Options, enable_v7
    from thesis.mappo.v10_transformer import install_v10
    # Must occur BEFORE creating AnyLogic environment and reward references.
    baseline.SIM_CONFIG['reward_part_completed'] = 2.0
    os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'operation_first'
    v7, a = make_run_args(cfg, device, port, sim_seconds, max_events)
    env = v7.v1.baseline.build_env(a)
    try:
        model, spec, _ = v7.make_v4_model(env, a)
        if spec.fingerprint != cfg['graph_fingerprint']:
            raise RuntimeError('AnyLogic topology changed from V8; stop instead of mixing graphs')
        enable_v7(model, V7Options(**cfg['ppo_v7_options']))
        model.load(str(checkpoint))
        feature_dim = int(model.roles['dispatch_front'].actor.encoder.main[0].in_features)
        n_stations = int(model.roles['dispatch_front'].actor.action_dim)
        if n_stations != int(model.roles['dispatch_rear'].actor.action_dim):
            raise ValueError('Dispatch groups must use the same station action space')
        install_v10(model, variant, feature_dim, n_stations, hidden, heads, layers)
        if v10_checkpoint is not None:
            model.load(str(v10_checkpoint))
        return env, model, spec, a
    except Exception:
        env.close()
        raise


def filter_roles(trajectories, train_roles):
    if train_roles == 'all':
        return trajectories
    if train_roles != 'dispatch':
        raise ValueError('train_roles must be dispatch or all')
    result = [trajectory for trajectory in trajectories
              if trajectory and trajectory[0]['role'] in ('dispatch_front', 'dispatch_rear')]
    if not result:
        raise RuntimeError('No completed Dispatching transitions to learn from')
    return result
