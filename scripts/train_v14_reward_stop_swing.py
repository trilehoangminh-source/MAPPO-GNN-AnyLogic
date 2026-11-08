"""Fork committed V14 STOP-UNLOCK weights into a NEW direct-reward experiment.

V14 architecture, stop mask, AnyLogic model, 20 AGV, PPO, reward from Java,
optimizer, ValueNorm, 3600s and minibatch limit remain unchanged. Only Python
Routing reward returned by MultiAgent.convert_from_observation() is shaped:
 - apparently unnecessary Stop: negative penalty (conservatively screened)
 - A -> B -> A positional swing: negative penalty

Source V14 STOP-UNLOCK run and original V14 GitHub backup are READ-ONLY.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from train_v14_stop_unlocked import (
    VERSION as SOURCE_VERSION,
    read_json,
    atomic_json,
    check_state,
    sha256,
)
from v14_deep_mlp_common import build_v14
from thesis.envs.v14_reward_shaping import RewardConfig, RoutingRewardShaper

VERSION = 'V14_GAT4_STOP_UNLOCK_DIRECT_WAIT_SWING_REWARD_V1'
STOP_MODE = 'baseline'
DEFAULT_SOURCE = ROOT / 'mappo_runs' / 'V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK'
DEFAULT_OUT = ROOT / 'mappo_runs' / 'V14_GAT4_MLP_3600S_MB128_WAIT_SWING_REWARD'
REWARD_MODULE = ROOT / 'thesis' / 'envs' / 'v14_reward_shaping.py'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fork-from', type=Path, default=None)
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--port', type=int, default=51880)
    p.add_argument('--sim-seconds', type=int, default=3600)
    p.add_argument('--max-events', type=int, default=None)
    p.add_argument('--max-episodes', type=int, default=1100,
                   help='Absolute episode target, NOT number of additional episodes')
    p.add_argument('--max-minibatch', type=int, default=128)
    p.add_argument('--gat-layers', type=int, default=4)
    p.add_argument('--gat-heads', type=int, default=2)
    p.add_argument('--dispatch-width', type=int, default=128)
    p.add_argument('--embedding', type=int, default=64)
    p.add_argument('--station-id-dim', type=int, default=16)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--snapshot-every', type=int, default=10)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--stop-penalty', type=float, default=-0.5)
    p.add_argument('--swing-penalty', type=float, default=-1.0)
    p.add_argument('--candidate-clearance', type=float, default=0.025)
    p.add_argument('--local-risk-radius', type=float, default=0.040)
    p.add_argument('--position-tolerance', type=float, default=0.002)
    p.add_argument('--init-only', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv)
    if (a.max_episodes < 1 or a.sim_seconds < 1 or a.max_minibatch < 1
            or a.cpu_threads < 1 or a.snapshot_every < 1 or a.port < 1
            or a.port > 65535 or a.gat_layers != 4 or a.gat_heads < 1
            or a.dispatch_width < 8 or a.embedding < 8 or a.station_id_dim < 1
            or (a.max_events is not None and a.max_events < 1)):
        p.error('Invalid V14 architecture, simulation, port, or training arguments')
    if a.resume and a.fork_from is not None:
        p.error('For --resume, DO NOT pass --fork-from')
    if a.resume and a.init_only:
        p.error('Use --init-only for FORKING only, not --resume')
    if a.resume and a.dry_run:
        # Read-only resume preflight is allowed.
        pass
    # Raises immediately for invalid reward settings.
    _ = configuration(a)
    return a


def configuration(args) -> RewardConfig:
    return RewardConfig(
        stop_penalty=args.stop_penalty,
        swing_penalty=args.swing_penalty,
        candidate_clearance=args.candidate_clearance,
        local_risk_radius=args.local_risk_radius,
        position_tolerance=args.position_tolerance,
    )


def parameter_check(args, meta):
    """Keep model, RNG, simulator and PPO hyperparameters identical to source."""
    args.seed = int(meta['seed']) if args.seed is None else args.seed
    args.max_events = int(meta['max_events']) if args.max_events is None else args.max_events
    names = ('seed', 'max_events', 'sim_seconds', 'max_minibatch',
             'gat_layers', 'gat_heads', 'dispatch_width', 'embedding',
             'station_id_dim', 'cpu_threads', 'snapshot_every', 'device', 'port')
    mismatch = {k: (meta[k], getattr(args, k)) for k in names
                if meta[k] != getattr(args, k)}
    if mismatch:
        raise RuntimeError('V14 configuration mismatch; no fork/resume: ' + str(mismatch))


def inspect_source(source: Path, out: Path, args):
    """Read-only validation; no source metrics or checkpoint can be modified."""
    source = source.resolve()
    if source == out or out.is_relative_to(source) or source.is_relative_to(out):
        raise RuntimeError('Source and destination must be DIFFERENT non-nested folders')
    meta = read_json(source / 'experiment.json')
    if meta.get('version') != SOURCE_VERSION:
        raise RuntimeError('Source must be the V14 STOP-UNLOCK run, not V14 scratch or V12')
    if meta.get('stop_mask_mode') != STOP_MODE:
        raise RuntimeError('Source must have Stop UNLOCKED, with baseline mask')
    if int(meta['fleetsize']) != 20 or int(meta['gat_layers']) != 4:
        raise RuntimeError('Incompatible fleet or GAT depth')
    cfg_path = source / 'run_config_blueprint.json'
    cfg = read_json(cfg_path)
    if sha256(cfg_path) != meta['source_blueprint_sha256']:
        raise RuntimeError('Source environment blueprint has changed')
    if sha256(Path(cfg['model_file']).resolve()) != meta['anylogic_sha256']:
        raise RuntimeError('Original AnyLogic ZIP has changed')
    parameter_check(args, meta)
    state = read_json(source / 'state.json')
    last_ep, checkpoint = check_state(state, source)
    if last_ep < int(meta['fork_episode']):
        raise RuntimeError('Source state predates its fork episode')
    rows = [json.loads(x) for x in (source / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()
            if x.strip()]
    committed = [int(x['episode']) for x in rows if int(x['episode']) <= last_ep]
    if committed != list(range(int(meta['fork_episode']) + 1, last_ep + 1)):
        raise RuntimeError('Source STOP-UNLOCK metrics are missing/duplicated')
    # Any uncommitted source tail is merely ignored, NEVER truncated here.
    return cfg, cfg_path, meta, last_ep, checkpoint


def resume_preflight(out: Path, args):
    meta = read_json(out / 'experiment.json')
    if meta.get('version') != VERSION or meta.get('stop_mask_mode') != STOP_MODE:
        raise RuntimeError('Not this V14 reward experiment; refusing resume')
    if meta.get('reward_config') != configuration(args).metadata():
        raise RuntimeError('Reward coefficients differ from fork; start ANOTHER run')
    if sha256(REWARD_MODULE) != meta['reward_implementation_sha256']:
        raise RuntimeError('Reward implementation changed since fork initialization')
    cfg_path = out / 'run_config_blueprint.json'
    cfg = read_json(cfg_path)
    if sha256(cfg_path) != meta['source_blueprint_sha256']:
        raise RuntimeError('Environment blueprint changed')
    if sha256(Path(cfg['model_file']).resolve()) != meta['anylogic_sha256']:
        raise RuntimeError('AnyLogic ZIP changed')
    parameter_check(args, meta)
    state = read_json(out / 'state.json')
    last_ep, checkpoint = check_state(state, out)
    rows = [json.loads(x) for x in (out / 'metrics.jsonl').read_text(encoding='utf-8').splitlines()
            if x.strip()]
    committed = [int(x['episode']) for x in rows if int(x['episode']) <= last_ep]
    if committed != list(range(int(meta['fork_episode']) + 1, last_ep + 1)):
        raise RuntimeError('Committed reward-run metrics missing/duplicated')
    # Do not mutate files in preflight. If an incomplete metrics tail exists,
    # resume later can append, but the next check will reject duplicates.
    if len(rows) != len(committed):
        raise RuntimeError('Uncommitted metrics tail found; inspect/backup before resume')
    return cfg, cfg_path, meta, state, last_ep, checkpoint


def main(argv=None):
    a = parse_args(argv)
    out = a.output.resolve()
    if a.resume:
        cfg, cfg_path, meta, state, current_ep, checkpoint = resume_preflight(out, a)
        source_cp = None
    else:
        source = (a.fork_from or DEFAULT_SOURCE).resolve()
        cfg, cfg_path, source_meta, current_ep, source_cp = inspect_source(source, out, a)
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Reward fork output is NOT empty: ' + str(out))
        cp_hash = sha256(source_cp)
        meta = dict(source_meta)
        meta.update(
            version=VERSION,
            source_model_version=source_meta['version'],
            source_checkpoint_loaded=True,
            optimizer_source='inherited_from_committed_v14_stop_unlock',
            fork_from_run=str(source),
            fork_episode=current_ep,
            original_stop_unlock_fork_episode=int(source_meta['fork_episode']),
            source_checkpoint=str(source_cp),
            source_checkpoint_sha256=cp_hash,
            parent_training_episodes=current_ep,
            train_started_at_episode=current_ep + 1,
            stop_mask_mode=STOP_MODE,
            reward_mode='direct_behavior_reward',
            reward_config=configuration(a).metadata(),
            reward_implementation_sha256=sha256(REWARD_MODULE),
            explanation='V14 STOP-UNLOCK checkpoint and Adam/ValueNorm unchanged; ONLY Routing behavior reward adjusted for apparent unnecessary Stop and A-B-A swing',
        )
        checkpoint = source_cp
        state = dict(last_episode=current_ep,
                     checkpoint=str((out / f'rolling_slot_{current_ep%2}.pt').resolve()),
                     snapshots=[], run_seconds=0.0,
                     fork_episode=current_ep, fork_from_run=str(source))
    if a.dry_run:
        print('[V14 REWARD DRY RUN] source committed EP=', current_ep)
        print('[V14 REWARD DRY RUN] output=', out)
        print('[V14 REWARD DRY RUN] checkpoint=', checkpoint)
        print('[V14 REWARD DRY RUN] Stop remains UNLOCKED; all roles still train; no Dijkstra')
        print('[V14 REWARD DRY RUN] reward=', configuration(a).metadata())
        print('[V14 REWARD DRY RUN] no files changed, no AnyLogic started')
        return
    if a.max_episodes <= current_ep and not a.init_only:
        print('[V14 REWARD DONE] run already committed EP', current_ep)
        return
    import numpy as np
    import torch
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
    from thesis.mappo.mappo_stability_v7 import audit_rollout_roles
    from v12_clean_audit import verify_v12_ppo_rollout_parity
    from v14_reward_collector import V14RewardEventCollector

    if str(a.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available')
    os.environ['MAPPO_MAX_MINIBATCH'] = str(a.max_minibatch)
    os.environ['OMP_NUM_THREADS'] = str(a.cpu_threads)
    os.environ['MKL_NUM_THREADS'] = str(a.cpu_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(a.cpu_threads)
    os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'operation_first'
    torch.set_num_threads(a.cpu_threads)
    logging.getLogger('alpyne.data').setLevel(logging.ERROR)
    logging.getLogger('alpyne.sim').setLevel(logging.ERROR)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    shaper = RoutingRewardShaper(configuration(a))
    shaper.install()  # actual MultiAgent behavior: directly adjust returned Routing reward
    env = None
    try:
        env, model, spec, arch = build_v14(
            cfg, a.device, a.port, a.sim_seconds, a.max_events,
            gat_layers=a.gat_layers, gat_heads=a.gat_heads,
            dispatch_width=a.dispatch_width, embedding=a.embedding,
            station_id_dim=a.station_id_dim, checkpoint=checkpoint)
        if not a.resume:
            # Do not create output until AnyLogic/model and old checkpoint loaded.
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cfg_path, out / 'run_config_blueprint.json')
            target = out / f'rolling_slot_{current_ep%2}.pt'
            shutil.copy2(source_cp, target)
            if sha256(target) != meta['source_checkpoint_sha256']:
                raise RuntimeError('Inherited checkpoint copy has WRONG hash')
            atomic_json(out / 'experiment.json', meta)
            (out / 'metrics.jsonl').touch(exist_ok=False)
            atomic_json(out / 'state.json', state)
            print(f'[V14 REWARD FORK] inherited EP{current_ep} FULL checkpoint; no weights/Adam/ValueNorm reset', flush=True)
        else:
            print(f'[V14 REWARD RESUME] committed EP{current_ep}: {checkpoint}', flush=True)
        if a.init_only:
            print('[V14 REWARD INIT ONLY] fork committed; no training performed', flush=True)
            return
        auditor = RoutingGraphAudit(model.roles['routing'].actor, sample_limit=256)
        collector = V14RewardEventCollector(model, env, shaper, stop_mask_mode=STOP_MODE)
        print('[V14 REWARD READY] 20AGV GAT4 + 2 candidate-aware MLP + 3 critics; 3600s; no-Dijkstra; Stop UNLOCKED', flush=True)
        while int(state['last_episode']) < a.max_episodes:
            ep = int(state['last_episode']) + 1
            seed = int(a.seed) + ep - 1
            random.seed(int(a.seed) + ep)
            np.random.seed(int(a.seed) + ep)
            torch.manual_seed(int(a.seed) + ep)
            t0 = time.monotonic()
            auditor.reset()
            trajectories, env_meta = collector.episode(
                seed=seed, train=True, max_event_steps=a.max_events)
            if GraphSpec.from_hive(env.agent_hive).fingerprint != spec.fingerprint:
                raise RuntimeError('Graph mapping changed; no update')
            if env_meta.get('end_reason') == 'max_event_steps':
                raise RuntimeError('Truncated at max-events: refusing PPO update')
            mapping = auditor.report()
            if mapping['candidate_map_rate'] < .99:
                raise RuntimeError('Routing node map rate < 99%')
            parity = verify_v12_ppo_rollout_parity(model, trajectories, max_samples=64)
            role_audit = audit_rollout_roles(model, trajectories, max_per_role=64)
            if isinstance(parity, dict) and parity.get('passed') is False:
                raise RuntimeError('PPO parity audit failed')
            present = {t[0]['role'] for t in trajectories if t}
            if 'routing' not in present:
                raise RuntimeError('No routing transitions; no update')
            updates = model.train(trajectories)  # all present roles, unchanged PPO
            slot = out / f'rolling_slot_{ep%2}.pt'
            tmp = Path(str(slot) + '.tmp')
            model.save(str(tmp))
            os.replace(tmp, slot)
            elapsed = time.monotonic() - t0
            row = dict(
                version=VERSION, episode=ep, environment_seed=seed,
                environment=env_meta, graph_mapping=mapping,
                ppo_audit=parity, all_roles_audit=role_audit,
                policy_updates=updates, roles_present=sorted(present),
                train_roles='all', max_minibatch=a.max_minibatch,
                dijkstra_guidance_strength=0.0,
                predicted_shortest_path_flow=False,
                traffic_schema='observed_only_v12',
                sim_seconds=a.sim_seconds, architectures=arch,
                stop_mask_mode=STOP_MODE,
                reward_mode=meta['reward_mode'],
                reward_config=meta['reward_config'],
                fork_episode=int(meta['fork_episode']),
                checkpoint_origin_sha256=meta['source_checkpoint_sha256'],
            )
            with (out / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(row, default=str, ensure_ascii=False) + '\n')
                handle.flush()
                os.fsync(handle.fileno())
            state['last_episode'] = ep
            state['checkpoint'] = str(slot.resolve())
            state['run_seconds'] = round(float(state['run_seconds']) + elapsed, 2)
            if ep % a.snapshot_every == 0:
                snapshot = out / f'snapshot_ep_{ep:04d}.pt'
                tmp_snap = Path(str(snapshot) + '.tmp')
                shutil.copy2(slot, tmp_snap)
                os.replace(tmp_snap, snapshot)
                state['snapshots'].append(str(snapshot.resolve()))
            atomic_json(out / 'state.json', state)
            stats = env_meta.get('statistics', {})
            rew = env_meta.get('reward_shaping', {})
            print(f'[V14 REWARD EP{ep}] JobDone={stats.get("jobDone",0)} '
                  f'Block={stats.get("block",0)} LiveLock={stats.get("liveLock",0)} '
                  f'wrongTarget={stats.get("wrongTarget",0)} '
                  f'StopPenalty={rew.get("stop_penalties_applied",0)} '
                  f'SwingPenalty={rew.get("swing_penalties_applied",0)} '
                  f'Reward={env_meta.get("delivered_reward",0):.2f} '
                  f'Elapsed={elapsed:.1f}s', flush=True)
    except KeyboardInterrupt:
        print('[V14 REWARD INTERRUPTED] Last committed EP', state['last_episode'], flush=True)
    finally:
        if env is not None:
            env.close()
    print('[V14 REWARD DONE] committed EP', state['last_episode'], 'output=', out, flush=True)


if __name__ == '__main__':
    main()
