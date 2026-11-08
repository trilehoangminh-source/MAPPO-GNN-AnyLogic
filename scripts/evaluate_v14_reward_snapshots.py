"""Evaluate V14 reward-run EP110/120/130 with identical held-out seeds.

Inference ONLY: V14 GAT4 + 2 Dispatch actors + 3 critics, original action masks,
Stop UNLOCKED, V14 Python reward shaping. No model.train(), no PPO updates,
no writes to source training state/metrics/checkpoints. Interruption-safe JSONL.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

VERSION = 'V14_REWARD_SNAPSHOT_HELDOUT_V1'
SOURCE_VERSION = 'V14_GAT4_STOP_UNLOCK_DIRECT_WAIT_SWING_REWARD_V1'
DEFAULT_RUN = ROOT / 'mappo_runs' / 'V14_GAT4_MLP_3600S_MB128_WAIT_SWING_REWARD'
DEFAULT_EPISODES = (110, 120, 130)
DEFAULT_SEEDS = tuple(range(900001, 900011))
REWARD_MODULE = ROOT / 'thesis' / 'envs' / 'v14_reward_shaping.py'
METRICS = ('jobDone', 'block', 'liveLock', 'wrongTarget', 'targetReached',
           'block_time_s', 'stop_ratio_pct', 'stop_with_move', 'stop_penalties',
           'swing_penalties', 'reward')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=DEFAULT_RUN)
    p.add_argument('--episodes', type=int, nargs='+', default=list(DEFAULT_EPISODES))
    p.add_argument('--seeds', type=int, nargs='+', default=list(DEFAULT_SEEDS))
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--port', type=int, default=51881)
    p.add_argument('--dry-run', action='store_true', help='Verify source and snapshots, no AnyLogic/GPU and no files written')
    p.add_argument('--resume', action='store_true', help='Continue incomplete evaluation JSONL; verify previous results')
    a = p.parse_args(argv)
    if not (1 <= a.port <= 65535):
        p.error('Invalid AnyLogic port')
    if len(set(a.seeds)) != len(a.seeds) or not a.seeds or any(s < 0 for s in a.seeds):
        p.error('Held-out seeds must be distinct nonnegative integers')
    if len(set(a.episodes)) != len(a.episodes) or not a.episodes or any(e <= 0 for e in a.episodes):
        p.error('Episodes must be distinct positive integers')
    return a


def parse_result_row(ep, seed, data, model_time=None, last_decision_time=None):
    stats = data.get('statistics') or {}
    action_diag = data.get('action_diagnostics') or {}
    role_actions = (action_diag.get('actions_by_role') or {}).get('routing') or {}
    actions = {str(k): int(v) for k, v in role_actions.items()}
    total_routing = sum(actions.values())
    stop_count = int(actions.get('0', 0))
    shaping = data.get('reward_shaping') or {}

    def number(key):
        v = stats.get(key)
        return float(v) if v is not None else None

    block_time = sum(float(value or 0) for name, value in stats.items()
                     if name.startswith('_blocksec_agv'))
    result = {
        'episode': int(ep), 'seed': int(seed),
        'jobDone': number('jobDone'), 'block': number('block'),
        'liveLock': number('liveLock'), 'wrongTarget': number('wrongTarget'),
        'targetReached': number('targetReached'),
        'block_time_s': float(block_time),
        'stop_ratio_pct': 100.0 * stop_count / total_routing if total_routing else None,
        'stop_count': stop_count, 'routing_decisions': total_routing,
        'stop_with_move': int(action_diag.get('routing_stop_with_move_available', 0)),
        'stop_penalties': int(shaping.get('stop_penalties_applied', 0)),
        'swing_penalties': int(shaping.get('swing_penalties_applied', 0)),
        'reward': float(data.get('delivered_reward', 0)),
        'reward_adjustment': float(shaping.get('total_reward_adjustment', 0)),
        'events': int(data.get('events', 0)),
        'end_reason': data.get('end_reason'),
        'reached_terminal': bool(data.get('reached_terminal', False)),
        'model_time_s': model_time, 'last_decision_time_s': last_decision_time,
    }
    return result


def preflight(run: Path, episodes, seeds):
    """Fully read-only and does not require importing AnyLogic or PyTorch."""
    run = Path(run).resolve()
    meta = read_json(run / 'experiment.json')
    if meta.get('version') != SOURCE_VERSION:
        raise RuntimeError('Expected V14 DIRECT WAIT/SWING REWARD run, not scratch/STOP-UNLOCK')
    if (int(meta.get('fleetsize', -1)) != 20 or int(meta.get('max_fleetsize', -1)) != 20
            or int(meta.get('sim_seconds', -1)) != 3600
            or int(meta.get('gat_layers', -1)) != 4
            or meta.get('stop_mask_mode') != 'baseline'):
        raise RuntimeError('Expected 20 AGV, 3600s, GAT4, Stop-unlocked baseline mask')
    if meta.get('reward_mode') != 'direct_behavior_reward':
        raise RuntimeError('Reward shaping experiment metadata is missing/incorrect')
    if not isinstance(meta.get('reward_config'), dict) or not meta['reward_config']:
        raise RuntimeError('Missing reward_config, refusing to assume coefficients')
    if not REWARD_MODULE.is_file() or sha256(REWARD_MODULE) != meta.get('reward_implementation_sha256'):
        raise RuntimeError('Reward implementation changed since run initialization')
    cfg_path = run / 'run_config_blueprint.json'
    cfg = read_json(cfg_path)
    if sha256(cfg_path) != meta.get('source_blueprint_sha256'):
        raise RuntimeError('Environment blueprint SHA256 mismatch')
    anylogic = Path(cfg['model_file']).resolve()
    if not anylogic.is_file() or sha256(anylogic) != meta.get('anylogic_sha256'):
        raise RuntimeError('AnyLogic model ZIP differs from training source')
    if (int(cfg.get('fleetsize', -1)) != 20 or int(cfg.get('max_fleetsize', -1)) != 20):
        raise RuntimeError('Blueprint is not 20 AGV')
    max_events = int(meta['max_events'])
    if max_events <= 0:
        raise RuntimeError('Invalid max_events from original experiment')
    if (run / 'state.json').is_file():
        state = read_json(run / 'state.json')
        if int(state['last_episode']) < max(episodes):
            raise RuntimeError('Source training state predates requested snapshot')
    cps = {}
    hashes = {}
    for ep in episodes:
        path = run / f'snapshot_ep_{ep:04d}.pt'
        if not path.is_file() or path.is_symlink() or path.resolve().parent != run:
            raise FileNotFoundError(f'Missing exact immutable snapshot EP{ep}: {path}')
        cps[ep] = path.resolve()
        hashes[ep] = sha256(path)
    return run, meta, cfg, cps, hashes, max_events


def output_paths(run: Path, episodes, seeds):
    ep_label = '-'.join(str(x) for x in episodes)
    stem = f'heldout_reward_ep{ep_label}_{len(seeds)}seeds_3600s'
    return (run / f'{stem}.jsonl', run / f'{stem}_per_seed.csv',
            run / f'{stem}_summary.csv', run / f'{stem}_paired.csv')


def load_existing(path, episodes, seeds, hashes, protocol_sha, implementation_sha):
    result = {}
    for line_number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'Partial/invalid JSONL line {line_number}; backup and inspect: {path}') from exc
        ep, seed = int(row['episode']), int(row['seed'])
        key = (ep, seed)
        if (ep not in episodes or seed not in seeds or key in result
                or row.get('checkpoint_sha256') != hashes[ep]
                or row.get('eval_protocol_sha256') != protocol_sha
                or row.get('reward_implementation_sha256') != implementation_sha
                or row.get('eval_version') != VERSION):
            raise RuntimeError(f'Evaluation result mismatch/duplicate on line {line_number}; refusing resume')
        result[key] = row
    return result


def mean(values):
    return statistics.mean(values) if values else None


def sd(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0 if values else None


def rows_summary(rows, episodes, seeds):
    summary, paired = [], []
    index = {(int(r['episode']), int(r['seed'])): r for r in rows}
    for ep in episodes:
        samples = [index[(ep, s)] for s in seeds if (ep, s) in index]
        item = {'checkpoint_ep': ep, 'n_seeds': len(samples)}
        for key in METRICS:
            vals = [float(x[key]) for x in samples if x.get(key) is not None]
            item[f'{key}_mean'] = mean(vals)
            item[f'{key}_sd'] = sd(vals)
        summary.append(item)
    baseline_ep = episodes[0]
    for ep in episodes[1:]:
        for key in METRICS:
            vals = [float(index[(ep, s)][key]) - float(index[(baseline_ep, s)][key])
                    for s in seeds if (ep, s) in index and (baseline_ep, s) in index
                    and index[(ep, s)].get(key) is not None
                    and index[(baseline_ep, s)].get(key) is not None]
            n = len(vals)
            paired.append({'metric': key, 'checkpoint_comparison': f'EP{ep} minus EP{baseline_ep}',
                           'paired_seeds': n, 'mean_delta': mean(vals), 'sd_delta': sd(vals),
                           'approx_95ci_low': (mean(vals) - 2.262 * sd(vals) / math.sqrt(n)
                                               if n == 10 and sd(vals) is not None else None),
                           'approx_95ci_high': (mean(vals) + 2.262 * sd(vals) / math.sqrt(n)
                                                if n == 10 and sd(vals) is not None else None)})
    return summary, paired


def write_csv_atomic(path: Path, rows):
    if not rows:
        return
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)
    os.replace(tmp, path)


def run_evaluation(args):
    run, meta, cfg, cps, hashes, max_events = preflight(args.run, args.episodes, args.seeds)
    paths = output_paths(run, args.episodes, args.seeds)
    protocol_sha = sha256(Path(__file__).resolve())
    implementation_sha = sha256(REWARD_MODULE)
    print(f'[PRECHECK] run={run}', flush=True)
    print(f'[PRECHECK] checkpoint EPs={args.episodes}; 20 AGV; 3600s; greedy; {len(args.seeds)} same seeds', flush=True)
    print(f'[PRECHECK] seeds={args.seeds}; max_events={max_events}; stop_mask=baseline', flush=True)
    print(f'[PRECHECK] reward_config={meta["reward_config"]}', flush=True)
    for ep in args.episodes:
        print(f'[SNAPSHOT] EP{ep} {cps[ep]} sha256={hashes[ep]}', flush=True)
    if args.dry_run:
        print('[DRY RUN OK] No checkpoint/model/metrics modified. Output will be:', paths[0], flush=True)
        return
    if paths[0].exists() and not args.resume:
        raise FileExistsError('Evaluation exists. Use --resume (never overwrite): ' + str(paths[0]))
    existing = (load_existing(paths[0], args.episodes, args.seeds, hashes, protocol_sha, implementation_sha)
                if paths[0].exists() else {})
    if args.resume and paths[0].exists():
        print(f'[RESUME] Found {len(existing)} verified completed evaluation rows', flush=True)

    import numpy as np
    import torch
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
    from thesis.envs.v14_reward_shaping import RewardConfig, RoutingRewardShaper
    from v14_deep_mlp_common import build_v14
    from v14_reward_collector import V14RewardEventCollector

    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available; use --device cpu')
    os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'operation_first'
    os.environ['MAPPO_MAX_MINIBATCH'] = str(int(meta['max_minibatch']))
    logging_levels = ('alpyne.data', 'alpyne.sim')
    import logging
    for name in logging_levels:
        logging.getLogger(name).setLevel(logging.ERROR)
    torch.set_num_threads(int(meta.get('cpu_threads', 2)))

    shaper = RoutingRewardShaper(RewardConfig(**meta['reward_config']))
    shaper.install()
    env = None
    try:
        # One AnyLogic engine; load each exact checkpoint without touching train files.
        env, model, spec, architecture = build_v14(
            cfg, args.device, args.port, 3600, max_events,
            gat_layers=int(meta['gat_layers']), gat_heads=int(meta['gat_heads']),
            dispatch_width=int(meta['dispatch_width']), embedding=int(meta['embedding']),
            station_id_dim=int(meta['station_id_dim']), checkpoint=cps[args.episodes[0]])
        for role in model.roles.values():
            role.actor.eval()
            role.critic.eval()
        collector = V14RewardEventCollector(model, env, shaper, stop_mask_mode='baseline')
        auditor = RoutingGraphAudit(model.roles['routing'].actor, sample_limit=256)
        with paths[0].open('a' if args.resume else 'x', encoding='utf-8') as handle:
            for ep in args.episodes:
                if ep != args.episodes[0]:
                    model.load(str(cps[ep]))
                    for role in model.roles.values():
                        role.actor.eval()
                        role.critic.eval()
                for seed in args.seeds:
                    if (ep, seed) in existing:
                        print(f'[SKIP] EP{ep} seed={seed} already evaluated', flush=True)
                        continue
                    # Reset PRNG so all checkpoints see identical random initialization.
                    random.seed(seed)
                    np.random.seed(seed)
                    torch.manual_seed(seed)
                    auditor.reset()
                    with torch.no_grad():
                        _, data = collector.episode(seed=seed, train=False, max_event_steps=max_events)
                    if data.get('end_reason') == 'max_event_steps' or not data.get('reached_terminal'):
                        raise RuntimeError(f'EP{ep} seed={seed}: incomplete simulation, not a valid held-out episode')
                    if GraphSpec.from_hive(env.agent_hive).fingerprint != spec.fingerprint:
                        raise RuntimeError(f'EP{ep} seed={seed}: graph topology changed')
                    report = auditor.report()
                    if report.get('candidate_map_rate', 0) < 0.99:
                        raise RuntimeError(f'EP{ep} seed={seed}: graph candidate mapping below 99%')
                    try:
                        last_decision_time = float(env.get_mappo_time())
                    except Exception:
                        last_decision_time = None
                    try:
                        # This clock is authoritative; get_mappo_time() only keeps
                        # the LAST DECISION timestamp and may lag model termination.
                        model_time = float(env.run.get_state()[1]['model_time'])
                    except Exception:
                        model_time = None
                    if model_time is not None and model_time < 3599.0:
                        raise RuntimeError(f'EP{ep} seed={seed}: actual sim clock ended early at {model_time:.1f}s')
                    row = parse_result_row(ep, seed, data, model_time=model_time,
                                           last_decision_time=last_decision_time)
                    row.update({
                        'eval_version': VERSION, 'training_run_version': SOURCE_VERSION,
                        'evaluation': True, 'greedy': True, 'sim_seconds': 3600,
                        'fleet_size': 20, 'checkpoint': str(cps[ep]),
                        'checkpoint_sha256': hashes[ep],
                        'eval_protocol_sha256': protocol_sha,
                        'reward_implementation_sha256': implementation_sha,
                        'reward_config': meta['reward_config'],
                        'graph_mapping': report,
                    })
                    handle.write(json.dumps(row, default=str, ensure_ascii=False) + '\n')
                    handle.flush()
                    os.fsync(handle.fileno())
                    existing[(ep, seed)] = row
                    print(f'[EVAL EP{ep} seed={seed}] JobDone={row["jobDone"]} '
                          f'Block={row["block"]} LiveLock={row["liveLock"]} '
                          f'Stop%={row["stop_ratio_pct"]:.1f} BlockTime={row["block_time_s"]:.1f}s', flush=True)
    except KeyboardInterrupt:
        print('[EVAL INTERRUPTED] Completed rows safely stored. Rerun with --resume.', flush=True)
        return
    finally:
        if env is not None:
            env.close()

    rows = [existing[(ep, seed)] for ep in args.episodes for seed in args.seeds
            if (ep, seed) in existing]
    if len(rows) != len(args.episodes) * len(args.seeds):
        raise RuntimeError(f'Expected {len(args.episodes)*len(args.seeds)} rows, got {len(rows)}')
    write_csv_atomic(paths[1], [{k: v for k, v in row.items() if k not in
                                 {'reward_config', 'graph_mapping'}} for row in rows])
    summary, paired = rows_summary(rows, args.episodes, args.seeds)
    write_csv_atomic(paths[2], summary)
    write_csv_atomic(paths[3], paired)
    print('\n[30/30 COMPLETE] Same held-out seeds; policy is GREEDY. No PPO updates.', flush=True)
    for s in summary:
        print(f'EP{s["checkpoint_ep"]}: JobDone={s["jobDone_mean"]:.2f} '
              f'Block={s["block_mean"]:.2f} LiveLock={s["liveLock_mean"]:.2f} '
              f'BlockTime={s["block_time_s_mean"]:.1f}s Stop={s["stop_ratio_pct_mean"]:.1f}%', flush=True)
    for p in paths:
        print('RESULT:', p, flush=True)


def main(argv=None):
    run_evaluation(parse_args(argv))


if __name__ == '__main__':
    main()
