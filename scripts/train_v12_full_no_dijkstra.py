"""V12 CLEAN: restore last V11 EP; remove BOTH shortest-path action prior
and path-predicted traffic flow. Keep V11 trainable GAT/MLP/critic/optimizer state.
Train ALL MAPPO roles through absolute EP1000, in a new independent directory.
V8, V11, and legacy V12 experiment directories are read-only.
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from v11_common import read_json, atomic_json, sha256, validate_v8_cfg
from v12_clean_common import build_v12

VERSION = 'V12_FULL_NO_DIJKSTRA_FROM_V11'
DEFAULT_V11 = ROOT / 'mappo_runs' / 'V11_CONGESTION_FROM_V8_EP304'
DEFAULT_OUT = ROOT / 'mappo_runs' / 'V12_FULL_NO_DIJKSTRA_FROM_V11'


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=DEFAULT_V11,
                   help='V11 source run; used only at initial transfer, not on --resume')
    p.add_argument('--output', type=Path, default=DEFAULT_OUT)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--port', type=int, default=51880)
    p.add_argument('--max-episodes', type=int, default=1000,
                   help='ABSOLUTE episode cap. Checkpoint EP529 -> EP530 .. EP1000')
    p.add_argument('--max-minibatch', type=int, default=64)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--snapshot-every', type=int, default=10)
    p.add_argument('--resume', action='store_true',
                   help='Resume this V12 run only; NEVER resume V11 in-place')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    if not 1 <= args.port <= 65535 or args.max_episodes < 1:
        p.error('Invalid port or episode cap')
    if args.max_minibatch < 1 or not 1 <= args.cpu_threads <= 32 or args.snapshot_every < 1:
        p.error('Invalid minibatch / CPU threads / snapshot frequency')
    return args


def inspect_v11_source(folder: Path):
    folder = folder.resolve()
    src_state = read_json(folder / 'state.json')
    src_meta = read_json(folder / 'experiment.json')
    if src_meta.get('version') != 'V11_CONGESTION_EDGEGAT':
        raise RuntimeError('Source must be the trained V11 Congestion Edge-GAT run')
    if src_meta.get('train_roles') != 'all':
        raise RuntimeError('Source V11 train_roles is not ALL; inspect state before transfer')
    episode = int(src_state['last_episode'])
    if episode < int(src_meta['source_episode']):
        raise RuntimeError('V11 source episode invalid')
    original = Path(src_state['checkpoint']).resolve()
    if not (original.is_file() and original.parent == folder):
        raise RuntimeError('Latest committed V11 checkpoint missing or outside V11 folder')
    if original.name not in ('rolling_slot_0.pt', 'rolling_slot_1.pt'):
        raise RuntimeError('Expected state-confirmed V11 rolling checkpoint')
    cfg_file = Path(src_meta['source_run']).resolve() / 'run_config.json'
    cfg = read_json(cfg_file)
    validate_v8_cfg(cfg)
    v8_file = Path(src_meta['source_checkpoint']).resolve()
    if not v8_file.is_file() or sha256(v8_file) != src_meta['source_sha256']:
        raise RuntimeError('Missing/changed V8 source checkpoint; refusing transfer')
    if sha256(cfg_file) != src_meta['source_run_config_sha256']:
        raise RuntimeError('V8 source run config changed')
    if sha256(Path(cfg['model_file']).resolve()) != src_meta['anylogic_model_sha256']:
        raise RuntimeError('AnyLogic model changed since V11')
    return episode, original, src_meta, cfg, v8_file, cfg_file


def manifest_for_source(folder, ep, src_ckpt, src_meta, cfg, v8_ckpt, cfg_file, opts):
    return dict(
        version=VERSION, source_run=str(folder.resolve()), source_episode=ep,
        source_committed_checkpoint=str(src_ckpt),
        source_checkpoint_sha256=sha256(src_ckpt),
        source_manifest_sha256=sha256(folder / 'experiment.json'),
        v8_source_checkpoint=str(v8_ckpt), v8_source_sha256=sha256(v8_ckpt),
        v8_config_file=str(cfg_file), v8_config_sha256=sha256(cfg_file),
        anylogic_model_sha256=sha256(Path(cfg['model_file']).resolve()),
        graph_fingerprint=cfg['graph_fingerprint'],
        train_roles='all', routing='V11 Congestion-Aware Residual Edge-GAT',
        dispatch='V11 MLP Front/Rear; train both actors and critics',
        dijkstra_action_guidance=0.0, dijkstra_guidance_enabled=False,
        v11_traffic_shortest_path_flow_proxy=False,
        traffic_observation_only=True, original_next_hop_removed=True,
        traffic_features=['edge_occupancy','observed_edge_intent','upstream_waiting',
                          'downstream_load','pressure','observed_contraflow'],
        sim_seconds=int(src_meta['sim_seconds']), max_events=int(src_meta['max_events']),
        device=opts.device, port=opts.port, cpu_threads=opts.cpu_threads,
        max_minibatch=opts.max_minibatch, snapshot_every=opts.snapshot_every,
        seed=int(cfg['seed']), fleet_size=int(cfg['fleetsize']),
    )


def committed_rows(path, source_ep, last_ep):
    if not path.is_file():
        if last_ep > source_ep:
            raise RuntimeError('Missing V12 metrics for completed episodes')
        return []
    raw = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    good = [r for r in raw if int(r['episode']) <= last_ep]
    expected = list(range(source_ep + 1, last_ep + 1))
    if [int(r['episode']) for r in good] != expected:
        raise RuntimeError('V12 metrics missing, duplicated or out of order')
    if len(good) != len(raw):
        # A crash after appending metrics but before atomic state may leave
        # a future uncommitted row; it must not be counted or trained over.
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in good), encoding='utf-8')
    return good


def validate_state(state, meta, out):
    ep = int(state['last_episode'])
    source_ep = int(meta['source_episode'])
    if ep < source_ep:
        raise RuntimeError('V12 state rewound before transfer')
    cp = Path(state['checkpoint']).resolve()
    if cp.parent != out:
        raise RuntimeError('Checkpoint outside V12 output, refusing resume')
    if ep == source_ep:
        if cp.name != f'v11_source_ep_{source_ep:04d}.pt' or sha256(cp) != meta['source_checkpoint_sha256']:
            raise RuntimeError('Copied V11 source checkpoint corrupted')
    elif cp.name != f'rolling_slot_{ep%2}.pt':
        raise RuntimeError('V12 rolling checkpoint name inconsistent with episode')
    if not cp.is_file():
        raise FileNotFoundError(cp)
    return ep, cp


def main(argv=None):
    x = arguments(argv)
    source = x.source_run.resolve()
    output = x.output.resolve()
    if source == output or output in source.parents or source in output.parents:
        raise RuntimeError('V12 must be a separate sibling folder, never overwrite V11')
    manifest_file = output / 'experiment.json'
    state_file = output / 'state.json'
    metrics_file = output / 'metrics.jsonl'
    if x.resume:
        meta = read_json(manifest_file)
        if meta.get('version') != VERSION:
            raise RuntimeError('Not a V12 no-Dijkstra run; refusing resume')
        if source != Path(meta['source_run']).resolve():
            raise RuntimeError('Source run differs from V12 initial source')
        requested = dict(device=x.device, port=x.port, cpu_threads=x.cpu_threads,
                         max_minibatch=x.max_minibatch, snapshot_every=x.snapshot_every)
        mismatch = {k:(meta[k],v) for k,v in requested.items() if meta[k]!=v}
        if mismatch:
            raise RuntimeError(f'V12 resume flags differ from initial run: {mismatch}')
        cfg = read_json(Path(meta['v8_config_file']))
        validate_v8_cfg(cfg)
        if sha256(Path(meta['v8_config_file'])) != meta['v8_config_sha256']:
            raise RuntimeError('V8 configuration changed')
        v8_ckpt = Path(meta['v8_source_checkpoint']).resolve()
        if sha256(v8_ckpt) != meta['v8_source_sha256']:
            raise RuntimeError('V8 checkpoint changed')
        if sha256(Path(cfg['model_file']).resolve()) != meta['anylogic_model_sha256']:
            raise RuntimeError('AnyLogic model changed')
        if cfg['graph_fingerprint'] != meta['graph_fingerprint']:
            raise RuntimeError('Graph mismatch')
        source_ep = int(meta['source_episode'])
        transfer = output / f'v11_source_ep_{source_ep:04d}.pt'
        if sha256(transfer) != meta['source_checkpoint_sha256']:
            raise RuntimeError('Copied V11 source checkpoint changed')
        state = read_json(state_file)
        current_ep, checkpoint = validate_state(state, meta, output)
        committed_rows(metrics_file, source_ep, current_ep)
    else:
        source_ep, v11_checkpoint, source_meta, cfg, v8_ckpt, cfg_file = inspect_v11_source(source)
        if x.max_episodes <= source_ep:
            raise RuntimeError(f'EP target {x.max_episodes} <= source EP {source_ep}')
        meta = manifest_for_source(source, source_ep, v11_checkpoint, source_meta,
                                   cfg, v8_ckpt, cfg_file, x)
        if x.dry_run:
            print('[V12 DRY RUN] source V11 EP',source_ep,'checkpoint:',v11_checkpoint)
            print('[V12 DRY RUN] source_sha256:',meta['source_checkpoint_sha256'])
            print('[V12 DRY RUN] output:',output)
            print('[V12 DRY RUN] train_roles=ALL; no shortest-path guide OR traffic; no frozen parameters')
            print('[V12 DRY RUN] target ABSOLUTE EP',x.max_episodes)
            print('[V12 DRY RUN] NO shortest-path flow/next_hop/bias; observed edge traffic ONLY')
            print('[V12 DRY RUN] AnyLogic:',cfg['model_file'])
            return
        if output.exists() and any(output.iterdir()):
            raise FileExistsError('V12 output already contains files; use --resume or choose other output')
        output.mkdir(parents=True, exist_ok=True)
        transfer = output / f'v11_source_ep_{source_ep:04d}.pt'
        tmp = Path(str(transfer) + '.tmp')
        shutil.copy2(v11_checkpoint, tmp)
        os.replace(tmp, transfer)
        if sha256(transfer) != meta['source_checkpoint_sha256']:
            raise RuntimeError('V11 checkpoint changed during copy; stop V11 train and retry with new V12 folder')
        if (read_json(source / 'state.json')['last_episode'] != source_ep or
                Path(read_json(source / 'state.json')['checkpoint']).resolve() != v11_checkpoint):
            raise RuntimeError('V11 source state changed while transferring; stop concurrent V11 train')
        atomic_json(manifest_file,meta)
        state = dict(last_episode=source_ep,checkpoint=str(transfer.resolve()),
                     snapshots=[],run_seconds=0.0)
        atomic_json(state_file,state)
        current_ep, checkpoint = validate_state(state,meta,output)
    if x.max_episodes <= current_ep:
        print(f'[V12 DONE] completed EP{current_ep}; requested cap EP{x.max_episodes}',flush=True)
        return
    if x.dry_run:
        print(f'[V12 DRY RUN] resume from EP{current_ep} -> EP{x.max_episodes}',flush=True)
        return

    import numpy as np
    import torch
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
    from v12_clean_audit import verify_v12_ppo_rollout_parity
    from thesis.mappo.mappo_stability_v7 import audit_rollout_roles
    from v12_clean_collector import RawRoutingEventCollector

    if x.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable: refusing CPU fallback')
    os.environ['MAPPO_MAX_MINIBATCH'] = str(x.max_minibatch)
    os.environ['OMP_NUM_THREADS'] = str(x.cpu_threads)
    os.environ['MKL_NUM_THREADS'] = str(x.cpu_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(x.cpu_threads)
    torch.set_num_threads(x.cpu_threads)
    logging.getLogger('alpyne.data').setLevel(logging.ERROR)
    logging.getLogger('alpyne.sim').setLevel(logging.ERROR)
    env = None
    try:
        env, model, spec, installation = build_v12(
            cfg, v8_ckpt, x.device, x.port, int(meta['sim_seconds']),
            int(meta['max_events']), checkpoint=checkpoint,
            source_v11=(current_ep == source_ep))
        collector = RawRoutingEventCollector(model,env,stop_mask_mode=cfg['stop_mask_mode'])
        auditor = RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        print(f'[V12 READY] V11 source EP{source_ep}; loaded EP{current_ep}; target EP{x.max_episodes}; '
              f'NO Dijkstra / NO shortest flow; train ALL roles; channels: {installation["channels"]}',flush=True)
        last_sec = 0.0
        while int(state['last_episode']) < x.max_episodes:
            ep = int(state['last_episode']) + 1
            seed = int(cfg['seed']) + ep - 1
            random.seed(int(cfg['seed']) + ep)
            np.random.seed(int(cfg['seed']) + ep)
            torch.manual_seed(int(cfg['seed']) + ep)
            auditor.reset()
            start = time.monotonic()
            trajectories, environment = collector.episode(seed=seed, train=True, max_event_steps=int(meta['max_events']))
            if GraphSpec.from_hive(env.agent_hive).fingerprint != spec.fingerprint:
                raise RuntimeError('AnyLogic graph changed during rollout')
            audit = auditor.report()
            if audit['candidate_map_rate'] < 0.99:
                raise RuntimeError('Routing candidate mapping under 99%; no update')
            parity = verify_v12_ppo_rollout_parity(model,trajectories,max_samples=64)
            role_audit = audit_rollout_roles(model,trajectories,max_per_role=64)
            roles_present = {t[0]['role'] for t in trajectories if t}
            if 'routing' not in roles_present:
                raise RuntimeError('No Routing trajectory in episode')
            # All three roles remain trainable. If an episode has no transitions
            # for a Dispatch role, it cannot update that role in this episode.
            updates = model.train(trajectories)
            slot = output / f'rolling_slot_{ep%2}.pt'
            tmp = Path(str(slot)+'.tmp')
            model.save(str(tmp))
            os.replace(tmp,slot)
            elapsed = time.monotonic()-start
            row = dict(version=VERSION,episode=ep,environment_seed=seed,environment=environment,
                       graph_mapping=audit,ppo_audit=parity,all_roles_audit=role_audit,
                       policy_updates=updates,roles_present=sorted(roles_present),
                       train_roles='all',dijkstra_guidance_strength=0.0,
                       predicted_shortest_path_flow=False,traffic_schema='observed_only_v12',
                       max_minibatch=x.max_minibatch)
            with metrics_file.open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,default=str,ensure_ascii=False)+'\n')
                f.flush()
                os.fsync(f.fileno())
            state['last_episode'] = ep
            state['checkpoint'] = str(slot.resolve())
            state['run_seconds'] = round(state.get('run_seconds',0) + elapsed,2)
            if ep % x.snapshot_every == 0:
                snapshot = output / f'snapshot_ep_{ep:04d}.pt'
                snap_tmp = Path(str(snapshot)+'.tmp')
                shutil.copy2(slot,snap_tmp)
                os.replace(snap_tmp,snapshot)
                state['snapshots'].append(str(snapshot.resolve()))
            atomic_json(state_file,state)
            last_sec = elapsed
            stats = environment.get('statistics',{})
            print(f'[V12 EP{ep}] JobDone={stats.get("jobDone",0)} '
                  f'WrongTarget={stats.get("wrongTarget",0)} Block={stats.get("block",0)} '
                  f'LiveLock={stats.get("liveLock",0)} Reward={environment.get("delivered_reward",0):.2f} '
                  f'Elapsed={last_sec:.1f}s',flush=True)
    except KeyboardInterrupt:
        print('[V12 INTERRUPTED] Last committed EP',state['last_episode'],flush=True)
    finally:
        if env is not None:
            env.close()
    if int(state['last_episode']) > source_ep:
        latest = output / 'latest_v12.pt'
        tmp = Path(str(latest)+'.tmp')
        shutil.copy2(Path(state['checkpoint']),tmp)
        os.replace(tmp,latest)
    print('[V12 DONE] last committed EP',state['last_episode'],'output',output,flush=True)

if __name__=='__main__':
    main()
