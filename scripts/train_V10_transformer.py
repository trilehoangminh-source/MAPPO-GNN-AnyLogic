"""V10 MAPPO Transformer ablations, transfer from V8 Reward+2 checkpoint.

  control: V8 MLP Dispatcher + pooled critic (matched fine-tuning control)
  A:       residual Transformer Dispatch Actor + pooled critic
  B:       V8 MLP Dispatcher + residual Transformer Dispatch Critic
  C:       residual Transformer Dispatch Actor + Dispatch Critic

Routing Actor = unchanged V8 Residual Edge-GAT, Dijkstra 1.5.
Default optimizer trains ONLY Dispatch roles; Routing weights stay fixed.
All branches start from the same V8 checkpoint and use deterministic seeds.
All output/checkpoints are in NEW folders; V8 and V9 remain untouched.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT))

from v10_common import (sha256, read_json, save_json_atomic, source_config,
                        setup_model, filter_roles)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run', type=Path, default=ROOT/'mappo_runs'/'V8_RESIDUAL_EDGEGAT_JOBDONE2_FROM205')
    p.add_argument('--source-checkpoint', type=Path, default=None, help='Default: committed latest V8 checkpoint in source state.json')
    p.add_argument('--source-episode', type=int, default=None, help='Required with custom --source-checkpoint')
    p.add_argument('--variant', choices=['control','A','B','C'], default='A')
    p.add_argument('--output', type=Path, default=None, help='Default: mappo_runs/V10_<variant>_FROM_V8')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--port', type=int, default=51880)
    p.add_argument('--hours', type=float, default=1.0)
    p.add_argument('--episodes', type=int, default=500, help='Maximum NEW episodes since V8 source; can increase on resume')
    p.add_argument('--sim-seconds', type=int, default=1800)
    p.add_argument('--max-events', type=int, default=None)
    p.add_argument('--train-roles', choices=['dispatch','all'], default='dispatch')
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--heads', type=int, default=2)
    p.add_argument('--layers', type=int, default=1)
    p.add_argument('--max-minibatch', type=int, default=32)
    p.add_argument('--snapshot-every', type=int, default=10)
    p.add_argument('--cpu-threads', type=int, default=2)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not 0 < a.hours <= 72 or a.episodes < 1 or a.sim_seconds < 1:
        p.error('Invalid hours, episodes or sim-seconds')
    if a.max_minibatch < 1 or a.snapshot_every < 1 or a.cpu_threads < 1:
        p.error('max-minibatch, snapshot-every, cpu-threads must be positive')
    if a.heads < 1 or a.hidden % a.heads or a.layers < 1:
        p.error('hidden must be divisible by heads; layers>=1')
    if a.source_checkpoint is not None and a.source_episode is None:
        p.error('Explicit --source-checkpoint requires --source-episode')
    if a.output is None:
        a.output = ROOT/'mappo_runs'/f'V10_{a.variant}_FROM_V8'
    return a


def committed_metric_rows(path, last_episode, source_episode):
    if not path.exists():
        if last_episode > source_episode:
            raise RuntimeError('Missing metrics.jsonl on resume')
        return []
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    good = [r for r in rows if int(r['episode']) <= last_episode]
    if good:
        expected = list(range(source_episode+1, last_episode+1))
        if [int(x['episode']) for x in good] != expected:
            raise RuntimeError('Missing or duplicate completed episodes in metrics')
    elif last_episode > source_episode:
        raise RuntimeError('Empty metrics for a committed run')
    if len(rows) != len(good):
        path.write_text(''.join(json.dumps(x)+'\n' for x in good), encoding='utf-8')
    return good


def main():
    a = parse_args()
    out = a.output.resolve()
    manifest_file = out/'experiment.json'
    state_file = out/'state.json'
    metric_file = out/'metrics.jsonl'
    base_file = out/'v8_source_checkpoint.pt'

    # Resume always uses the FROZEN V8 source copied into V10, even if the
    # original V8 run has been trained further or moved after V10 started.
    if a.resume:
        old = read_json(manifest_file)
        cfg = read_json(out/'v8_source_run_config.json')
        source_ep = int(old['source_episode'])
        src_ckpt = base_file
        if sha256(base_file) != old['source_sha256']:
            raise RuntimeError('Copied V8 source checkpoint changed')
        if sha256(out/'v8_source_run_config.json') != old['source_run_config_sha256']:
            raise RuntimeError('Copied V8 source run_config changed')
    else:
        source = a.source_run.resolve()
        cfg = source_config(source)
        source_state = read_json(source/'state.json')
        source_ep = int(source_state['last_episode'] if a.source_episode is None else a.source_episode)
        src_ckpt = Path(source_state['checkpoint'] if a.source_checkpoint is None else a.source_checkpoint).resolve()
        if not src_ckpt.is_file():
            raise FileNotFoundError('Missing source V8 checkpoint: '+str(src_ckpt))
    if source_ep < 1:
        raise ValueError('Source episode must be positive')
    max_events = int(cfg['max_events'] if a.max_events is None else a.max_events)
    if max_events < 1:
        raise ValueError('Invalid max_events')
    cfg_model = Path(cfg['model_file']).resolve()
    if not cfg_model.is_file():
        raise FileNotFoundError('Missing original AnyLogic ZIP: '+str(cfg_model))
    base_settings = dict(
        version='MAPPO_V10_TRANSFORMER_RESIDUAL', variant=a.variant,
        source_episode=source_ep, source_sha256=sha256(src_ckpt),
        source_run_config_sha256=(
            sha256(out/'v8_source_run_config.json') if a.resume else sha256(source/'run_config.json')),
        anylogic_sha256=sha256(cfg_model),
        reward_part_completed=2.0, fleetsize=20,
        architecture_dispatch=('residual_station_transformer' if a.variant in ('A','C') else 'v8_pooled_mlp'),
        architecture_dispatch_critic=('residual_transformer' if a.variant in ('B','C') else 'v8_pooled_mlp'),
        architecture_routing='v8_residual_edge_gat_unchanged',
        train_roles=a.train_roles, hidden=a.hidden, heads=a.heads, layers=a.layers,
        sim_seconds=a.sim_seconds, max_events=max_events,
        fixed_routing_guide=1.5, seed=int(cfg['seed']),
        graph_fingerprint=cfg['graph_fingerprint'],
        ppo_v7_options=cfg['ppo_v7_options'],
    )
    if a.dry_run:
        print('[V10 DRY RUN] Variant',a.variant,'| from V8 episode',source_ep)
        print('[V10 DRY RUN] Checkpoint',src_ckpt)
        print('[V10 DRY RUN] Destination',out)
        print('[V10 DRY RUN] Train roles',a.train_roles,'| device',a.device)
        print('[V10 DRY RUN] Original V8 never overwritten')
        return
    import torch
    import numpy as np
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit, verify_ppo_rollout_parity
    from thesis.mappo.mappo_stability_v7 import audit_rollout_roles
    from thesis.mappo.routing_guidance_v121 import GuidedEventCollector

    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for smoke test')
    torch.set_num_threads(a.cpu_threads)
    os.environ['OMP_NUM_THREADS']=str(a.cpu_threads)
    os.environ['MKL_NUM_THREADS']=str(a.cpu_threads)
    os.environ['MAPPO_MAX_MINIBATCH']=str(a.max_minibatch)
    os.environ['MAPPO_DISPATCH_MASK_MODE']='operation_first'
    random.seed(int(cfg['seed'])); np.random.seed(int(cfg['seed'])); torch.manual_seed(int(cfg['seed']))
    if a.resume:
        existing = read_json(manifest_file)
        if existing != base_settings:
            raise RuntimeError('V10 resume configuration changed; use original flags or a new run directory')
        state = read_json(state_file)
        cp = Path(state['checkpoint']).resolve()
        legal_cp = ('rolling_slot_0.pt','rolling_slot_1.pt') if int(state['last_episode']) > source_ep else ('v8_source_checkpoint.pt',)
        if cp.parent != out or cp.name not in legal_cp or not cp.is_file():
            raise RuntimeError('Unsafe resume: missing/foreign checkpoint')
        if sha256(base_file) != base_settings['source_sha256']:
            raise RuntimeError('V8 source checkpoint copy has changed')
        committed_metric_rows(metric_file, int(state['last_episode']), source_ep)
        print('[V10 RESUME] after EP',state['last_episode'],'using',cp,flush=True)
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('V10 output already contains files: '+str(out))
        out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(src_ckpt, base_file)
        shutil.copy2(source/'run_config.json',out/'v8_source_run_config.json')
        save_json_atomic(manifest_file,base_settings)
        state = dict(last_episode=source_ep, checkpoint=str(base_file.resolve()),
                     start_episode=source_ep, snapshots=[], run_seconds=0.)
        save_json_atomic(state_file,state)
        # Create file immediately; no training metric for source EP.
        metric_file.touch(exist_ok=False)
        print('[V10 START] V8 episode',source_ep,'-> variant',a.variant,'at',out,flush=True)
        cp = None
    env=None
    started=time.monotonic()
    last_duration=0.0
    try:
        env,model,spec,model_args=setup_model(
            cfg,base_file,a.variant,a.device,a.port,a.sim_seconds,max_events,
            a.hidden,a.heads,a.layers,
            v10_checkpoint=(cp if state['last_episode'] > source_ep else None))
        auditor=RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        collector=GuidedEventCollector(model,env,stop_mask_mode=cfg['stop_mask_mode'],routing_guide_strength=1.5)
        while int(state['last_episode']) < source_ep+a.episodes:
            if time.monotonic()-started+max(last_duration,0.) >= a.hours*3600:
                print('[V10] Wall-clock budget reached between episodes',flush=True)
                break
            ep=int(state['last_episode'])+1
            t0=time.monotonic()
            random.seed(int(cfg['seed'])+ep)
            np.random.seed(int(cfg['seed'])+ep)
            torch.manual_seed(int(cfg['seed'])+ep)
            auditor.reset()
            trajectories,meta=collector.episode(seed=int(cfg['seed'])+ep-1,train=True,max_event_steps=max_events)
            if GraphSpec.from_hive(env.agent_hive).fingerprint != spec.fingerprint:
                raise RuntimeError('Live AnyLogic graph differs from V8; no PPO update')
            rep=auditor.report()
            if float(rep['candidate_map_rate']) < .99:
                raise RuntimeError('Routing graph candidate mapping <99%')
            parity=verify_ppo_rollout_parity(model,trajectories,max_samples=64)
            role_audit=audit_rollout_roles(model,trajectories,max_per_role=64)
            train_traj=filter_roles(trajectories,a.train_roles)
            updates=model.train(train_traj)
            slot=out/f'rolling_slot_{ep%2}.pt'
            temp=slot.with_suffix('.pt.tmp')
            model.save(str(temp))
            os.replace(temp,slot)
            record=dict(version='V10',variant=a.variant,episode=ep,
                environment_seed=int(cfg['seed'])+ep-1,environment=meta,
                graph_mapping=rep,ppo_audit=parity,all_roles_audit=role_audit,
                policy_updates=updates,train_roles=a.train_roles)
            with metric_file.open('a',encoding='utf-8') as f:
                f.write(json.dumps(record,default=str)+'\n');f.flush();os.fsync(f.fileno())
            state['checkpoint']=str(slot.resolve())
            state['last_episode']=ep
            state['run_seconds']=round(state.get('run_seconds',0)+time.monotonic()-t0,2)
            if (ep-source_ep)%a.snapshot_every==0:
                snap=out/f'snapshot_ep_{ep:04d}.pt'
                tmp=snap.with_suffix('.pt.tmp')
                shutil.copy2(slot,tmp);os.replace(tmp,snap)
                state['snapshots'].append(str(snap.resolve()))
            save_json_atomic(state_file,state)
            last_duration=time.monotonic()-t0
            stats=meta.get('statistics',{})
            print(f'[V10-{a.variant} EP{ep}] JobDone={stats.get("jobDone",0)} '
                  f'WrongTarget={stats.get("wrongTarget",0)} '
                  f'Block={stats.get("block",0)} LiveLock={stats.get("liveLock",0)} '
                  f'Reward={meta.get("delivered_reward",0):.1f} '
                  f'DispatchKL={updates.get("dispatch_front",{}).get("approx_kl",0):.6f} '
                  f'elapsed={(time.monotonic()-started)/60:.1f}min',flush=True)
    except KeyboardInterrupt:
        print('[V10 INTERRUPTED] Last committed EP',state['last_episode'],flush=True)
        return
    finally:
        if env is not None:
            env.close()
    if state['last_episode'] > source_ep:
        last=out/'latest_mappo.pt'
        tmp=last.with_suffix('.pt.tmp')
        shutil.copy2(state['checkpoint'],tmp);os.replace(tmp,last)
    print('[V10 DONE] Last committed EP',state['last_episode'],'| run',out,flush=True)

if __name__=='__main__':
    main()
