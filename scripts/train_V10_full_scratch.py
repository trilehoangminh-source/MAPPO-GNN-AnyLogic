"""V10-D full scratch: TRAIN ALL roles from random initialization, EP0 checkpoint.

Uses V8 run_config.json ONLY as a configuration blueprint to reproduce the
exact AnyLogic model, graph, rewards, PPO configuration and action masks.
No pretrained model weights/optimizers/ValueNorm are loaded at initialization.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from v10_pure_common import read_json,sha256,save_json_atomic,source_config
from v10_full_scratch_common import setup_from_scratch


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-run',type=Path,default=ROOT/'mappo_runs'/'V8_RESIDUAL_EDGEGAT_JOBDONE2_FROM205',help='V8 JSON CONFIG ONLY; no V8 checkpoint loading')
    p.add_argument('--output',type=Path,default=ROOT/'mappo_runs'/'V10_D_FULL_SCRATCH_2LAYERS')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51880)
    p.add_argument('--hours',type=float,default=0.5)
    p.add_argument('--episodes',type=int,default=100,help='Total new episodes starting at EP1')
    p.add_argument('--sim-seconds',type=int,default=1800)
    p.add_argument('--max-events',type=int,default=None)
    p.add_argument('--hidden',type=int,default=32)
    p.add_argument('--heads',type=int,default=2)
    p.add_argument('--layers',type=int,default=2)
    p.add_argument('--max-minibatch',type=int,default=64,help='64 recommended for TRAINING Routing Edge-GAT on 4GB CUDA; 128 may OOM')
    p.add_argument('--snapshot-every',type=int,default=10)
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    return p


def metrics_preflight(path, last):
    rows=[json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    good=[r for r in rows if int(r['episode'])<=last]
    if [int(r['episode']) for r in good]!=list(range(1,last+1)):
        raise RuntimeError('Scratch metrics mismatch committed episode range')
    if len(rows)!=len(good):
        # Trim uncommitted metric if interrupted after metrics append before state write.
        path.write_text(''.join(json.dumps(x)+'\n' for x in good),encoding='utf-8')


def main(argv=None):
    a=parser().parse_args(argv)
    if not (0<a.hours<=72 and a.episodes>=1 and a.sim_seconds>=1 and a.max_minibatch>=1 and a.snapshot_every>=1 and a.cpu_threads>=1 and a.hidden>=8 and a.heads>0 and a.layers>=1 and a.hidden%a.heads==0):
        raise ValueError('Invalid training arguments')
    out=a.output.resolve()
    if a.resume:
        meta=read_json(out/'experiment.json')
        cfg=read_json(out/'run_config_blueprint.json')
        if sha256(out/'run_config_blueprint.json')!=meta['run_config_sha256']:
            raise RuntimeError('Blueprint config changed since scratch began')
        if (a.hidden,a.heads,a.layers,a.sim_seconds,a.max_minibatch,a.snapshot_every,a.cpu_threads)!=tuple(meta[k] for k in ('hidden','heads','layers','sim_seconds','max_minibatch','snapshot_every','cpu_threads')):
            raise RuntimeError('Resume flags differ from original run; use same --layers, minibatch etc')
    else:
        cfg=source_config(a.config_run.resolve())
        cfg_path=a.config_run.resolve()/'run_config.json'
        meta=dict(version='MAPPO_V10_D_FULL_SCRATCH',architecture='pure_transformer_dispatch_and_critic',
                  routing='residual_edge_gat_random_init',source_checkpoint_loaded=False,
                  trained_roles=['routing','dispatch_front','dispatch_rear'],
                  baseline_config_path=str(cfg_path),run_config_sha256=sha256(cfg_path),
                  seed=int(cfg['seed']),graph_fingerprint=cfg['graph_fingerprint'],
                  reward_part_completed=2.0,guide_strength=1.5,
                  fleetsize=20,hidden=a.hidden,heads=a.heads,layers=a.layers,
                  sim_seconds=a.sim_seconds,
                  max_events=int(cfg['max_events'] if a.max_events is None else a.max_events),
                  max_minibatch=a.max_minibatch,snapshot_every=a.snapshot_every,cpu_threads=a.cpu_threads)
    if a.resume:
        if a.max_events is not None and a.max_events!=int(meta['max_events']):
            raise RuntimeError('Cannot change max-events during resume')
        if a.config_run.resolve()!=Path(meta['baseline_config_path']).parent and (a.config_run.resolve()/'run_config.json').exists():
            print('[NOTICE] Resuming with embedded immutable blueprint, not current V8 run')
    if a.dry_run:
        print('[FULL SCRATCH DRY RUN] initial EP=0; first rollout EP=1')
        print('[FULL SCRATCH DRY RUN] no V8 .pt loaded, all three actor+critic roles initialized anew')
        print('[FULL SCRATCH DRY RUN] blueprint from',a.config_run if not a.resume else out/'run_config_blueprint.json')
        print('[FULL SCRATCH DRY RUN] layers=',a.layers,'heads=',a.heads,'minibatch=',a.max_minibatch,'output=',out)
        return
    import torch
    import numpy as np
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.routing_guidance_v121 import GuidedEventCollector
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit,verify_ppo_rollout_parity
    from thesis.mappo.mappo_stability_v7 import audit_rollout_roles
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu to smoke test')
    torch.set_num_threads(a.cpu_threads)
    os.environ['MAPPO_DISPATCH_MASK_MODE']='operation_first'
    os.environ['MAPPO_MAX_MINIBATCH']=str(a.max_minibatch)
    random.seed(int(cfg['seed']));np.random.seed(int(cfg['seed']));torch.manual_seed(int(cfg['seed']))
    state_file=out/'state.json';metrics_file=out/'metrics.jsonl'
    if a.resume:
        state=read_json(state_file)
        cp=Path(state['checkpoint']).resolve()
        if cp.parent!=out or cp.name not in {'rolling_slot_0.pt','rolling_slot_1.pt','checkpoint_ep_0000.pt'} or not cp.is_file():
            raise RuntimeError('Unsafe scratch resume checkpoint')
        metrics_preflight(metrics_file,int(state['last_episode']))
        print('[FULL SCRATCH RESUME] from EP',state['last_episode'],flush=True)
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Scratch output dir already contains files: '+str(out))
        out.mkdir(parents=True,exist_ok=True)
        shutil.copy2(a.config_run.resolve()/'run_config.json',out/'run_config_blueprint.json')
        save_json_atomic(out/'experiment.json',meta)
        state=dict(last_episode=0,checkpoint=str((out/'checkpoint_ep_0000.pt').resolve()),snapshots=[],run_seconds=0)
        metrics_file.touch(exist_ok=False)
        cp=None
    env=None;started=time.monotonic();last_duration=0.
    try:
        env,model,spec=setup_from_scratch(cfg,a.device,a.port,a.sim_seconds,int(meta['max_events']),a.hidden,a.heads,a.layers,checkpoint=cp)
        if cp is None:
            model.save(str(out/'checkpoint_ep_0000.pt'))
            save_json_atomic(state_file,state)
            print('[FULL SCRATCH CREATED] EP0 random weights saved; no V8 pretrained weights',flush=True)
        auditor=RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        collector=GuidedEventCollector(model,env,stop_mask_mode=cfg['stop_mask_mode'],routing_guide_strength=1.5)
        while int(state['last_episode'])<a.episodes:
            if time.monotonic()-started+max(last_duration,0)>=a.hours*3600:
                print('[FULL SCRATCH] wall-clock budget ended between episodes',flush=True)
                break
            ep=int(state['last_episode'])+1
            st=time.monotonic()
            random.seed(int(cfg['seed'])+ep)
            np.random.seed(int(cfg['seed'])+ep)
            torch.manual_seed(int(cfg['seed'])+ep)
            auditor.reset()
            trajectories,env_meta=collector.episode(seed=int(cfg['seed'])+ep-1,train=True,max_event_steps=int(meta['max_events']))
            if GraphSpec.from_hive(env.agent_hive).fingerprint!=spec.fingerprint:
                raise RuntimeError('Graph topology changed from baseline')
            graph=auditor.report()
            if float(graph['candidate_map_rate'])<.99:
                raise RuntimeError('Routing candidate mapping less than 99%')
            parity=verify_ppo_rollout_parity(model,trajectories,max_samples=64)
            role_audit=audit_rollout_roles(model,trajectories,max_per_role=64)
            updates=model.train(trajectories) # ALL roles are trained, including Routing.
            slot=out/f'rolling_slot_{ep%2}.pt'
            tmp=slot.with_suffix('.pt.tmp')
            model.save(str(tmp));os.replace(tmp,slot)
            row=dict(version='V10_D_FULL_SCRATCH',episode=ep,environment_seed=int(cfg['seed'])+ep-1,
                     environment=env_meta,graph_mapping=graph,ppo_audit=parity,
                     all_roles_audit=role_audit,policy_updates=updates)
            with metrics_file.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,default=str)+'\n');stream.flush();os.fsync(stream.fileno())
            state['checkpoint']=str(slot.resolve());state['last_episode']=ep
            state['run_seconds']=round(float(state['run_seconds'])+time.monotonic()-st,2)
            if ep%a.snapshot_every==0:
                snap=out/f'snapshot_ep_{ep:04d}.pt'
                snap_temp=snap.with_suffix('.pt.tmp')
                shutil.copy2(slot,snap_temp);os.replace(snap_temp,snap)
                state['snapshots'].append(str(snap.resolve()))
            save_json_atomic(state_file,state)
            last_duration=time.monotonic()-st
            stats=env_meta.get('statistics',{})
            print(f'[FULL SCRATCH EP{ep}] JobDone={stats.get("jobDone",0)} WrongTarget={stats.get("wrongTarget",0)} '
                  f'Block={stats.get("block",0)} LiveLock={stats.get("liveLock",0)} '
                  f'Reward={env_meta.get("delivered_reward",0):.1f} '
                  f'RoutingUpdates={updates.get("routing",{}).get("optimizer_steps",0)} '
                  f'DispatchFrontUpdates={updates.get("dispatch_front",{}).get("optimizer_steps",0)}',flush=True)
    except KeyboardInterrupt:
        print('[FULL SCRATCH INTERRUPTED] last committed episode',state['last_episode'],flush=True)
        return
    finally:
        if env is not None:env.close()
    if int(state['last_episode'])>0:
        dst=out/'latest_mappo.pt';tmp=dst.with_suffix('.pt.tmp')
        shutil.copy2(state['checkpoint'],tmp);os.replace(tmp,dst)
    print('[FULL SCRATCH DONE] last committed episode',state['last_episode'],flush=True)

if __name__=='__main__':
    main()
