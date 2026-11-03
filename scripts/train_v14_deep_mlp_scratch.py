"""V14: 20 AGVs, full-scratch 3600s; GAT4 + candidate-aware MLP Dispatch + 3-layer MLP Critics.

Start EP0 with random actor/critic weights + fresh Adam/ValueNorm;
EP1 is the FIRST rollout/update. Source V8 run_config.json is an immutable
ENVIRONMENT blueprint ONLY: NO V8/V11/V12 model .pt is ever read.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from v14_deep_mlp_common import build_v14
def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path, value):
    path=Path(path)
    tmp=Path(str(path)+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    os.replace(tmp,path)

VERSION = 'V14_GAT4_CANDIDATE_MLP_ALL_MLP_CRITICS_SCRATCH'
DEFAULT_OUT = ROOT/'mappo_runs'/'V14_GAT4_CANDIDATE_MLP_FULL_SCRATCH_3600S'
DEFAULT_BLUEPRINT = ROOT/'mappo_runs'/'V8_RESIDUAL_EDGEGAT_JOBDONE2_FROM205'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config-run',type=Path,default=DEFAULT_BLUEPRINT,
                   help='V8 run_config.json ONLY; no source checkpoint loaded')
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51880)
    p.add_argument('--sim-seconds',type=int,default=3600)
    p.add_argument('--max-events',type=int,default=None,
                   help='Default: twice original V8 1800s event ceiling')
    p.add_argument('--max-episodes',type=int,default=1100,help='Absolute last episode; EP0 is untrained initialization')
    p.add_argument('--max-minibatch',type=int,default=16,
                   help='Minibatch ceiling for GPU; 32 conservative starting point')
    p.add_argument('--gat-layers',type=int,default=4,help='V14 fixed experimental condition: 4 Edge-GAT layers')
    p.add_argument('--gat-heads',type=int,default=2)
    p.add_argument('--dispatch-width',type=int,default=128)
    p.add_argument('--embedding',type=int,default=64)
    p.add_argument('--station-id-dim',type=int,default=16)
    p.add_argument('--init-only',action='store_true',help='Initialize/commit EP0 and exit without training')
    p.add_argument('--seed',type=int,default=None,help='Default original environment seed from blueprint')
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--snapshot-every',type=int,default=10)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    args=p.parse_args(argv)
    if (args.port<1 or args.port>65535 or args.sim_seconds<1 or
        args.max_episodes<1 or args.max_minibatch<1 or args.cpu_threads<1 or
        args.snapshot_every<1 or args.gat_layers!=4 or args.gat_heads<1 or
        args.dispatch_width<8 or args.embedding<8 or args.station_id_dim<1 or
        (args.max_events is not None and args.max_events<1)):
        p.error('Invalid port, architecture, time, batch, or episode arguments')
    return args


def metrics_preflight(path, last_ep):
    if not path.is_file():
        raise FileNotFoundError('Missing scratch metrics.jsonl')
    all_rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    kept = [r for r in all_rows if int(r['episode']) <= last_ep]
    if [int(r['episode']) for r in kept] != list(range(1,last_ep+1)):
        raise RuntimeError('Missing/duplicated committed metrics in V14 run')
    if len(kept) != len(all_rows):
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in kept),encoding='utf-8')


def check_state(st, out):
    ep=int(st['last_episode'])
    cp=Path(st['checkpoint']).resolve()
    if cp.parent != out:
        raise RuntimeError('V14 checkpoint outside its own run folder')
    expected = 'checkpoint_ep_0000.pt' if ep==0 else f'rolling_slot_{ep%2}.pt'
    if cp.name != expected or not cp.is_file():
        raise RuntimeError('Missing/inconsistent V14 committed checkpoint '+str(cp))
    return ep,cp


def manifest_from(cfg, cfg_path, args):
    return dict(
        version=VERSION, source_checkpoint_loaded=False, optimizer_source='fresh',
        source_blueprint_sha256=sha256(cfg_path), source_blueprint_path=str(cfg_path),
        anylogic_model=str(Path(cfg['model_file']).resolve()),
        anylogic_sha256=sha256(Path(cfg['model_file']).resolve()),
        graph_fingerprint=cfg['graph_fingerprint'],
        routing_actor='4-layer V12 observed-only congestion residual Edge-GAT (NO Dijkstra)',
        dispatch_front_actor='Deep candidate-aware per-station MLP (3 hidden layers)',
        dispatch_rear_actor='Deep candidate-aware per-station MLP (3 hidden layers)',
        routing_critic='global pooled Deep MLP 128-128-64',
        dispatch_critics='global pooled Deep MLP 128-128-64',
        train_roles=['routing','dispatch_front','dispatch_rear'],
        static_shortest_path_guidance=False, predicted_shortest_path_flow=False,
        traffic_features=['edge_occupancy','observed_edge_intent','upstream_waiting',
                          'downstream_load','pressure','observed_contraflow'],
        reward_part_completed=2.0, fleetsize=20, max_fleetsize=20,
        sim_seconds=int(args.sim_seconds),max_events=int(args.max_events),
        gat_layers=int(args.gat_layers),gat_heads=int(args.gat_heads),
        dispatch_width=int(args.dispatch_width),embedding=int(args.embedding),
        station_id_dim=int(args.station_id_dim),
        max_minibatch=int(args.max_minibatch),cpu_threads=int(args.cpu_threads),
        snapshot_every=int(args.snapshot_every),device=str(args.device),port=int(args.port),
        seed=int(args.seed), dispatch_mask='operation_first',
        stop_mask_mode=cfg['stop_mask_mode'],ppo_v7_options=cfg['ppo_v7_options'])


def main(argv=None):
    a=parse_args(argv)
    out=a.output.resolve()
    manifest_path=out/'experiment.json'
    state_path=out/'state.json'
    metrics_path=out/'metrics.jsonl'
    blueprint_copy=out/'run_config_blueprint.json'
    if a.resume:
        meta=read_json(manifest_path)
        if meta.get('version') != VERSION or meta.get('source_checkpoint_loaded') is not False:
            raise RuntimeError('Not a V14 full-scratch run')
        cfg=read_json(blueprint_copy)
        if sha256(blueprint_copy)!=meta['source_blueprint_sha256']:
            raise RuntimeError('V14 blueprint changed since scratch initialization')
        if sha256(Path(cfg['model_file']).resolve())!=meta['anylogic_sha256']:
            raise RuntimeError('AnyLogic ZIP changed')
        a.seed = int(meta['seed']) if a.seed is None else a.seed
        a.max_events = int(meta['max_events']) if a.max_events is None else a.max_events
        options=('seed','max_events','sim_seconds','max_minibatch','gat_layers',
                 'gat_heads','dispatch_width','embedding','station_id_dim',
                 'cpu_threads','snapshot_every','device','port')
        mismatches={k:(meta[k],getattr(a,k)) for k in options if meta[k]!=getattr(a,k)}
        if mismatches:
            raise RuntimeError('Resume flags differ from V14 initial run: '+str(mismatches))
        state=read_json(state_path)
        current_ep, checkpoint=check_state(state,out)
        metrics_preflight(metrics_path,current_ep)
    else:
        cfg_path=(a.config_run.resolve()/'run_config.json')
        cfg=read_json(cfg_path)
        if int(cfg['fleetsize'])!=20 or int(cfg['max_fleetsize'])!=20:
            raise RuntimeError('Requires exactly 20 AGV blueprint')
        if cfg.get('routing_encoder')!='residual_edge_gat' or cfg.get('dispatch_mask_mode')!='operation_first':
            raise RuntimeError('Blueprint is not compatible with V12/V13/V14')
        a.seed=int(cfg['seed']) if a.seed is None else a.seed
        a.max_events=int(cfg['max_events'])*2 if a.max_events is None else a.max_events
        meta=manifest_from(cfg,cfg_path,a)
        current_ep=0;checkpoint=None
    if a.dry_run:
        print('[V14 DRY RUN] Full scratch, EP0 random initialization, EP1 first rollout')
        print('[V14 DRY RUN] 20 AGV; 3600s recommended; sim_seconds=',a.sim_seconds,'max_events=',a.max_events)
        print('[V14 DRY RUN] 4-layer V12 Congestion Edge-GAT NO Dijkstra; candidate-aware MLP Dispatch Front/Rear; 3 MLP critics (128-128-64)')
        print('[V14 DRY RUN] No source model checkpoint loaded; all 3 actors + 3 critics train')
        print('[V14 DRY RUN] origin model=',cfg['model_file'],'| output=',out)
        print('[V14 DRY RUN] device=',a.device,'minibatch=',a.max_minibatch,'last EP=',current_ep,'target=',a.max_episodes)
        return
    if not a.resume:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('V14 output not empty; do NOT overwrite an earlier run: '+str(out))
        state=dict(last_episode=0,checkpoint=str((out/'checkpoint_ep_0000.pt').resolve()),
                   snapshots=[],run_seconds=0.0)
    if a.max_episodes <= current_ep and not a.init_only:
        print('[V14 DONE] already EP',current_ep)
        return
    import numpy as np
    import torch
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
    from thesis.mappo.mappo_stability_v7 import audit_rollout_roles
    from v12_clean_audit import verify_v12_ppo_rollout_parity
    from v14_deep_mlp_collector import V14EventCollector
    if str(a.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Requested CUDA unavailable; no silent fallback')
    os.environ['MAPPO_MAX_MINIBATCH']=str(a.max_minibatch)
    os.environ['OMP_NUM_THREADS']=str(a.cpu_threads)
    os.environ['MKL_NUM_THREADS']=str(a.cpu_threads)
    os.environ['OPENBLAS_NUM_THREADS']=str(a.cpu_threads)
    os.environ['MAPPO_DISPATCH_MASK_MODE']='operation_first'
    torch.set_num_threads(int(a.cpu_threads))
    logging.getLogger('alpyne.data').setLevel(logging.ERROR)
    logging.getLogger('alpyne.sim').setLevel(logging.ERROR)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    env=None
    try:
        env,model,spec,info=build_v14(cfg,a.device,a.port,a.sim_seconds,a.max_events,
                                       gat_layers=a.gat_layers,gat_heads=a.gat_heads,
                                       dispatch_width=a.dispatch_width,embedding=a.embedding,
                                       station_id_dim=a.station_id_dim,checkpoint=checkpoint)
        if not a.resume:
            # Commit new run only AFTER model and AnyLogic were initialized.
            # A failed environment startup must not poison an empty output dir.
            out.mkdir(parents=True,exist_ok=True)
            shutil.copy2(cfg_path,blueprint_copy)
            atomic_json(manifest_path,meta)
            metrics_path.touch(exist_ok=False)
            model.save(str(out/'checkpoint_ep_0000.pt'))
            atomic_json(state_path,state)
            print('[V14 EP0] randomly initialized checkpoint saved, no V8/V11/V12 .pt loaded',flush=True)
        else:
            print(f'[V14 RESUME] loaded committed EP{current_ep}: {checkpoint}',flush=True)
        if a.init_only:
            print('[V14 INIT ONLY] EP0 initialized/loaded; no PPO rollout; output',out,flush=True)
            return
        auditor=RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        collector=V14EventCollector(model,env,stop_mask_mode=cfg['stop_mask_mode'])
        print('[V14 READY] GAT4 + 2 candidate-aware MLP Dispatch actors + 3 deep MLP Critics; NO Dijkstra; 3600s',flush=True)
        while int(state['last_episode']) < a.max_episodes:
            ep=int(state['last_episode'])+1
            seed=int(a.seed)+ep-1
            random.seed(int(a.seed)+ep)
            np.random.seed(int(a.seed)+ep)
            torch.manual_seed(int(a.seed)+ep)
            t0=time.monotonic()
            auditor.reset()
            trajectories, env_meta=collector.episode(seed=seed,train=True,max_event_steps=a.max_events)
            if GraphSpec.from_hive(env.agent_hive).fingerprint != spec.fingerprint:
                raise RuntimeError('AnyLogic graph changed; no PPO update')
            if env_meta.get('end_reason') == 'max_event_steps':
                raise RuntimeError('Episode stopped at max-events before proper termination; increase --max-events and start a NEW run')
            graph_report=auditor.report()
            if graph_report['candidate_map_rate']<.99:
                raise RuntimeError('Routing mapping below 99%')
            parity=verify_v12_ppo_rollout_parity(model,trajectories,max_samples=64)
            role_audit=audit_rollout_roles(model,trajectories,max_per_role=64)
            present={t[0]['role'] for t in trajectories if t}
            if 'routing' not in present:
                raise RuntimeError('No routing transitions; refusing update')
            # PPO updates every role that has transitions, including all three critics.
            updates=model.train(trajectories)
            slot=out/f'rolling_slot_{ep%2}.pt'
            temp=Path(str(slot)+'.tmp')
            model.save(str(temp))
            os.replace(temp,slot)
            elapsed=time.monotonic()-t0
            row=dict(version=VERSION,episode=ep,environment_seed=seed,
                     environment=env_meta,graph_mapping=graph_report,
                     ppo_audit=parity,all_roles_audit=role_audit,
                     policy_updates=updates,roles_present=sorted(present),
                     train_roles='all',max_minibatch=a.max_minibatch,
                     dijkstra_guidance_strength=0.0,
                     predicted_shortest_path_flow=False,traffic_schema='observed_only_v12',
                     sim_seconds=a.sim_seconds,architectures=info)
            with metrics_path.open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,default=str,ensure_ascii=False)+'\n')
                f.flush();os.fsync(f.fileno())
            state['last_episode']=ep
            state['checkpoint']=str(slot.resolve())
            state['run_seconds']=round(float(state['run_seconds'])+elapsed,2)
            if ep%a.snapshot_every==0:
                snap=out/f'snapshot_ep_{ep:04d}.pt'
                tmp=Path(str(snap)+'.tmp')
                shutil.copy2(slot,tmp);os.replace(tmp,snap)
                state['snapshots'].append(str(snap.resolve()))
            atomic_json(state_path,state)
            stats=env_meta.get('statistics',{})
            print(f'[V14 EP{ep}] JobDone={stats.get("jobDone",0)} WrongTarget={stats.get("wrongTarget",0)} '
                  f'Block={stats.get("block",0)} LiveLock={stats.get("liveLock",0)} '
                  f'Reward={env_meta.get("delivered_reward",0):.2f} Elapsed={elapsed:.1f}s',flush=True)
    except KeyboardInterrupt:
        print('[V14 INTERRUPTED] Last committed EP',state['last_episode'],flush=True)
    finally:
        if env is not None: env.close()
    print('[V14 DONE] Committed EP',state['last_episode'],'output',out,flush=True)

if __name__=='__main__':
    main()
