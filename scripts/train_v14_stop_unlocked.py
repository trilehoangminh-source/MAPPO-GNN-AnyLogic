"""V14 Stop-Unlock: continue an existing V14 checkpoint with optional Stop (action 0).

No weights or optimization state are reset. This is a NEW EXPERIMENT BRANCH
from the last COMMITTED original V14 checkpoint, not a full-scratch training.
The only change in behavior policy is Routing STOP masking: original AnyLogic
mask is honored; Stop is NOT forcibly disabled when movement is possible.
PPO rollout, stored action_mask, and PPO updates use the exact SAME mask.
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

VERSION = 'V14_GAT4_CANDIDATE_MLP_STOP_UNLOCK_CONTINUE'
SOURCE_VERSION = 'V14_GAT4_CANDIDATE_MLP_ALL_MLP_CRITICS_SCRATCH'
STOP_MODE = 'baseline'
DEFAULT_OUT = ROOT/'mappo_runs'/'V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK'


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fork-from',type=Path,default=None,
                   help='Original V14 run folder with state.json, experiment.json, and committed checkpoint')
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51880)
    p.add_argument('--sim-seconds',type=int,default=3600)
    p.add_argument('--max-events',type=int,default=None,
                   help='Defaults to the EXACT max_events used by the source V14 run')
    p.add_argument('--max-episodes',type=int,default=1100,help='Absolute last episode (e.g. 1100), continuing after source episode')
    p.add_argument('--max-minibatch',type=int,default=128,
                   help='Must match the source run mini-batch ceiling on fork')
    p.add_argument('--gat-layers',type=int,default=4,help='V14 fixed experimental condition: 4 Edge-GAT layers')
    p.add_argument('--gat-heads',type=int,default=2)
    p.add_argument('--dispatch-width',type=int,default=128)
    p.add_argument('--embedding',type=int,default=64)
    p.add_argument('--station-id-dim',type=int,default=16)
    p.add_argument('--init-only',action='store_true',help='Create a new fork with inherited checkpoint, but do not train')
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


def metrics_preflight(path, last_ep, base_episode):
    if not path.is_file():
        raise FileNotFoundError('Missing fork metrics.jsonl')
    all_rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    kept = [r for r in all_rows if int(r['episode']) <= last_ep]
    if [int(r['episode']) for r in kept] != list(range(base_episode+1,last_ep+1)):
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
        version=VERSION, source_checkpoint_loaded=True, optimizer_source='inherited_from_v14',
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
        stop_mask_mode=STOP_MODE, original_blueprint_stop_mask_mode=cfg['stop_mask_mode'],
        stop_mask_override=STOP_MODE,ppo_v7_options=cfg['ppo_v7_options'])


def main(argv=None):
    a=parse_args(argv)
    out=a.output.resolve()
    manifest_path=out/'experiment.json'
    state_path=out/'state.json'
    metrics_path=out/'metrics.jsonl'
    blueprint_copy=out/'run_config_blueprint.json'
    if a.resume:
        if a.fork_from is not None:
            raise ValueError('Use --fork-from only to INITIALIZE; use --resume for the new run')
        meta=read_json(manifest_path)
        if (meta.get('version') != VERSION or meta.get('stop_mask_mode') != STOP_MODE
                or meta.get('source_checkpoint_loaded') is not True):
            raise RuntimeError('Not a V14 STOP-UNLOCK branch; refusing incompatible resume')
        cfg=read_json(blueprint_copy)
        if sha256(blueprint_copy)!=meta['source_blueprint_sha256']:
            raise RuntimeError('Copied environment blueprint changed')
        if sha256(Path(cfg['model_file']).resolve())!=meta['anylogic_sha256']:
            raise RuntimeError('AnyLogic ZIP changed')
        a.seed=int(meta['seed']) if a.seed is None else a.seed
        a.max_events=int(meta['max_events']) if a.max_events is None else a.max_events
        options=('seed','max_events','sim_seconds','max_minibatch','gat_layers',
                 'gat_heads','dispatch_width','embedding','station_id_dim',
                 'cpu_threads','snapshot_every','device','port')
        mismatches={k:(meta[k],getattr(a,k)) for k in options if meta[k]!=getattr(a,k)}
        if mismatches:
            raise RuntimeError('Resume flags differ from fork initialization: '+str(mismatches))
        state=read_json(state_path)
        current_ep,checkpoint=check_state(state,out)
        metrics_preflight(metrics_path,current_ep,int(meta['fork_episode']))
        source_cp=None
    else:
        if a.fork_from is None:
            raise RuntimeError('Specify --fork-from PATH and --init-only for initial branch creation')
        source=a.fork_from.resolve()
        if source==out:
            raise RuntimeError('Fork destination MUST differ from original V14 run')
        original_meta=read_json(source/'experiment.json')
        if original_meta.get('version') != SOURCE_VERSION:
            raise RuntimeError('Source is not an original V14 full-scratch run')
        if original_meta.get('stop_mask_mode') != 'forbid_when_move':
            raise RuntimeError('Source is not the expected Stop-forbidden experiment')
        cfg_path=source/'run_config_blueprint.json'
        cfg=read_json(cfg_path)
        if sha256(cfg_path)!=original_meta['source_blueprint_sha256']:
            raise RuntimeError('Original V14 blueprint checksum mismatch')
        if sha256(Path(cfg['model_file']).resolve())!=original_meta['anylogic_sha256']:
            raise RuntimeError('Original AnyLogic ZIP checksum mismatch')
        if cfg.get('stop_mask_mode')!='forbid_when_move':
            raise RuntimeError('Unexpected source blueprint Stop mask mode')
        original_state=read_json(source/'state.json')
        current_ep,source_cp=check_state(original_state,source)
        # Original source metrics may have an uncommitted tail after Ctrl+C;
        # verify committed range without touching or rewriting the original run.
        source_rows=[json.loads(line) for line in (source/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        source_committed=[int(row['episode']) for row in source_rows if int(row['episode'])<=current_ep]
        if source_committed != list(range(1,current_ep+1)):
            raise RuntimeError('Original V14 source metrics missing/duplicated for committed episode range')
        if current_ep<1:
            raise RuntimeError('Source has not trained even one episode; use original scratch runner')
        # Keep architecture, PPO, simulation, batch, and RNG unchanged.
        a.seed=int(original_meta['seed']) if a.seed is None else a.seed
        a.max_events=int(original_meta['max_events']) if a.max_events is None else a.max_events
        options=('seed','max_events','sim_seconds','max_minibatch','gat_layers',
                 'gat_heads','dispatch_width','embedding','station_id_dim',
                 'cpu_threads','snapshot_every','device','port')
        mismatches={k:(original_meta[k],getattr(a,k)) for k in options if original_meta[k]!=getattr(a,k)}
        if mismatches:
            raise RuntimeError('Fork args differ from original V14 setup: '+str(mismatches))
        if int(original_meta['fleetsize'])!=20 or int(original_meta['gat_layers'])!=4:
            raise RuntimeError('Expected 20 AGVs and 4 GAT layers')
        meta=manifest_from(cfg,cfg_path,a)
        meta.update(fork_from_run=str(source),fork_episode=current_ep,
                    source_checkpoint=str(source_cp),source_checkpoint_sha256=sha256(source_cp),
                    source_model_version=original_meta['version'],
                    parent_training_episodes=current_ep,
                    train_started_at_episode=current_ep+1,
                    explanation='Inherited trained V14 weights, Adam and ValueNorm; ONLY changed Routing Stop mask after episode boundary')
        checkpoint=source_cp
        state=dict(last_episode=current_ep,
                   checkpoint=str((out/f'rolling_slot_{current_ep%2}.pt').resolve()),
                   snapshots=[],run_seconds=0.0,
                   fork_episode=current_ep,
                   fork_from_run=str(source))
    if a.dry_run:
        print('[V14 STOP-UNLOCK DRY RUN] Continuing from trained V14 episode',current_ep,'new mask=',STOP_MODE)
        print('[V14 STOP-UNLOCK DRY RUN] 20 AGV; sim_seconds=',a.sim_seconds,'max_events=',a.max_events)
        print('[V14 STOP-UNLOCK DRY RUN] GAT4 no Dijkstra; candidate-aware MLP; three 128-128-64 critics')
        print('[V14 STOP-UNLOCK DRY RUN] Original model/optimizers/ValueNorm loaded; no EP0 reset')
        print('[V14 STOP-UNLOCK DRY RUN] origin model=',cfg['model_file'],'| output=',out)
        print('[V14 STOP-UNLOCK DRY RUN] device=',a.device,'minibatch=',a.max_minibatch,'last EP=',current_ep,'target=',a.max_episodes)
        return
    if not a.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError('STOP-UNLOCK output not empty; original and fork must remain separate: '+str(out))
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
            # Commit ONLY after loading the original model into AnyLogic successfully.
            out.mkdir(parents=True,exist_ok=True)
            shutil.copy2(cfg_path,blueprint_copy)
            # Preserve optimizer and ValueNorm in the inherited checkpoint; no reset.
            shutil.copy2(source_cp,out/f'rolling_slot_{current_ep%2}.pt')
            if sha256(out/f'rolling_slot_{current_ep%2}.pt') != meta['source_checkpoint_sha256']:
                raise RuntimeError('Inherited checkpoint copy checksum mismatch')
            atomic_json(manifest_path,meta)
            metrics_path.touch(exist_ok=False)
            atomic_json(state_path,state)
            checkpoint=Path(state['checkpoint'])
            print(f'[V14 STOP UNLOCK FORK] inherited EP{current_ep} checkpoint; Stop allowed when moves exist',flush=True)
        else:
            print(f'[V14 STOP UNLOCK RESUME] loaded committed EP{current_ep}: {checkpoint}',flush=True)
        if a.init_only:
            print('[V14 STOP UNLOCK FORK ONLY] checkpoint inherited, no PPO rollout; output',out,flush=True)
            return
        auditor=RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        collector=V14EventCollector(model,env,stop_mask_mode=STOP_MODE)
        if collector.stop_mask_mode != 'baseline':
            raise RuntimeError('STOP UNLOCK mask was not installed')
        print('[V14 STOP UNLOCK READY] GAT4 + candidate-aware MLP + 3 critics; Stop enabled even if movement exists; NO Dijkstra; 3600s',flush=True)
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
                     sim_seconds=a.sim_seconds,architectures=info,
                     stop_mask_mode=STOP_MODE,fork_episode=int(meta['fork_episode']),
                     checkpoint_origin_sha256=meta['source_checkpoint_sha256'])
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
        print('[V14 STOP UNLOCK INTERRUPTED] Last committed EP',state['last_episode'],flush=True)
    finally:
        if env is not None: env.close()
    print('[V14 STOP UNLOCK DONE] Committed EP',state['last_episode'],'output',out,flush=True)

if __name__=='__main__':
    main()
