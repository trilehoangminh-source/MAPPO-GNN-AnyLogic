"""Fork exact, held-out-verified EP130 V14 Reward snapshot into a NEW run.

No source run, model, optimizer, or old reward file is overwritten. Uses the
original V14 baseline Stop action mask and all-role MAPPO training. Incremental
safe-yield / persistent-stall shaping changes ONLY Python Routing reward.
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
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
from train_v14_stop_unlocked import read_json, atomic_json, sha256, check_state
from v14_deep_mlp_common import build_v14
from thesis.envs.v14_safe_yield_stall import SafeYieldConfig, SafeYieldStallReward

VERSION = 'V14_SAFE_YIELD_STALL_FROM_EP130_V1'
SOURCE_VERSION = 'V14_GAT4_STOP_UNLOCK_DIRECT_WAIT_SWING_REWARD_V1'
SOURCE_EP = 130
SOURCE_HASH = 'd9060d5478cee62725d7bdd91c8391a1584870cce69a7b3665afe492cd0abb0c'
STOP_MODE = 'baseline'
DEFAULT_SOURCE = ROOT/'mappo_runs'/'V14_GAT4_MLP_3600S_MB128_WAIT_SWING_REWARD'
DEFAULT_OUT = ROOT/'mappo_runs'/'V14_GAT4_MLP_3600S_MB128_SAFE_YIELD_STALL_FROM_EP130'
REWARD_MODULE = ROOT/'thesis'/'envs'/'v14_safe_yield_stall.py'
OLD_REWARD_MODULE = ROOT/'thesis'/'envs'/'v14_reward_shaping.py'


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fork-from',type=Path,default=DEFAULT_SOURCE)
    p.add_argument('--output',type=Path,default=DEFAULT_OUT)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51880)
    p.add_argument('--sim-seconds',type=int,default=3600)
    p.add_argument('--max-events',type=int,default=None)
    p.add_argument('--max-episodes',type=int,default=160,
                   help='Absolute episode (EP131 is first new update), not count')
    p.add_argument('--max-minibatch',type=int,default=128)
    p.add_argument('--gat-layers',type=int,default=4)
    p.add_argument('--gat-heads',type=int,default=2)
    p.add_argument('--dispatch-width',type=int,default=128)
    p.add_argument('--embedding',type=int,default=64)
    p.add_argument('--station-id-dim',type=int,default=16)
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--snapshot-every',type=int,default=10)
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--yield-bonus',type=float,default=0.35)
    p.add_argument('--stall-grace-seconds',type=float,default=45.0)
    p.add_argument('--stall-penalty-per-10s',type=float,default=-0.10)
    p.add_argument('--stall-penalty-cap',type=float,default=-2.0)
    p.add_argument('--swing-penalty',type=float,default=-1.0)
    p.add_argument('--approach-radius',type=float,default=0.05)
    p.add_argument('--pass-displacement',type=float,default=0.008)
    p.add_argument('--position-tolerance',type=float,default=0.002)
    p.add_argument('--init-only',action='store_true')
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    if (a.port<1 or a.port>65535 or a.sim_seconds!=3600 or a.max_minibatch<1
        or a.gat_layers!=4 or a.gat_heads<1 or a.cpu_threads<1 or a.snapshot_every<1
        or a.max_episodes<SOURCE_EP or (a.max_events is not None and a.max_events<1)):
        p.error('Invalid architecture/horizon/port/episode configuration')
    if a.resume and a.init_only:
        p.error('Use --init-only only when creating a new fork')
    config(a)  # Validate shaping parameters immediately, including --dry-run.
    return a


def config(a):
    return SafeYieldConfig(
        yield_bonus=a.yield_bonus,stall_grace_seconds=a.stall_grace_seconds,
        stall_penalty_per_10s=a.stall_penalty_per_10s,
        stall_penalty_cap=a.stall_penalty_cap,swing_penalty=a.swing_penalty,
        approach_radius=a.approach_radius,
        pass_displacement=a.pass_displacement,
        position_tolerance=a.position_tolerance)


def same_environment(a, meta):
    a.seed = int(meta['seed']) if a.seed is None else a.seed
    a.max_events = int(meta['max_events']) if a.max_events is None else a.max_events
    locks = ('seed','max_events','sim_seconds','max_minibatch','gat_layers',
             'gat_heads','dispatch_width','embedding','station_id_dim',
             'cpu_threads','snapshot_every','device','port')
    mismatch = {k:(meta[k],getattr(a,k)) for k in locks if meta[k]!=getattr(a,k)}
    if mismatch:
        raise RuntimeError('Cannot change V14 environment/architecture/PPO flags: '+str(mismatch))


def run_preflight(a):
    out = a.output.resolve()
    if a.resume:
        if a.fork_from != DEFAULT_SOURCE:
            # --fork-from can only name the source at initialization.
            raise ValueError('Do not specify --fork-from when resuming')
        meta=read_json(out/'experiment.json')
        if (meta.get('version')!=VERSION or meta.get('stop_mask_mode')!=STOP_MODE or
            meta.get('source_checkpoint_sha256')!=SOURCE_HASH or
            meta.get('reward_config')!=config(a).metadata() or
            meta.get('reward_implementation_sha256')!=sha256(REWARD_MODULE)):
            raise RuntimeError('Reward fork metadata/code mismatch')
        cfg_path=out/'run_config_blueprint.json'
        cfg=read_json(cfg_path)
        if (sha256(cfg_path)!=meta['source_blueprint_sha256'] or
            sha256(Path(cfg['model_file']).resolve())!=meta['anylogic_sha256']):
            raise RuntimeError('AnyLogic blueprint/model changed')
        same_environment(a,meta)
        state=read_json(out/'state.json')
        last_ep,cp=check_state(state,out)
        lines=[json.loads(s) for s in (out/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        ids=[int(r['episode']) for r in lines]
        if ids!=list(range(SOURCE_EP+1,last_ep+1)):
            raise RuntimeError('Uncommitted/duplicate/missing reward-run metrics. Inspect before resume')
        return cfg,cfg_path,meta,state,last_ep,cp,None
    source=a.fork_from.resolve()
    if source==out or out.is_relative_to(source) or source.is_relative_to(out):
        raise RuntimeError('Source and output must be separate directories')
    meta_old=read_json(source/'experiment.json')
    if (meta_old.get('version')!=SOURCE_VERSION or
        meta_old.get('reward_mode')!='direct_behavior_reward' or
        meta_old.get('stop_mask_mode')!=STOP_MODE or
        int(meta_old['fleetsize'])!=20 or int(meta_old['gat_layers'])!=4 or
        int(meta_old['sim_seconds'])!=3600):
        raise RuntimeError('Source is not expected V14 Stop/Swing Reward 20-AGV run')
    if sha256(OLD_REWARD_MODULE)!=meta_old['reward_implementation_sha256']:
        raise RuntimeError('Old reward implementation differs from source experiment')
    cfg_path=source/'run_config_blueprint.json'
    cfg=read_json(cfg_path)
    if (sha256(cfg_path)!=meta_old['source_blueprint_sha256'] or
        sha256(Path(cfg['model_file']).resolve())!=meta_old['anylogic_sha256']):
        raise RuntimeError('Original AnyLogic blueprint/model changed')
    same_environment(a,meta_old)
    state_old=read_json(source/'state.json')
    if int(state_old['last_episode'])<SOURCE_EP:
        raise RuntimeError('Source run has not reached snapshot EP130')
    cp=source/f'snapshot_ep_{SOURCE_EP:04d}.pt'
    if not cp.is_file() or cp.is_symlink() or cp.resolve().parent!=source:
        raise FileNotFoundError('Missing exact EP130 snapshot: '+str(cp))
    if sha256(cp)!=SOURCE_HASH:
        raise RuntimeError('EP130 snapshot hash differs from 10-seed held-out evaluation')
    rows=[json.loads(x) for x in (source/'metrics.jsonl').read_text(encoding='utf-8').splitlines() if x.strip()]
    if sum(int(x.get('episode',-1))==SOURCE_EP for x in rows)!=1:
        raise RuntimeError('Source training metrics do not identify EP130 exactly once')
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('Output already exists; never overwrite a run: '+str(out))
    meta=dict(meta_old)
    meta.update(version=VERSION,source_model_version=SOURCE_VERSION,
                source_checkpoint_loaded=True,optimizer_source='EP130_full_snapshot',
                fork_episode=SOURCE_EP,train_started_at_episode=SOURCE_EP+1,
                source_checkpoint=str(cp.resolve()),source_checkpoint_sha256=SOURCE_HASH,
                fork_from_run=str(source),stop_mask_mode=STOP_MODE,
                source_reward_config=meta_old['reward_config'],
                source_reward_implementation_sha256=meta_old['reward_implementation_sha256'],
                reward_mode='direct_environment_safe_yield_stall',
                reward_config=config(a).metadata(),
                reward_implementation_sha256=sha256(REWARD_MODULE),
                explanation='EP130 V14 weights/critic/Adam/ValueNorm inherited, Stop unlocked; only Routing safe-yield/time-congestion shaping changes')
    state=dict(last_episode=SOURCE_EP,
               checkpoint=str((out/f'rolling_slot_{SOURCE_EP%2}.pt').resolve()),
               snapshots=[],run_seconds=0.0,fork_episode=SOURCE_EP,
               fork_from_run=str(source))
    return cfg,cfg_path,meta,state,SOURCE_EP,cp,source


def main(argv=None):
    a=parse_args(argv)
    cfg,cfg_path,meta,state,ep0,cp,source=run_preflight(a)
    out=a.output.resolve()
    if a.dry_run:
        print('[SAFE YIELD DRY RUN OK] source EP130 checksum:',SOURCE_HASH)
        print('[SAFE YIELD DRY RUN OK] source:',cp,'output:',out)
        print('[SAFE YIELD DRY RUN OK] reward:',config(a).metadata())
        print('[SAFE YIELD DRY RUN OK] 20 AGV, GAT4, 3600s, Stop unlocked, no-Dijkstra')
        return
    if a.max_episodes<=ep0 and not a.init_only:
        print('[SAFE YIELD DONE] already committed',ep0)
        return
    import numpy as np
    import torch
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
    from thesis.mappo.mappo_stability_v7 import audit_rollout_roles
    from v12_clean_audit import verify_v12_ppo_rollout_parity
    from v14_reward_collector import V14RewardEventCollector
    if str(a.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no implicit CPU fallback')
    os.environ['MAPPO_MAX_MINIBATCH']=str(a.max_minibatch)
    os.environ['OMP_NUM_THREADS']=str(a.cpu_threads)
    os.environ['MKL_NUM_THREADS']=str(a.cpu_threads)
    os.environ['OPENBLAS_NUM_THREADS']=str(a.cpu_threads)
    os.environ['MAPPO_DISPATCH_MASK_MODE']='operation_first'
    torch.set_num_threads(int(a.cpu_threads))
    logging.getLogger('alpyne.data').setLevel(logging.ERROR)
    logging.getLogger('alpyne.sim').setLevel(logging.ERROR)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    shaper=SafeYieldStallReward(config(a))
    shaper.install()  # Direct environment reward, NOT PPO trajectory rewrites.
    env=None
    try:
        env,model,spec,arch=build_v14(
            cfg,a.device,a.port,a.sim_seconds,a.max_events,
            gat_layers=a.gat_layers,gat_heads=a.gat_heads,
            dispatch_width=a.dispatch_width,embedding=a.embedding,
            station_id_dim=a.station_id_dim,checkpoint=cp)
        if not a.resume:
            out.mkdir(parents=True,exist_ok=True)
            shutil.copy2(cfg_path,out/'run_config_blueprint.json')
            target=out/f'rolling_slot_{SOURCE_EP%2}.pt'
            shutil.copy2(cp,target)
            if sha256(target)!=SOURCE_HASH:
                raise RuntimeError('Copied EP130 checkpoint failed SHA verification')
            atomic_json(out/'experiment.json',meta)
            (out/'metrics.jsonl').touch(exist_ok=False)
            atomic_json(out/'state.json',state)
            print('[SAFE YIELD FORK] copied intact EP130 checkpoint (actors, critics, optimizers, ValueNorm)')
        else:
            print(f'[SAFE YIELD RESUME] committed EP{ep0}: {cp}')
        if a.init_only:
            print('[SAFE YIELD INIT ONLY] no training performed')
            return
        collector=V14RewardEventCollector(model,env,shaper,stop_mask_mode=STOP_MODE)
        auditor=RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        while state['last_episode'] < a.max_episodes:
            ep=state['last_episode']+1
            seed=int(a.seed)+ep-1
            random.seed(int(a.seed)+ep);np.random.seed(int(a.seed)+ep);torch.manual_seed(int(a.seed)+ep)
            auditor.reset();started=time.monotonic()
            trajectories,data=collector.episode(seed=seed,train=True,max_event_steps=a.max_events)
            if data.get('end_reason')=='max_event_steps':
                raise RuntimeError('Simulation capped by max-events; refusing PPO update')
            if GraphSpec.from_hive(env.agent_hive).fingerprint !=spec.fingerprint:
                raise RuntimeError('AnyLogic graph fingerprint changed')
            mapping=auditor.report()
            if mapping.get('candidate_map_rate',0)<.99:
                raise RuntimeError('Routing graph mapping below 0.99')
            parity=verify_v12_ppo_rollout_parity(model,trajectories,max_samples=64)
            all_roles=audit_rollout_roles(model,trajectories,max_per_role=64)
            if (isinstance(parity,dict) and parity.get('passed') is False):
                raise RuntimeError('PPO parity audit failed')
            if any(isinstance(v,dict) and v.get('passed') is False for v in
                   (all_roles.values() if isinstance(all_roles,dict) else [])):
                raise RuntimeError('All-role PPO audit failed')
            present={t[0]['role'] for t in trajectories if t}
            if 'routing' not in present:
                raise RuntimeError('No Routing transitions; refusing optimizer update')
            updates=model.train(trajectories)
            slot=out/f'rolling_slot_{ep%2}.pt';tmp=Path(str(slot)+'.tmp')
            model.save(str(tmp));os.replace(tmp,slot)
            elapsed=time.monotonic()-started
            row=dict(version=VERSION,episode=ep,environment_seed=seed,environment=data,
                     graph_mapping=mapping,ppo_audit=parity,all_roles_audit=all_roles,
                     policy_updates=updates,roles_present=sorted(present),train_roles='all',
                     max_minibatch=a.max_minibatch,sim_seconds=3600,architectures=arch,
                     stop_mask_mode=STOP_MODE,reward_mode=meta['reward_mode'],
                     reward_config=meta['reward_config'],fork_episode=SOURCE_EP,
                     checkpoint_origin_sha256=SOURCE_HASH,dijkstra_guidance_strength=0.0,
                     predicted_shortest_path_flow=False,traffic_schema='observed_only_v12')
            with (out/'metrics.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,default=str,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            state['last_episode']=ep;state['checkpoint']=str(slot.resolve())
            state['run_seconds']=round(float(state['run_seconds'])+elapsed,2)
            if ep%a.snapshot_every==0:
                snap=out/f'snapshot_ep_{ep:04d}.pt';temp_snap=Path(str(snap)+'.tmp')
                shutil.copy2(slot,temp_snap);os.replace(temp_snap,snap)
                state['snapshots'].append(str(snap.resolve()))
            atomic_json(out/'state.json',state)
            stats=data.get('statistics',{});rew=data.get('reward_shaping',{})
            print(f'[SAFE YIELD EP{ep}] JobDone={stats.get("jobDone",0)} '
                  f'Block={stats.get("block",0)} LiveLock={stats.get("liveLock",0)} '
                  f'YieldBonus={rew.get("yield_proxy_bonuses_applied",0)} '
                  f'StallPenalty={rew.get("stall_penalties_applied",0)} '
                  f'Reward={data.get("delivered_reward",0):.2f} '
                  f'Elapsed={elapsed:.1f}s',flush=True)
    except KeyboardInterrupt:
        print('[SAFE YIELD INTERRUPTED] Last committed EP',state['last_episode'],flush=True)
    finally:
        if env is not None:
            env.close()
    print('[SAFE YIELD DONE] committed EP',state['last_episode'],'output',out,flush=True)

if __name__=='__main__':
    main()
