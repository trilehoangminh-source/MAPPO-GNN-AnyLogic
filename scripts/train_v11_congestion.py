"""Fine-tune V8 from a verified source episode with traffic-aware Routing Edge-GAT until 08:30 VN.

Default: only Routing is updated, Dispatch MLP is preserved and frozen.
--train-roles all continues adapting both original V8 Dispatch MLP roles too.
Original V8 and V10 folders are NEVER modified.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timedelta
from datetime import timezone
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT))
from v11_common import (read_json,atomic_json,sha256,locate_source_episode,
                         validate_v8_cfg,build_v11)
VERSION='V11_CONGESTION_EDGEGAT'
# Vietnam is UTC+7 without DST. Fixed offset avoids missing IANA tzdata on Windows.
TZ=timezone(timedelta(hours=7),name='Asia/Ho_Chi_Minh')


def cutoff_datetime(override):
    if override:
        result=datetime.fromisoformat(override)
        if result.tzinfo is None:
            result=result.replace(tzinfo=TZ)
        else:
            result=result.astimezone(TZ)
        return result
    now=datetime.now(TZ)
    end=now.replace(hour=8,minute=30,second=0,microsecond=0)
    if end <= now:
        end += timedelta(days=1)
    return end


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-run',type=Path,default=ROOT/'mappo_runs/V8_RESIDUAL_EDGEGAT_JOBDONE2_FROM205')
    p.add_argument('--source-checkpoint',type=Path,default=None)
    p.add_argument('--source-episode',type=int,default=304,
                   help='V8 source snapshot episode; default 304 (next training EP305)')
    p.add_argument('--output',type=Path,default=None,
                   help='Default: mappo_runs/V11_CONGESTION_FROM_V8_EP<source-episode>')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51880)
    p.add_argument('--sim-seconds',type=int,default=1800)
    p.add_argument('--max-events',type=int,default=None)
    p.add_argument('--train-roles',choices=('routing','all'),default='routing')
    p.add_argument('--horizon',type=int,default=4)
    p.add_argument('--decay',type=float,default=.75)
    p.add_argument('--max-episodes',type=int,default=10000,
                   help='Safety cap, absolute episode number')
    p.add_argument('--end-at',default=None,
                   help='ISO datetime incl timezone, default next 08:30 Asia/Ho_Chi_Minh')
    p.add_argument('--max-minibatch',type=int,default=32,help='GPU memory safety; changes PPO minibatch count')
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--snapshot-every',type=int,default=10)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--allow-minibatch-change',action='store_true',
                   help='Explicitly allow changing max-minibatch on resume, with immutable provenance backup and history')
    p.add_argument('--allow-role-change',action='store_true',
                   help='Explicitly allow routing -> all on resume, keeping existing checkpoint and logging role schedule')
    p.add_argument('--dry-run',action='store_true')
    x=p.parse_args()
    if x.sim_seconds<1 or x.port<1 or x.port>65535 or x.source_episode<1 or x.max_episodes <= x.source_episode:
        p.error('Invalid sim-seconds, port, or max-episodes')
    if x.max_events is not None and x.max_events<1:
        p.error('Invalid max-events')
    if not 1<=x.cpu_threads<=32 or x.snapshot_every<1 or x.max_minibatch<1:
        p.error('Invalid threads or snapshot-every')
    if x.allow_minibatch_change and not x.resume:
        p.error('--allow-minibatch-change requires --resume')
    if x.allow_role_change and not x.resume:
        p.error('--allow-role-change requires --resume')
    if x.output is None:
        x.output=ROOT/'mappo_runs'/f'V11_CONGESTION_FROM_V8_EP{x.source_episode}'
    return x



def check_role_resume(old, expected, allow_change=False):
    """Never change a run's trained roles without an explicit one-way opt-in."""
    previous = old.get('train_roles')
    new = expected.get('train_roles')
    if previous not in ('routing', 'all') or new not in ('routing', 'all'):
        raise RuntimeError('Invalid train_roles in V11 experiment.json')
    if previous == new:
        return False
    if (previous, new) != ('routing', 'all'):
        raise RuntimeError('Only routing -> all is allowed; refuse to re-freeze a trained role')
    if not allow_change:
        raise RuntimeError(
            'V11 run was started with train_roles=routing. '
            'To train Routing + both Dispatch roles from the next episode, '
            'add --allow-role-change to --resume.')
    return True


def check_minibatch_resume(old, expected, allow_change=False, allow_role_change=False):
    """Check all immutable run settings; explicitly allow ONLY max_minibatch change.

    Legacy V11 manifests lack minibatch_schedule; that is supported.
    This routine is side-effect free so it can be unit tested without AnyLogic.
    """
    check_role_resume(old, expected, allow_role_change)
    previous=old.get('max_minibatch')
    if not isinstance(previous,int) or previous < 1:
        raise RuntimeError('Invalid recorded max_minibatch in V11 experiment.json')
    mutable=('max_minibatch','minibatch_schedule','train_roles','train_roles_schedule')
    old_config={k:v for k,v in old.items() if k not in mutable}
    new_config={k:v for k,v in expected.items() if k not in mutable}
    if old_config != new_config:
        raise RuntimeError('V11 resume parameters/source changed (other than minibatch); refusing to mix runs')
    changed=previous!=expected['max_minibatch']
    if changed and not allow_change:
        raise RuntimeError(
            f'V11 run was started with max-minibatch={previous}, requested={expected["max_minibatch"]}. '
            'To change it deliberately, add --allow-minibatch-change to --resume.')
    return changed


def record_minibatch_change(meta_file, old, source_episode, last_episode, new_minibatch):
    """Persist a truthful per-episode minibatch schedule, preserving old manifest.

    Only call after the checkpoint, state and complete metrics have been validated.
    """
    previous=int(old['max_minibatch'])
    if previous == new_minibatch:
        return
    schedule=old.get('minibatch_schedule')
    if schedule is None:
        schedule=[dict(from_episode=int(source_episode)+1, max_minibatch=previous)]
    else:
        schedule=list(schedule)
        if not schedule or int(schedule[-1]['max_minibatch']) != previous:
            raise RuntimeError('V11 minibatch history inconsistent with recorded config')
    next_episode=int(last_episode)+1
    if next_episode <= int(source_episode):
        raise RuntimeError('Invalid next episode for minibatch change')
    if int(schedule[-1]['from_episode']) > next_episode:
        raise RuntimeError('V11 minibatch history has a future entry')
    # Preserve original schedule even if zero completed episodes used prior size.
    # In that case replace the initial entry rather than falsely recording training.
    if next_episode==int(source_episode)+1:
        schedule=[dict(from_episode=next_episode,max_minibatch=int(new_minibatch))]
    else:
        schedule.append(dict(from_episode=next_episode,max_minibatch=int(new_minibatch)))
    backup=meta_file.parent/f'experiment_before_minibatch_{previous}_to_{new_minibatch}_EP{next_episode:04d}.json'
    if backup.exists():
        raise FileExistsError('Refusing to overwrite previous minibatch migration backup: '+str(backup))
    atomic_json(backup,old)
    updated=dict(old)
    updated['max_minibatch']=int(new_minibatch)
    updated['minibatch_schedule']=schedule
    atomic_json(meta_file,updated)
    print(f'[V11 MINIBATCH CHANGE] {previous} -> {new_minibatch} starting EP{next_episode}; '
          f'original experiment backed up to {backup.name}',flush=True)


def record_role_change(meta_file, old, source_episode, last_episode, new_roles):
    """Record when Dispatch MLP Actor/Critic start receiving PPO updates."""
    previous=old['train_roles']
    if previous == new_roles:
        return
    if (previous, new_roles) != ('routing', 'all'):
        raise RuntimeError('Only routing -> all migration is supported')
    next_episode=int(last_episode)+1
    if next_episode<=int(source_episode):
        raise RuntimeError('Invalid next episode for V11 role change')
    schedule=old.get('train_roles_schedule')
    if schedule is None:
        schedule=[dict(from_episode=int(source_episode)+1,train_roles=previous)]
    else:
        schedule=list(schedule)
        if not schedule or schedule[-1]['train_roles'] != previous:
            raise RuntimeError('V11 role schedule inconsistent with manifest')
    if next_episode == int(source_episode)+1:
        schedule=[dict(from_episode=next_episode,train_roles=new_roles)]
    else:
        if int(schedule[-1]['from_episode']) >= next_episode:
            raise RuntimeError('V11 role schedule overlaps the upcoming episode')
        schedule.append(dict(from_episode=next_episode,train_roles=new_roles))
    backup=meta_file.parent/f'experiment_before_roles_{previous}_to_{new_roles}_EP{next_episode:04d}.json'
    if backup.exists():
        raise FileExistsError('Refusing to overwrite role migration backup: '+str(backup))
    atomic_json(backup,old)
    updated=dict(old)
    updated['train_roles']=new_roles
    updated['train_roles_schedule']=schedule
    atomic_json(meta_file,updated)
    print(f'[V11 ROLE CHANGE] {previous} -> {new_roles} starting EP{next_episode}; '
          f'manifest backed up as {backup.name}',flush=True)


def main():
    x=arguments()
    src=x.source_run.resolve();out=x.output.resolve()
    if out==src or out in src.parents or src in out.parents:
        raise ValueError('V11 output must be separate from V8 source')
    cfg=read_json(src/'run_config.json');validate_v8_cfg(cfg)
    cp=locate_source_episode(src,x.source_episode,x.source_checkpoint)
    source_hash=sha256(cp)
    end=cutoff_datetime(x.end_at)
    if end <= datetime.now(TZ) and not x.dry_run:
        raise ValueError('Deadline is in the past; use a future --end-at')
    max_events=x.max_events if x.max_events is not None else int(cfg['max_events'])
    manifest=dict(version=VERSION,source_episode=x.source_episode,source_run=str(src),source_checkpoint=str(cp),
                  source_sha256=source_hash,source_run_config_sha256=sha256(src/'run_config.json'),
                  graph_fingerprint=cfg['graph_fingerprint'],
                  anylogic_model_sha256=sha256(Path(cfg['model_file']).resolve()),
                  train_roles=x.train_roles,fleetsize=20,dispatch='V8 pooled MLP',
                  routing='V8 Residual Edge-GAT plus six observation-derived traffic channels',
                  horizon=x.horizon,decay=x.decay,sim_seconds=x.sim_seconds,
                  max_events=max_events,cpu_threads=x.cpu_threads,max_minibatch=x.max_minibatch,
                  device=x.device,snapshot_every=x.snapshot_every,seed=int(cfg['seed']))
    if x.dry_run:
        print(f'[V11 DRY RUN] source EP{x.source_episode}:',cp)
        print('[V11 DRY RUN] source SHA256:',source_hash)
        print('[V11 DRY RUN] output:',out)
        print('[V11 DRY RUN] train roles:',x.train_roles,'; dispatch remains MLP')
        print('[V11 DRY RUN] cutoff:',end.isoformat(),'local VN')
        print('[V11 DRY RUN] AnyLogic:',cfg['model_file'])
        return
    import numpy as np
    import torch
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit,verify_ppo_rollout_parity
    from thesis.mappo.mappo_stability_v7 import audit_rollout_roles
    from thesis.mappo.routing_guidance_v121 import GuidedEventCollector
    if x.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU fallback')
    os.environ['MAPPO_MAX_MINIBATCH']=str(x.max_minibatch)
    torch.set_num_threads(x.cpu_threads)
    os.environ['OMP_NUM_THREADS']=str(x.cpu_threads)
    os.environ['MKL_NUM_THREADS']=str(x.cpu_threads)
    os.environ['OPENBLAS_NUM_THREADS']=str(x.cpu_threads)
    state_file=out/'state.json';meta_file=out/'experiment.json';metric_file=out/'metrics.jsonl'
    if x.resume:
        old=read_json(meta_file)
        minibatch_changed=check_minibatch_resume(old,manifest,x.allow_minibatch_change,x.allow_role_change)
        role_changed=check_role_resume(old,manifest,x.allow_role_change)
        state=read_json(state_file)
        ep=int(state['last_episode'])
        checkpoint=Path(state['checkpoint']).resolve()
        if ep<x.source_episode or (ep>x.source_episode and (checkpoint.parent!=out or
            checkpoint.name not in ('rolling_slot_0.pt','rolling_slot_1.pt'))):
            raise RuntimeError('Unsafe V11 resume checkpoint')
        if ep>x.source_episode and not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        rows=[json.loads(s) for s in metric_file.read_text(encoding='utf-8').splitlines() if s.strip()] if metric_file.exists() else []
        good=[r for r in rows if int(r['episode'])<=ep]
        if [int(r['episode']) for r in good] != list(range(x.source_episode+1,ep+1)):
            raise RuntimeError('V11 metrics missing/duplicate completed episodes')
        if len(rows)!=len(good):
            metric_file.write_text(''.join(json.dumps(r)+'\n' for r in good),encoding='utf-8')
        if minibatch_changed:
            record_minibatch_change(meta_file,old,x.source_episode,ep,x.max_minibatch)
        if role_changed:
            record_role_change(meta_file,read_json(meta_file),x.source_episode,ep,x.train_roles)
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('V11 output nonempty; do not overwrite '+str(out))
        out.mkdir(parents=True,exist_ok=True)
        atomic_json(meta_file,manifest)
        state=dict(last_episode=x.source_episode,checkpoint=str(cp),snapshots=[],run_seconds=0.0)
        atomic_json(state_file,state)
    current_cp=None if int(state['last_episode'])==x.source_episode else Path(state['checkpoint']).resolve()
    env=None
    try:
        env,model,spec,installation=build_v11(cfg,cp,x.device,x.port,x.sim_seconds,
                       max_events,x.horizon,x.decay,v11_checkpoint=current_cp,
                       verify_parity=current_cp is None)
        collector=GuidedEventCollector(model,env,stop_mask_mode=cfg['stop_mask_mode'],routing_guide_strength=1.5)
        auditor=RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        print('[V11 READY] EP',state['last_episode'],'cutoff:',end.isoformat(),
              'train_roles:',x.train_roles,'new traffic:',installation['channels'],flush=True)
        last_sec=0.0
        while int(state['last_episode']) < x.max_episodes:
            remain=(end-datetime.now(TZ)).total_seconds()
            if remain <= max(40.,last_sec*1.15):
                print('[V11 STOP] deadline reserve: ',round(remain,1),'seconds',flush=True)
                break
            ep=int(state['last_episode'])+1
            seed=int(cfg['seed'])+ep-1
            random.seed(int(cfg['seed'])+ep)
            np.random.seed(int(cfg['seed'])+ep)
            torch.manual_seed(int(cfg['seed'])+ep)
            auditor.reset()
            started=time.monotonic()
            trajectories,environment=collector.episode(seed=seed,train=True,max_event_steps=max_events)
            if GraphSpec.from_hive(env.agent_hive).fingerprint!=spec.fingerprint:
                raise RuntimeError('AnyLogic graph changed during rollout')
            audit=auditor.report()
            if audit['candidate_map_rate']<.99:
                raise RuntimeError('Routing candidate map < 99%; no PPO update')
            parity=verify_ppo_rollout_parity(model,trajectories,max_samples=64)
            role_audit=audit_rollout_roles(model,trajectories,max_per_role=64)
            if x.train_roles=='routing':
                selected=[traj for traj in trajectories if traj and traj[0]['role']=='routing']
                if not selected:
                    raise RuntimeError('No routing trajectory!')
            else:
                selected=trajectories
            updates=model.train(selected)
            slot=out/f'rolling_slot_{ep%2}.pt'
            tmp=Path(str(slot)+'.tmp')
            model.save(str(tmp));os.replace(tmp,slot)
            elapsed=time.monotonic()-started
            row=dict(version=VERSION,episode=ep,environment_seed=seed,
                     environment=environment,graph_mapping=audit,
                     ppo_audit=parity,all_roles_audit=role_audit,
                     policy_updates=updates,train_roles=x.train_roles,
                     max_minibatch=x.max_minibatch)
            with metric_file.open('a',encoding='utf-8') as f:
                f.write(json.dumps(row,default=str)+'\n');f.flush();os.fsync(f.fileno())
            state['last_episode']=ep;state['checkpoint']=str(slot.resolve())
            state['run_seconds']=round(state.get('run_seconds',0)+elapsed,2)
            if (ep-x.source_episode)%x.snapshot_every==0:
                snap=out/f'snapshot_ep_{ep:04d}.pt'
                shutil.copy2(slot,str(snap)+'.tmp')
                os.replace(str(snap)+'.tmp',snap)
                state['snapshots'].append(str(snap.resolve()))
            atomic_json(state_file,state)
            stat=environment.get('statistics',{})
            last_sec=elapsed
            print(f'[V11 EP{ep}] JobDone={stat.get("jobDone",0)} '
                  f'WrongTarget={stat.get("wrongTarget",0)} '
                  f'Block={stat.get("block",0)} LiveLock={stat.get("liveLock",0)} '
                  f'Reward={environment.get("delivered_reward",0):.2f} '
                  f'Elapsed={elapsed:.1f}s',flush=True)
    except KeyboardInterrupt:
        print('[V11 INTERRUPTED] last committed',state['last_episode'],flush=True)
    finally:
        if env is not None:env.close()
    if int(state['last_episode'])>x.source_episode:
        latest=out/'latest_v11.pt'
        shutil.copy2(Path(state['checkpoint']),str(latest)+'.tmp')
        os.replace(str(latest)+'.tmp',latest)
    print('[V11 DONE] last committed EP',state['last_episode'],'output',out,flush=True)


if __name__=='__main__':main()
