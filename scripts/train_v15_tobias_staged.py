"""V15: scratch 3600s MAPPO training, Tobias-style actors, staged policy learning.

No change to thesis/mappo/algorithm.py, mappo_stability_v7.py, collector, or Java.
No Dijkstra guidance in the learned Routing GAT, no reward shaping,
no V14 deep GAT or candidate scoring. Dispatch-only uses native Tobias
CollisionFreeRouting with time-window reservations (not GAT guidance).

Phase order: routing8 -> dispatch8 -> joint8 -> joint12 -> joint16 -> joint20
Each phase starts at its own EP0. Every next phase inherits full role weights,
critics, ValueNorm and Adam optimizers from the preceding phase checkpoint.
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

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from v15_staged_collector import PHASES, StagePolicyAdapter, trained_trajectories

VERSION='V15_TOBIAS_ARCH_STAGED_NATIVE_ROUTER_MAPPO_V2'
PHASE_ORDER=tuple(PHASES)
DEFAULT_OUT=ROOT/'mappo_runs'/'V15_TOBIAS_GATV2_LINFE_MAPPO_NATIVE_ROUTING_3600S'
V15_MODEL_MODULE=ROOT/'thesis'/'mappo'/'v15_tobias_arch.py'
V15_BUILD_MODULE=ROOT/'scripts'/'v15_tobias_build.py'
V15_STAGE_MODULE=ROOT/'scripts'/'v15_staged_collector.py'


def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for c in iter(lambda:stream.read(1024*1024),b''):
            h.update(c)
    return h.hexdigest()


def _read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def _atomic_json(path,data):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    os.replace(tmp,p)


def _pick_blueprint(provided):
    if provided is not None: return Path(provided).resolve()
    options=[
        ROOT/'mappo_runs'/'V14_GAT4_MLP_3600S_MB128_WAIT_SWING_REWARD'/'run_config_blueprint.json',
        ROOT/'mappo_runs'/'V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK'/'run_config_blueprint.json',
        ROOT/'mappo_runs'/'V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK'/'run_config.json',
    ]
    for p in options:
        if p.is_file():return p.resolve()
    raise FileNotFoundError('Missing topology/AnyLogic blueprint. Specify --blueprint PATH_TO_V14/run_config_blueprint.json')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=PHASE_ORDER,required=True)
    p.add_argument('--max-episodes',type=int,default=1,help='Max completed EP in THIS phase, beginning from 0')
    p.add_argument('--blueprint',type=Path,default=None)
    p.add_argument('--model',type=Path,default=None,help='Existing original AnyLogic simulator .zip; overrides blueprint model_file')
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUT)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51880)
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--max-events',type=int,default=None)
    p.add_argument('--max-minibatch',type=int,default=128)
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--snapshot-every',type=int,default=10)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--init-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    if a.max_episodes<0 or a.max_minibatch<1 or a.cpu_threads<1 or a.snapshot_every<1 or not 1<=a.port<=65535:
        p.error('Invalid episode/minibatch/threads/snapshot/port')
    if a.resume and a.init_only:p.error('--resume and --init-only are mutually exclusive')
    return a


def preflight(a):
    blueprint=_pick_blueprint(a.blueprint)
    if not blueprint.exists():raise FileNotFoundError(blueprint)
    cfg=_read(blueprint)
    keys=('max_fleetsize','hidden','model_file','seed','gamma','gae_lambda','clip','lr',
          'epochs','mini_batches','ppo_v7_options')
    missing=[x for x in keys if x not in cfg]
    if missing:raise ValueError('Blueprint missing: '+repr(missing))
    if int(cfg['max_fleetsize'])!=20:
        raise ValueError('Must reuse V8/V14 blueprint with max_fleetsize=20')
    from v15_tobias_build import TOBIAS_JOINT_REWARDS
    assert 'reward_part_completed' not in TOBIAS_JOINT_REWARDS
    if a.seed is None:a.seed=int(cfg['seed'])
    if a.max_events is None:a.max_events=int(cfg.get('max_events',100000))
    if a.max_events<1:raise ValueError('max-events must be positive')
    model=Path(a.model or cfg['model_file']).resolve()
    if not model.is_file():
        raise FileNotFoundError('AnyLogic model .zip missing: '+str(model)+'; use --model PATH')
    fingerprints={
        'architecture':_digest(V15_MODEL_MODULE),
        'environment_builder':_digest(V15_BUILD_MODULE),
        'staged_policy':_digest(V15_STAGE_MODULE),
        'ppo_algorithm':_digest(ROOT/'thesis'/'mappo'/'algorithm.py'),
        'ppo_stability':_digest(ROOT/'thesis'/'mappo'/'mappo_stability_v7.py'),
        'event_collector':_digest(ROOT/'thesis'/'mappo'/'collector_stop_ab.py'),
        'matrix_environment':_digest(ROOT/'thesis'/'envs'/'matrix_v121.py'),
        'behaviors':_digest(ROOT/'thesis'/'envs'/'behaviors_v121.py'),
        'alpyne_bridge':_digest(ROOT/'thesis'/'envs'/'alpyne121_bridge.py'),
        'reward_config':hashlib.sha256(json.dumps(TOBIAS_JOINT_REWARDS,sort_keys=True).encode()).hexdigest(),
    }
    p=PHASE_ORDER.index(a.phase)
    out=a.output_root.resolve()/a.phase
    return cfg,blueprint,model,fingerprints,out,p


def _checkpoint(state,out):
    path=Path(state['checkpoint']).resolve()
    if not path.is_file() or path.parent!=Path(out).resolve():
        raise RuntimeError('Missing checkpoint in own phase directory: '+str(path))
    if _digest(path)!=state.get('checkpoint_sha256'):
        raise RuntimeError('Checkpoint hash mismatch; refusing corrupt resume/transfer')
    return path


def _meta(a,cfg,blueprint,model,fingerprints,phaseidx):
    return dict(version=VERSION,phase=a.phase,phase_index=phaseidx,
        fleet_size=PHASES[a.phase][0],active_roles=list(PHASES[a.phase][1]),
        sim_seconds=3600,base_seed=a.seed,max_events=a.max_events,device=a.device,
        max_minibatch=a.max_minibatch,cpu_threads=a.cpu_threads,
        snapshot_every=a.snapshot_every,blueprint_sha256=_digest(blueprint),
        blueprint_file=str(blueprint),anylogic_sha256=_digest(model),
        model_file=str(model),fingerprints=fingerprints,
        architecture='Tobias GATv2 2x4 + LinFE x2 + CTDE MLP x3',
        original_tobias_dispatch='random_when_training_routing',
        dispatch_pretraining_router='Tobias native CollisionFreeRouting (time-window reservation)',
        reward_source='Tobias training_disp_rout.ipynb joint configuration',
        reward_part_completed='not overridden (commented in original notebook)',
        ppo_core='unchanged MAPPO + existing V7 stability optimizer',
        phase_train_episode_index_starts_at=0)


def _verify_lineage(old,current):
    comparable=('version','blueprint_sha256','anylogic_sha256','fingerprints',
                'model_file','max_events','base_seed','device','max_minibatch',
                'cpu_threads','snapshot_every','sim_seconds')
    diff={k:(old.get(k),current.get(k)) for k in comparable if old.get(k)!=current.get(k)}
    if diff:raise RuntimeError('Cross-stage source/config mismatch: '+str(diff))



def _frozen_role_hashes(model, trained_roles):
    """Read-only audit: verify pretraining does not change frozen-role weights."""
    allowed=set(trained_roles)
    output={}
    for name,role in model.roles.items():
        if name in allowed: continue
        h=hashlib.sha256()
        for component in ('actor','critic'):
            for key,value in sorted(getattr(role,component).state_dict().items()):
                h.update(component.encode());h.update(key.encode())
                h.update(value.detach().contiguous().cpu().numpy().tobytes())
        h.update(json.dumps(role.valuenorm.state_dict(),sort_keys=True).encode())
        output[name]=h.hexdigest()
    return output


def run(argv=None):
    a=parse_args(argv)
    cfg,blueprint,simulator,fingerprints,out,index=preflight(a)
    expected=_meta(a,cfg,blueprint,simulator,fingerprints,index)
    if a.dry_run:
        print('[V15 DRY RUN OK] phase=',a.phase,'fleet=',PHASES[a.phase][0],
              'roles=',PHASES[a.phase][1],'sim=3600 seconds',flush=True)
        print('[V15 DRY RUN OK] GAT Dijkstra guidance=OFF; dispatch8 native Tobias CollisionFreeRouting=ON; stop mask=original')
        print('[V15 DRY RUN OK] AnyLogic ZIP=',simulator)
        print('[V15 DRY RUN OK] blueprint=',blueprint)
        print('[V15 DRY RUN OK] no environment launched and no output modified')
        return
    if a.resume:
        actual=_read(out/'experiment.json')
        _verify_lineage(actual,expected)
        if actual['phase']!=a.phase or actual['fleet_size']!=PHASES[a.phase][0]:
            raise RuntimeError('Resume stage mismatch')
        state=_read(out/'state.json')
        ck=_checkpoint(state,out)
        if not (out/'metrics.jsonl').exists():raise FileNotFoundError('Missing metrics.jsonl')
        ids=[int(json.loads(s)['episode']) for s in (out/'metrics.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
        if ids!=list(range(1,int(state['last_episode'])+1)):
            raise RuntimeError('Metrics/committed checkpoint mismatch, inspect before resuming')
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Output not empty: '+str(out)+'; use --resume')
        ck=None
        if index>0:
            prev=out.parent/PHASE_ORDER[index-1]
            previous=_read(prev/'experiment.json')
            _verify_lineage(previous,expected)
            state_previous=_read(prev/'state.json')
            if int(state_previous['last_episode'])<1:
                raise RuntimeError('Previous phase must complete at least one episode')
            ck=_checkpoint(state_previous,prev)
        state=dict(last_episode=0,checkpoint=None,checkpoint_sha256=None,
                   run_seconds=0.0,snapshots=[],source_checkpoint=str(ck) if ck else None)
    import numpy as np
    import torch
    from v15_tobias_build import build_v15
    from thesis.mappo.collector_stop_ab import EventCollector
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.mappo_stability_v7 import audit_rollout_roles
    os.environ['MAPPO_MAX_MINIBATCH']=str(a.max_minibatch)
    os.environ['MAPPO_DISPATCH_MASK_MODE']='operation_first'
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name]=str(a.cpu_threads)
    torch.set_num_threads(a.cpu_threads)
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; use --device cpu only for offline smoke')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    logging.getLogger('alpyne').setLevel(logging.ERROR)
    env=None
    try:
        env,model,graph,arch=build_v15(cfg,model_path=simulator,
            fleet_size=PHASES[a.phase][0],port=a.port,device=a.device,seed=a.seed,
            max_events=a.max_events,sim_seconds=3600,checkpoint=ck,phase=a.phase)
        if not a.resume:
            out.mkdir(parents=True,exist_ok=True)
            shutil.copy2(blueprint,out/'run_config_blueprint.json')
            initial=out/'initial_ep_0000.pt'
            model.save(str(initial))
            state['checkpoint']=str(initial.resolve())
            state['checkpoint_sha256']=_digest(initial)
            expected['architecture_actual']=arch
            expected['source_checkpoint']=str(ck) if ck else None
            expected['source_checkpoint_sha256']=_digest(ck) if ck else None
            _atomic_json(out/'experiment.json',expected)
            (out/'metrics.jsonl').touch(exist_ok=False)
            _atomic_json(out/'state.json',state)
            print('[V15 PHASE INITIALIZED]',a.phase,'EP0','src=',ck,flush=True)
        if a.init_only:
            print('[V15 INIT ONLY] No training conducted');return
        # No guidance, no GNN wrapper, no collector/algorithm source edits.
        adapter=StagePolicyAdapter(model,a.phase,a.seed)
        collector=EventCollector(adapter,env,stop_mask_mode='baseline')
        phase_number=index
        while state['last_episode']<a.max_episodes:
            ep=int(state['last_episode'])+1
            roll_seed=int(a.seed)+phase_number*100000+ep-1
            random.seed(roll_seed);np.random.seed(roll_seed);torch.manual_seed(roll_seed)
            adapter.rng=np.random.default_rng(roll_seed)
            t0=time.monotonic()
            paths,meta=collector.episode(seed=roll_seed,train=True,max_event_steps=a.max_events)
            if meta['end_reason']=='max_event_steps':
                raise RuntimeError('Max-events truncated rollout. No PPO update.')
            if GraphSpec.from_hive(env.agent_hive).fingerprint!=graph.fingerprint:
                raise RuntimeError('Hive graph changed within training')
            selected=trained_trajectories(paths,a.phase)
            observed={segment[0]['role'] for segment in selected if segment}
            if not observed.intersection(PHASES[a.phase][1]):
                raise RuntimeError('No trainable transitions collected')
            parity=audit_rollout_roles(model,selected,max_per_role=48)
            if any(isinstance(x,dict) and x.get('passed') is False for x in parity.values()):
                raise RuntimeError('PPO rollout logprob mismatch; refusing update')
            # Reuse V7 stable PPO, unchanged: only selected role trajectories.
            frozen_before=_frozen_role_hashes(model,PHASES[a.phase][1])
            updates=model.train(selected)
            if _frozen_role_hashes(model,PHASES[a.phase][1])!=frozen_before:
                raise RuntimeError('A frozen role changed during PPO; aborting commit')
            if set(updates)-set(PHASES[a.phase][1]):
                raise RuntimeError('Unexpected frozen role optimizer updated')
            slot=out/f'rolling_slot_{ep%2}.pt'
            tmp=out/f'rolling_slot_{ep%2}.pt.tmp'
            model.save(str(tmp));os.replace(tmp,slot)
            elapsed=time.monotonic()-t0
            row=dict(version=VERSION,phase=a.phase,episode=ep,
                     fleet_size=PHASES[a.phase][0],sim_seconds=3600,
                     environment_seed=roll_seed,environment=meta,
                     ppo_audit=parity,policy_updates=updates,architectures=arch,
                     train_roles=sorted(PHASES[a.phase][1]),
                     non_train_actions=('random_dispatch' if a.phase=='routing8' else
                                       'tobias_native_collision_free_routing' if a.phase=='dispatch8' else 'none'),
                     reward='Tobias joint notebook, no shaping',
                     gat_dijkstra_guidance=False,
                     native_conventional_routing=(a.phase=='dispatch8'),
                     elapsed_real_seconds=elapsed)
            with (out/'metrics.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
                f.flush();os.fsync(f.fileno())
            state['last_episode']=ep
            state['checkpoint']=str(slot.resolve())
            state['checkpoint_sha256']=_digest(slot)
            state['run_seconds']=round(float(state['run_seconds'])+elapsed,2)
            if ep%a.snapshot_every==0:
                snapshot=out/f'snapshot_ep_{ep:04d}.pt'
                partial=out/f'snapshot_ep_{ep:04d}.pt.tmp'
                shutil.copy2(slot,partial);os.replace(partial,snapshot)
                state['snapshots'].append(str(snapshot.resolve()))
            _atomic_json(out/'state.json',state)
            stats=meta.get('statistics',{})
            print(f'[V15 {a.phase.upper()} EP{ep}] '
                  f'AGV={PHASES[a.phase][0]} JobDone={stats.get("jobDone",0)} '
                  f'Block={stats.get("block",0)} LiveLock={stats.get("liveLock",0)} '
                  f'Reward={meta.get("delivered_reward",0.):.2f} '
                  f'Updated={sorted(updates)} Elapsed={elapsed:.1f}s',flush=True)
    except KeyboardInterrupt:
        print('[V15 INTERRUPTED] Last committed EP',state['last_episode'],flush=True)
    finally:
        if env is not None:env.close()
    print('[V15 PHASE DONE]',a.phase,'committed EP',state['last_episode'],'output',out,flush=True)


if __name__=='__main__':run()
