"""Deterministic held-out evaluation for V15: 10 seeds, 3600s, no PPO updates.

Evaluates any completed stage, using its actual trained + frozen controllers.
For joint20 this evaluates all learned policies. Never overwrites checkpoints.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from v15_staged_collector import StagePolicyAdapter,PHASES
from train_v15_tobias_staged import (_read,_digest,_checkpoint,PHASE_ORDER,
                                     preflight,parse_args)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=PHASE_ORDER,default='joint20')
    p.add_argument('--blueprint',type=Path,default=None)
    p.add_argument('--model',type=Path,default=None)
    p.add_argument('--output-root',type=Path,default=ROOT/'mappo_runs'/'V15_TOBIAS_ORIGINAL_DISPATCH_MASK_3600S')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51881)
    p.add_argument('--seeds',type=int,default=10)
    p.add_argument('--first-seed',type=int,default=900001)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(argv)
    if args.seeds<1 or args.seeds>100 or not 1<=args.port<=65535:
        p.error('Invalid seeds/port')
    stage_dir=args.output_root.resolve()/args.phase
    meta=_read(stage_dir/'experiment.json')
    state=_read(stage_dir/'state.json')
    checkpoint=_checkpoint(state,stage_dir)
    from v15_tobias_build import build_v15, TOBIAS_JOINT_REWARDS, TOBIAS_DISPATCH_MASK_MODE
    blueprint=Path(args.blueprint or meta['blueprint_file']).resolve()
    cfg=_read(blueprint)
    if _digest(blueprint)!=meta['blueprint_sha256']:
        raise RuntimeError('Evaluation blueprint changed')
    simulator=Path(args.model or meta['model_file']).resolve()
    if _digest(simulator)!=meta['anylogic_sha256']:
        raise RuntimeError('AnyLogic model changed')
    if args.device!=meta['device']:
        raise RuntimeError('Device changed; use same --device as training or change design explicitly')
    for name,path in [('architecture',ROOT/'thesis/mappo/v15_tobias_arch.py'),
                      ('environment_builder',ROOT/'scripts/v15_tobias_build.py'),
                      ('staged_policy',ROOT/'scripts/v15_staged_collector.py'),
                      ('ppo_algorithm',ROOT/'thesis/mappo/algorithm.py'),
                      ('ppo_stability',ROOT/'thesis/mappo/mappo_stability_v7.py'),
                      ('event_collector',ROOT/'thesis/mappo/collector_stop_ab.py'),
                      ('matrix_environment',ROOT/'thesis/envs/matrix_v121.py'),
                      ('behaviors',ROOT/'thesis/envs/behaviors_v121.py'),
                      ('alpyne_bridge',ROOT/'thesis/envs/alpyne121_bridge.py')]:
        if _digest(path)!=meta['fingerprints'][name]:
            raise RuntimeError('Changed code since training: '+name)
    result=stage_dir/f'heldout_{args.phase}_{args.seeds}seeds_3600s.jsonl'
    seeds=list(range(args.first_seed,args.first_seed+args.seeds))
    rows=[]
    if result.exists():
        if not args.resume:raise FileExistsError('Evaluation exists; use --resume: '+str(result))
        rows=[json.loads(v) for v in result.read_text(encoding='utf8').splitlines() if v.strip()]
        if any(v.get('checkpoint_sha256')!=state['checkpoint_sha256'] for v in rows):
            raise RuntimeError('Checkpoint changed since earlier evaluation; use a new result file')
        if len({int(v['seed']) for v in rows})!=len(rows):
            raise RuntimeError('Duplicate heldout seed in existing result')
        if any(int(v['seed']) not in seeds for v in rows):
            raise RuntimeError('Existing evaluation contains seed outside requested range')
    todo=[n for n in seeds if n not in {int(row['seed']) for row in rows}]
    if args.dry_run:
        print('[V15 EVAL DRY RUN OK] phase',args.phase,'EP',state['last_episode'],
              'checkpoint=',checkpoint,'pending seeds=',todo)
        print('[V15 EVAL DRY RUN OK] no PPO updates, no environment started')
        return
    import random
    import numpy as np
    import torch
    from thesis.mappo.collector_stop_ab import EventCollector
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available')
    os.environ['MAPPO_DISPATCH_MASK_MODE']=TOBIAS_DISPATCH_MASK_MODE
    os.environ['MAPPO_MAX_MINIBATCH']=str(meta['max_minibatch'])
    torch.set_num_threads(int(meta['cpu_threads']))
    random.seed(meta['base_seed']);np.random.seed(meta['base_seed']);torch.manual_seed(meta['base_seed'])
    env=None
    try:
        env,model,graph,arch=build_v15(cfg,model_path=simulator,
            fleet_size=PHASES[args.phase][0],port=args.port,device=args.device,
            seed=meta['base_seed'],max_events=meta['max_events'],
            sim_seconds=3600,checkpoint=checkpoint,phase=args.phase)
        adapter=StagePolicyAdapter(model,args.phase,meta['base_seed'])
        collector=EventCollector(adapter,env,stop_mask_mode='baseline')
        for seed in todo:
            adapter.rng=np.random.default_rng(seed)
            random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
            with torch.no_grad():
                _,environment=collector.episode(seed=seed,train=False,max_event_steps=meta['max_events'])
            if environment['end_reason']=='max_event_steps':
                raise RuntimeError('Truncated evaluation, refusing to save as complete')
            if GraphSpec.from_hive(env.agent_hive).fingerprint!=graph.fingerprint:
                raise RuntimeError('Graph fingerprint changed')
            row=dict(phase=args.phase,seed=seed,checkpoint_sha256=state['checkpoint_sha256'],
                     episode=int(state['last_episode']),fleet=PHASES[args.phase][0],
                     sim_seconds=3600,greedy=True,ppo_updates=0,
                     dispatch_mask_mode=TOBIAS_DISPATCH_MASK_MODE,environment=environment)
            with result.open('a',encoding='utf8') as f:
                f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
                f.flush();os.fsync(f.fileno())
            stat=environment.get('statistics',{})
            print(f'[V15 EVAL {args.phase} seed={seed}] JobDone={stat.get("jobDone",0)} '
                  f'Block={stat.get("block",0)} LiveLock={stat.get("liveLock",0)}',flush=True)
    finally:
        if env is not None:env.close()
    print('[V15 EVAL DONE]',result,flush=True)

if __name__=='__main__':main()
