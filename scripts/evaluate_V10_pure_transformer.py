"""Evaluate V10-D pure Transformer checkpoint (no PPO updates).

Never writes to / loads any V8 checkpoint other than the frozen source copy.
Produces JSONL comparable to V8 heldout metrics on same seeds.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT))
from v10_pure_common import read_json, sha256, setup_model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'mappo_runs'/'V10_D_PURE_both_FROM_V8')
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--seeds',type=int,nargs='+',default=[910001,910002,910003,910004,910005])
    p.add_argument('--sim-seconds',type=int,default=None,help='Default: training horizon from experiment.json')
    p.add_argument('--device',default='cpu')
    p.add_argument('--port',type=int,default=51881)
    p.add_argument('--output',type=Path,default=None)
    args=p.parse_args()
    if not args.seeds or len(args.seeds)!=len(set(args.seeds)):
        p.error('Provide unique seeds')
    run=args.run.resolve()
    meta=read_json(run/'experiment.json')
    cfg=read_json(run/'v8_source_run_config.json')
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
    from thesis.mappo.routing_guidance_v121 import GuidedEventCollector
    from thesis.mappo.mappo_stability_v7 import audit_rollout_roles
    import torch
    os.environ['MAPPO_DISPATCH_MASK_MODE']='operation_first'
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available, try --device cpu')
    source=run/'v8_source_checkpoint.pt'
    if sha256(source)!=meta['source_sha256']:
        raise RuntimeError('V8 frozen checkpoint changed')
    if args.checkpoint is None:
        state=read_json(run/'state.json')
        cp=Path(state['checkpoint']).resolve()
    else:
        cp=args.checkpoint.resolve()
    if not cp.is_file() or cp.parent != run:
        raise FileNotFoundError('Checkpoint must exist inside selected run')
    sim_seconds=meta['sim_seconds'] if args.sim_seconds is None else int(args.sim_seconds)
    if args.output is None:
        output=run/('heldout_'+cp.stem+'_'+str(len(args.seeds))+'seeds.jsonl')
    else:
        output=args.output.resolve()
    if output.exists():
        raise FileExistsError('Evaluation output already exists: '+str(output))
    if sim_seconds<1:
        raise ValueError('Invalid sim-seconds')
    env=None
    results=[]
    try:
        env,model,spec,ma=setup_model(
            cfg,source,meta['variant'],args.device,args.port,sim_seconds,
            int(meta['max_events']),int(meta['hidden']),int(meta['heads']),
            int(meta['layers']),v10_checkpoint=cp)
        auditor=RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        collector=GuidedEventCollector(model,env,stop_mask_mode=cfg['stop_mask_mode'],routing_guide_strength=1.5)
        for seed in args.seeds:
            auditor.reset()
            trajectories,metadata=collector.episode(seed=seed,train=False,max_event_steps=int(meta['max_events']))
            if GraphSpec.from_hive(env.agent_hive).fingerprint != spec.fingerprint:
                raise RuntimeError('AnyLogic graph changed during evaluation')
            audit=auditor.report()
            if float(audit['candidate_map_rate'])<.99:
                raise RuntimeError('Routing candidate map below 99%')
            row=dict(version='V10_D_PURE_HELDOUT',variant=meta['variant'],seed=seed,
                     checkpoint=str(cp),sim_seconds=sim_seconds,
                     environment=metadata,graph_mapping=audit)
            results.append(row)
            stats=metadata.get('statistics',{})
            print(f'[V10-D PURE EVAL {meta["variant"]} seed={seed}] JobDone={stats.get("jobDone",0)} '
                  f'WrongTarget={stats.get("wrongTarget",0)} Block={stats.get("block",0)} '
                  f'LiveLock={stats.get("liveLock",0)}',flush=True)
    finally:
        if env is not None:
            env.close()
    # Write only after completing ALL seeds; failed eval is never presented as final.
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r,default=str)+'\n')
    print('[V10-D PURE EVAL DONE]',output,flush=True)

if __name__=='__main__':
    main()
