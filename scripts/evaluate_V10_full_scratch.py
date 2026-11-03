"""Evaluate a truly from-scratch V10-D model; no V8 checkpoint is read."""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from v10_pure_common import read_json,sha256
from v10_full_scratch_common import setup_from_scratch


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'mappo_runs'/'V10_D_FULL_SCRATCH_2LAYERS')
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--seeds',type=int,nargs='+',default=[900001,900002,900003,900004,900005])
    p.add_argument('--sim-seconds',type=int,default=None)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51881)
    p.add_argument('--output',type=Path,default=None)
    args=p.parse_args(argv)
    if not args.seeds or len(set(args.seeds))!=len(args.seeds):p.error('Seeds must be unique')
    run=args.run.resolve()
    meta=read_json(run/'experiment.json')
    cfg_path=run/'run_config_blueprint.json'
    if sha256(cfg_path)!=meta['run_config_sha256']:
        raise RuntimeError('Scratch run blueprint config changed')
    cfg=read_json(cfg_path)
    cp=(args.checkpoint or Path(read_json(run/'state.json')['checkpoint'])).resolve()
    if not cp.is_file() or cp.parent!=run:
        raise RuntimeError('Checkpoint must exist within the scratch run directory')
    sim_seconds=int(args.sim_seconds or meta['sim_seconds'])
    if sim_seconds<1:raise ValueError('Invalid sim-seconds')
    output=(args.output or run/f'heldout_{cp.stem}_{len(args.seeds)}seeds.jsonl').resolve()
    if output.exists():
        raise FileExistsError('Evaluation would overwrite existing output: '+str(output))
    import torch
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
    from thesis.mappo.routing_guidance_v121 import GuidedEventCollector
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu')
    os.environ['MAPPO_DISPATCH_MASK_MODE']='operation_first'
    env=None;results=[]
    try:
        # Restore all roles from the scratch run checkpoint; DO NOT load V8 .pt.
        env,model,spec=setup_from_scratch(cfg,args.device,args.port,sim_seconds,
            int(meta['max_events']),int(meta['hidden']),int(meta['heads']),int(meta['layers']),
            checkpoint=cp)
        auditor=RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        collector=GuidedEventCollector(model,env,stop_mask_mode=cfg['stop_mask_mode'],routing_guide_strength=1.5)
        for seed in args.seeds:
            auditor.reset()
            _,environment=collector.episode(seed=seed,train=False,max_event_steps=int(meta['max_events']))
            if GraphSpec.from_hive(env.agent_hive).fingerprint!=spec.fingerprint:
                raise RuntimeError('Live graph fingerprint changed')
            report=auditor.report()
            if float(report['candidate_map_rate'])<.99:
                raise RuntimeError('Routing graph mapping <99%')
            row=dict(version='V10_D_FULL_SCRATCH_HELDOUT',seed=seed,checkpoint=str(cp),
                     sim_seconds=sim_seconds,environment=environment,graph_mapping=report)
            results.append(row)
            stat=environment.get('statistics',{})
            print(f'[SCRATCH EVAL seed={seed}] JobDone={stat.get("jobDone",0)} '
                  f'WrongTarget={stat.get("wrongTarget",0)} Block={stat.get("block",0)} '
                  f'LiveLock={stat.get("liveLock",0)}',flush=True)
    finally:
        if env is not None:env.close()
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        for row in results:stream.write(json.dumps(row,default=str)+'\n')
    print('[FULL SCRATCH EVAL DONE]',output,flush=True)

if __name__=='__main__':
    main()
