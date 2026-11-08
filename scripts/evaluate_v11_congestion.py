"""Held-out 5-seed V11 evaluation: NO PPO updates, Same V8 task and horizon."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT))
from v11_common import read_json,sha256,build_v11


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=ROOT/'mappo_runs/V11_CONGESTION_FROM_V8_EP304')
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51881)
    p.add_argument('--sim-seconds',type=int,default=1800)
    p.add_argument('--seeds',type=int,nargs='+',default=[900001,900002,900003,900004,900005])
    p.add_argument('--output',type=Path,default=None)
    x=p.parse_args()
    if x.sim_seconds<1 or len(set(x.seeds))!=len(x.seeds) or not x.seeds:
        p.error('Invalid sim-seconds or duplicate/empty seeds')
    import torch
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
    from thesis.mappo.routing_guidance_v121 import GuidedEventCollector
    if x.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available, choose --device cpu')
    run=x.run.resolve()
    meta=read_json(run/'experiment.json')
    cfg=read_json(Path(meta['source_run'])/'run_config.json')
    source=Path(meta['source_checkpoint']).resolve()
    if sha256(source)!=meta['source_sha256']:
        raise RuntimeError('V8 source checkpoint changed')
    cp=Path(read_json(run/'state.json')['checkpoint']).resolve() if x.checkpoint is None else x.checkpoint.resolve()
    if not cp.is_file() or cp.parent!=run:
        raise FileNotFoundError('V11 evaluation checkpoint must exist in run directory')
    if x.output is None:
        outfile=run/f'heldout_{cp.stem}_{len(x.seeds)}seeds.jsonl'
    else:
        outfile=x.output.resolve()
    if outfile.exists():
        raise FileExistsError('Will not overwrite evaluation result: '+str(outfile))
    env=None;results=[]
    try:
        env,model,spec,_=build_v11(cfg,source,x.device,x.port,x.sim_seconds,
                  int(meta['max_events']),int(meta['horizon']),float(meta['decay']),
                  v11_checkpoint=cp)
        auditor=RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        collector=GuidedEventCollector(model,env,stop_mask_mode=cfg['stop_mask_mode'],routing_guide_strength=1.5)
        for seed in x.seeds:
            auditor.reset()
            traj,md=collector.episode(seed=seed,train=False,max_event_steps=int(meta['max_events']))
            if GraphSpec.from_hive(env.agent_hive).fingerprint != spec.fingerprint:
                raise RuntimeError('Live AnyLogic graph changed')
            report=auditor.report()
            if report['candidate_map_rate']<.99:
                raise RuntimeError('Graph mapping below 99%')
            results.append(dict(version='V11_HELDOUT',seed=seed,checkpoint=str(cp),
                         sim_seconds=x.sim_seconds,environment=md,graph_mapping=report))
            stats=md.get('statistics',{})
            print(f'[V11 EVAL seed={seed}] JobDone={stats.get("jobDone",0)} '
                  f'WrongTarget={stats.get("wrongTarget",0)} '
                  f'Block={stats.get("block",0)} LiveLock={stats.get("liveLock",0)}',flush=True)
    finally:
        if env is not None:env.close()
    outfile.parent.mkdir(parents=True,exist_ok=True)
    with outfile.open('x',encoding='utf-8') as f:
        for row in results:f.write(json.dumps(row,default=str)+'\n')
    print('[V11 EVAL DONE]',outfile,flush=True)
if __name__=='__main__':main()
