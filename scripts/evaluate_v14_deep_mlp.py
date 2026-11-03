"""V14 held-out evaluation, 3600 simulated seconds, no PPO updates.

Outputs contain environment KPIs and Dispatch multi-choice diagnostics.
Use a separate AnyLogic port from the training server.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts')]
from train_v14_deep_mlp_scratch import VERSION,DEFAULT_OUT,read_json,sha256
from v14_deep_mlp_common import build_v14


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=DEFAULT_OUT)
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51881)
    p.add_argument('--sim-seconds',type=int,default=3600)
    p.add_argument('--seeds',type=int,nargs='+',default=[900001,900002,900003,900004,900005])
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args(argv)
    if a.sim_seconds<1 or not a.seeds or len(set(a.seeds))!=len(a.seeds):
        p.error('Require positive horizon and distinct held-out seeds')
    import torch
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
    from v14_deep_mlp_collector import V14EventCollector
    if str(a.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicit --device cpu required')
    run=a.run.resolve()
    meta=read_json(run/'experiment.json')
    if meta.get('version')!=VERSION or meta.get('source_checkpoint_loaded') is not False:
        raise RuntimeError('Not a V14 full-scratch output folder')
    cfg_copy=run/'run_config_blueprint.json'
    cfg=read_json(cfg_copy)
    if sha256(cfg_copy)!=meta['source_blueprint_sha256']:
        raise RuntimeError('Source blueprint SHA differs from original experiment')
    if sha256(Path(cfg['model_file']).resolve())!=meta['anylogic_sha256']:
        raise RuntimeError('Original AnyLogic model ZIP changed')
    if a.checkpoint is None:
        cp=Path(read_json(run/'state.json')['checkpoint']).resolve()
    else:
        cp=a.checkpoint.resolve()
    if cp.parent != run or not cp.is_file() or cp.suffix != '.pt':
        raise RuntimeError('Checkpoint must be a V14 .pt file INSIDE the run directory')
    if a.output is None:
        output=run/f'heldout_{cp.stem}_{a.sim_seconds}s_{len(a.seeds)}seeds.jsonl'
    else:
        output=a.output.resolve()
    if output.parent!=run:
        raise RuntimeError('Output must be placed inside V14 run folder')
    if output.exists():
        raise FileExistsError('Evaluation output already exists: '+str(output))
    logging.getLogger('alpyne.data').setLevel(logging.ERROR)
    logging.getLogger('alpyne.sim').setLevel(logging.ERROR)
    env=None
    try:
        env,model,spec,info=build_v14(cfg,a.device,a.port,a.sim_seconds,int(meta['max_events']),
            gat_layers=int(meta['gat_layers']),gat_heads=int(meta['gat_heads']),
            dispatch_width=int(meta['dispatch_width']),embedding=int(meta['embedding']),
            station_id_dim=int(meta['station_id_dim']),checkpoint=cp)
        auditor=RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        collector=V14EventCollector(model,env,stop_mask_mode=cfg['stop_mask_mode'])
        with output.open('x',encoding='utf-8') as f:
            for seed in a.seeds:
                auditor.reset()
                _,data=collector.episode(seed=seed,train=False,max_event_steps=int(meta['max_events']))
                if data.get('end_reason')=='max_event_steps':
                    raise RuntimeError('Heldout episode was capped by event limit: '+str(seed))
                if GraphSpec.from_hive(env.agent_hive).fingerprint!=spec.fingerprint:
                    raise RuntimeError('Graph changed during evaluation')
                report=auditor.report()
                if report.get('candidate_map_rate',0)<.99:
                    raise RuntimeError('Graph candidate mapping below 99%')
                row=dict(version=VERSION,evaluation=True,seed=int(seed),sim_seconds=a.sim_seconds,
                         checkpoint=str(cp),environment=data,graph_mapping=report,
                         architectures=info, dijkstra_guidance_strength=0.0)
                f.write(json.dumps(row,default=str,ensure_ascii=False)+'\n');f.flush()
                stats=data.get('statistics',{})
                diag=data.get('dispatch_choice_diagnostics',{})
                multi = {k:round(v.get('multi_choice_rate') or 0.0,3)
                         for k,v in diag.items()}
                print(f'[V14 EVAL seed={seed}] JobDone={stats.get("jobDone",0)} '
                      f'Block={stats.get("block",0)} LiveLock={stats.get("liveLock",0)} '
                      f'WrongTarget={stats.get("wrongTarget",0)} '
                      f'DispatchMultiChoice={multi}',flush=True)
    finally:
        if env is not None:env.close()
    print('[V14 EVAL DONE]',output,flush=True)

if __name__=='__main__':
    main()
