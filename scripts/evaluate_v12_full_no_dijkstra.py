"""Held-out evaluation for V12 FULL no-Dijkstra and no shortest-path traffic. No PPO updates.

Use exact same seeds and sim_seconds as V8/V11, on a separate port.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from v11_common import read_json, sha256, validate_v8_cfg
from v12_clean_common import build_v12


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=ROOT / 'mappo_runs' / 'V12_FULL_NO_DIJKSTRA_FROM_V11')
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51881)
    p.add_argument('--sim-seconds',type=int,default=1800)
    p.add_argument('--seeds',type=int,nargs='+',default=[900001,900002,900003,900004,900005])
    p.add_argument('--output',type=Path,default=None)
    x=p.parse_args()
    if x.sim_seconds<1 or len(set(x.seeds))!=len(x.seeds) or not x.seeds:
        p.error('Invalid seeds or simulation seconds')
    import torch
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
    from v12_clean_collector import RawRoutingEventCollector
    if x.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    logging.getLogger('alpyne.data').setLevel(logging.ERROR)
    logging.getLogger('alpyne.sim').setLevel(logging.ERROR)
    run=x.run.resolve()
    meta=read_json(run/'experiment.json')
    if meta.get('version')!='V12_FULL_NO_DIJKSTRA_FROM_V11':
        raise RuntimeError('Not a V12 no-Dijkstra run')
    cfg=read_json(Path(meta['v8_config_file']))
    validate_v8_cfg(cfg)
    v8_ckpt=Path(meta['v8_source_checkpoint']).resolve()
    if sha256(v8_ckpt)!=meta['v8_source_sha256']:
        raise RuntimeError('V8 source checkpoint changed')
    if sha256(Path(meta['v8_config_file']))!=meta['v8_config_sha256']:
        raise RuntimeError('V8 source config changed')
    if sha256(Path(cfg['model_file']).resolve())!=meta['anylogic_model_sha256']:
        raise RuntimeError('AnyLogic model changed')
    if x.checkpoint is not None:
        cp=x.checkpoint.resolve()
    else:
        cp=Path(read_json(run/'state.json')['checkpoint']).resolve()
    if cp.parent!=run or not cp.is_file():
        raise RuntimeError('Checkpoint must be an existing file inside V12 output')
    if x.output is None:
        output=run/f'heldout_{cp.stem}_{len(x.seeds)}seeds.jsonl'
    else:
        output=x.output.resolve()
    if output.exists():
        raise FileExistsError('Evaluation output already exists: '+str(output))
    if output.parent != run:
        raise RuntimeError('Evaluation output must be inside V12 run folder')
    env=None
    try:
        env,model,spec,_=build_v12(cfg,v8_ckpt,x.device,x.port,x.sim_seconds,
               int(meta['max_events']),checkpoint=cp,
               source_v11=(cp.stem == f"v11_source_ep_{int(meta['source_episode']):04d}"))
        collector=RawRoutingEventCollector(model,env,stop_mask_mode=cfg['stop_mask_mode'])
        auditor=RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        with output.open('x',encoding='utf-8') as f:
            for seed in x.seeds:
                auditor.reset()
                _,md=collector.episode(seed=seed,train=False,max_event_steps=int(meta['max_events']))
                if GraphSpec.from_hive(env.agent_hive).fingerprint!=spec.fingerprint:
                    raise RuntimeError('AnyLogic graph mismatch')
                audit=auditor.report()
                if audit['candidate_map_rate']<.99:
                    raise RuntimeError('Bad graph mapping')
                record=dict(version='V12_HELDOUT_NO_DIJKSTRA_NO_SHORTEST_FLOW',seed=seed,evaluation=True,
                            checkpoint=str(cp),environment=md,graph_mapping=audit,
                            dijkstra_guidance_strength=0.0,shortest_path_traffic=False,sim_seconds=x.sim_seconds)
                f.write(json.dumps(record,ensure_ascii=False,default=str)+'\n')
                f.flush()
                stats=md.get('statistics',{})
                print(f'[V12 EVAL seed={seed}] JobDone={stats.get("jobDone",0)} '
                      f'Block={stats.get("block",0)} LiveLock={stats.get("liveLock",0)} '
                      f'WrongTarget={stats.get("wrongTarget",0)}',flush=True)
    finally:
        if env is not None:
            env.close()
    print('[V12 EVAL DONE]',output,flush=True)

if __name__=='__main__':main()
