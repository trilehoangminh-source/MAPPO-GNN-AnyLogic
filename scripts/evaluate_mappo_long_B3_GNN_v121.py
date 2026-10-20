"""Held-out greedy evaluation for B3-GNN checkpoints (never trains)."""
from __future__ import annotations
import os
os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'operation_first'
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
import numpy as np
import torch
from train_mappo_long_B3_GNN_v121 import VERSION, build_gnn_model
import train_mappo_long_B3_v121 as baseline
from thesis.mappo.routing_guidance_v121 import GuidedEventCollector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[1001,1002,1003])
    p.add_argument('--sim-seconds', type=int, default=3600)
    p.add_argument('--port', type=int, default=51691)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--java-exe', default=r'C:\Program Files\AnyLogic 8.9 Personal Learning Edition\jre\bin\java.exe')
    p.add_argument('--startup-timeout', type=int, default=180)
    p.add_argument('--lock-timeout', type=int, default=180)
    opt=p.parse_args()
    ckpt=opt.checkpoint.resolve()
    if not ckpt.is_file() or not (ckpt.parent/'run_config.json').is_file():
        p.error('Expected checkpoint beside its run_config.json')
    c=json.loads((ckpt.parent/'run_config.json').read_text(encoding='utf-8'))
    if c.get('version')!=VERSION or c.get('routing_backbone')!='directed_graph_gnn':
        p.error('NOT a B3-GNN checkpoint. Do not load pooled/B2 model as GNN.')
    if len(set(opt.seeds))!=len(opt.seeds) or opt.sim_seconds<1:
        p.error('Require distinct held-out seeds, positive sim-seconds')
    if set(opt.seeds) & set(range(c['seed'], c['seed']+c['episodes'])):
        p.error('Evaluation seeds overlap with training seeds')
    output = (opt.output or ckpt.parent/'evaluation_gnn_3600.jsonl').resolve()
    if output.exists():
        p.error('Refusing to overwrite previous evaluation: '+str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    args=SimpleNamespace(model=c['model_file'], port=opt.port,
                         sim_seconds=opt.sim_seconds, max_events=c['max_events'],
                         fleetsize=c['fleetsize'], max_fleetsize=c['max_fleetsize'],
                         single_dispatch_policy=c['single_dispatch_policy'],
                         routing_backbone='pooled', dispatch_backbone='pooled',
                         stop_mask_mode=c['stop_mask_mode'], env_type='matrix',
                         seed=opt.seeds[0], hidden=c['hidden'],
                         graph_layers=c['graph_layers'],
                         routing_guide_strength=c['routing_guide_strength'],
                         gamma=c['gamma'], gae_lambda=c['gae_lambda'],clip=c['clip'],
                         lr=c['lr'],epochs=c['epochs'],mini_batches=c['mini_batches'],
                         device=opt.device,startup_timeout=opt.startup_timeout,
                         lock_timeout=opt.lock_timeout,java_exe=opt.java_exe)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    env=None
    try:
        env=baseline.build_env(args)
        model, graph=build_gnn_model(env,args)
        if graph.fingerprint != c['graph_fingerprint']:
            raise RuntimeError('Graph topology differs from checkpoint manifest; refusing evaluation')
        model.load(str(ckpt))
        collector=GuidedEventCollector(model,env,stop_mask_mode=c['stop_mask_mode'],
                                       routing_guide_strength=c['routing_guide_strength'])
        with output.open('x', encoding='utf-8') as out:
            for seed in opt.seeds:
                before=env._mappo_source_counts.copy()
                _,meta=collector.episode(seed=seed,train=False,max_event_steps=args.max_events)
                meta['station_state_sources']={
                    k:int(env._mappo_source_counts[k]-before[k])
                    for k in sorted(env._mappo_source_counts)}
                from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
                if GraphSpec.from_hive(env.agent_hive).fingerprint != graph.fingerprint:
                    raise RuntimeError('Hive graph changed during held-out evaluation')
                out.write(json.dumps(dict(seed=seed, checkpoint=str(ckpt),
                                          version=VERSION, environment=meta), default=str)+'\n')
                out.flush()
                k=meta.get('statistics', {})
                print(f'[EVAL B3-GNN] seed={seed} jobDone={k.get("jobDone",0)} '
                      f'targetReached={k.get("targetReached",0)} '
                      f'wrongTarget={k.get("wrongTarget",0)} '
                      f'guide={meta.get("routing_guidance",{}).get("guided",0)}', flush=True)
    finally:
        if env is not None:
            env.close()
    print('[EVAL B3-GNN] Saved:', output)

if __name__=='__main__':
    main()
