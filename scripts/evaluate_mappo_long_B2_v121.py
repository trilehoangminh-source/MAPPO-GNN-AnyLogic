"""Greedy held-out evaluation of paper-inspired B2 operation-first checkpoint.

Does NOT update MAPPO weights. Uses same AnyLogic inputs as training, but a
longer simulation horizon (default 3600 s) and independent evaluation seeds.
"""
from __future__ import annotations
import argparse
import os
os.environ["MAPPO_DISPATCH_MASK_MODE"] = "operation_first"
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
import train_mappo_stop_ab as training
from thesis.mappo.collector_stop_ab import EventCollector


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[105,106,107])
    p.add_argument('--sim-seconds', type=int, default=3600)
    p.add_argument('--port', type=int, default=51620)
    p.add_argument('--output', type=Path, default=None)
    p.add_argument('--device', default='cpu')
    p.add_argument('--startup-timeout', type=int, default=180)
    p.add_argument('--lock-timeout', type=int, default=180)
    p.add_argument('--java-exe', default=r'C:\Program Files\AnyLogic 8.9 Personal Learning Edition\jre\bin\java.exe')
    opt=p.parse_args()
    checkpoint=opt.checkpoint.resolve()
    manifest=checkpoint.parent/'run_config.json'
    if not checkpoint.is_file() or not manifest.is_file():
        p.error('Checkpoint and run_config.json must be in the same training output directory.')
    c=json.loads(manifest.read_text(encoding='utf-8'))
    if c.get('dispatch_mask_mode')!='operation_first' or c['stop_mask_mode']!='forbid_when_move' or (c['routing_backbone'],c['dispatch_backbone'])!=('pooled','pooled'):
        p.error('Only pooled B2 operation_first checkpoint is supported.')
    if opt.sim_seconds<=0 or len(set(opt.seeds))!=len(opt.seeds):
        p.error('Positive simulation time and distinct seeds are required.')
    if set(opt.seeds) & set(range(c['seed'],c['seed']+c['episodes'])):
        p.error('Use HELD-OUT seeds not used to train this checkpoint.')
    output=(opt.output or (checkpoint.parent / 'evaluation_3600.jsonl')).resolve()
    if output.exists():
        p.error(f'Output already exists, will not overwrite: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(
        model=c['model_file'], port=opt.port, startup_timeout=opt.startup_timeout,
        lock_timeout=opt.lock_timeout, java_exe=opt.java_exe,
        fleetsize=c['fleetsize'], max_fleetsize=c['max_fleetsize'],
        sim_seconds=opt.sim_seconds, max_events=c['max_events'],
        routing_backbone='pooled',dispatch_backbone='pooled',
        single_dispatch_policy=c['single_dispatch_policy'],
        gamma=c['gamma'], gae_lambda=c['gae_lambda'],clip=c['clip'],
        lr=c['lr'],epochs=c['epochs'],mini_batches=c['mini_batches'],
        hidden=c['hidden'],env_type='matrix',device=opt.device,
        seed=int(opt.seeds[0]),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env=None
    try:
        env=training.build_env(args)
        model=training.make_model(env,args)
        model.load(str(checkpoint))
        collector=EventCollector(model,env,stop_mask_mode=c['stop_mask_mode'])
        with output.open('w', encoding='utf-8') as fp:
            for seed in opt.seeds:
                before=env._mappo_source_counts.copy()
                _,meta=collector.episode(seed=seed,train=False,max_event_steps=args.max_events)
                meta['station_state_sources']={
                    k:int(env._mappo_source_counts[k]-before[k]) for k in sorted(env._mappo_source_counts)
                }
                fp.write(json.dumps({'seed':seed,'checkpoint':str(checkpoint),'environment':meta},default=str)+'\n')
                fp.flush()
                s=meta.get('statistics',{})
                print(f'[EVAL] seed={seed} reward={meta["delivered_reward"]:.2f} '
                      f'jobDone={s.get("jobDone",0)} targetReached={s.get("targetReached",0)} '
                      f'wrongTarget={s.get("wrongTarget",0)} invalid={s.get("taken_invalid",0)} '
                      f'fresh_routing={meta["station_state_sources"].get("fresh_routing",0)}',flush=True)
    finally:
        if env is not None:env.close()
    print(f'[DONE] Evaluation saved to {output}')


if __name__=='__main__':main()
