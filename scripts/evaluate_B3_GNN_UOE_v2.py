"""Held-out evaluation for B3-UOE-V2 models; no optimizer steps.
Requires installed V1 B3-GNN code and local AnyLogic model ZIP.
"""
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
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch
from train_B3_GNN_UOE_v2 import VERSION, build_v2_model, v1
from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
from thesis.mappo.routing_guidance_v121 import GuidedEventCollector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--seeds', type=int, nargs='+', default=[1001, 1002, 1003])
    p.add_argument('--sim-seconds', type=int, default=3600)
    p.add_argument('--port', type=int, default=51694)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path)
    p.add_argument('--java-exe', default=r'C:\Program Files\AnyLogic 8.9 Personal Learning Edition\jre\bin\java.exe')
    p.add_argument('--startup-timeout', type=int, default=180)
    p.add_argument('--lock-timeout', type=int, default=180)
    opt = p.parse_args()
    ckpt = opt.checkpoint.resolve()
    manifest = ckpt.parent / 'run_config.json'
    if not ckpt.is_file() or not manifest.is_file():
        p.error('Checkpoint AND run_config.json are required')
    c = json.loads(manifest.read_text(encoding='utf-8'))
    if c.get('version') != VERSION:
        p.error('This evaluator only reads V2 GNN checkpoints')
    if len(set(opt.seeds)) != len(opt.seeds) or opt.sim_seconds < 1:
        p.error('Distinct seeds, positive sim-seconds required')
    if set(opt.seeds) & set(range(c['seed'], c['seed'] + c['episodes'])):
        p.error('Evaluation seeds overlap with training')
    outfile = (opt.output or ckpt.parent/'evaluation_uoe_v2.jsonl').resolve()
    if outfile.exists():
        p.error('Refusing to overwrite evaluation: '+str(outfile))
    outfile.parent.mkdir(parents=True, exist_ok=True)
    args = SimpleNamespace(model=c['model_file'], port=opt.port,
        sim_seconds=opt.sim_seconds, max_events=c['max_events'],
        fleetsize=c['fleetsize'], max_fleetsize=c['max_fleetsize'],
        single_dispatch_policy=c['single_dispatch_policy'],
        routing_backbone='pooled', dispatch_backbone='pooled',
        stop_mask_mode=c['stop_mask_mode'], env_type='matrix',
        seed=opt.seeds[0], hidden=c['hidden'], graph_layers=c['graph_layers'],
        routing_guide_strength=c['routing_guide_strength'],
        routing_encoder=c['routing_encoder'], gat_heads=c['gat_heads'],
        gamma=c['gamma'], gae_lambda=c['gae_lambda'], clip=c['clip'],
        lr=c['lr'], epochs=c['epochs'], mini_batches=c['mini_batches'],
        device=opt.device, startup_timeout=opt.startup_timeout,
        lock_timeout=opt.lock_timeout, java_exe=opt.java_exe)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = None
    try:
        env = v1.baseline.build_env(args)
        model, graph, _ = build_v2_model(env, args)
        if graph.fingerprint != c['graph_fingerprint']:
            raise RuntimeError('Evaluation graph differs from training graph')
        # Load only checkpoints that you generated/trust locally.
        model.load(str(ckpt))
        audit = RoutingGraphAudit(model.roles['routing'].actor,
                                  sample_limit=c['diagnostic_samples'])
        collector = GuidedEventCollector(model, env,
            stop_mask_mode=c['stop_mask_mode'],
            routing_guide_strength=c['routing_guide_strength'])
        with outfile.open('x', encoding='utf-8') as stream:
            for seed in opt.seeds:
                audit.reset()
                _, metadata = collector.episode(seed=seed, train=False,
                                                max_event_steps=args.max_events)
                if GraphSpec.from_hive(env.agent_hive).fingerprint != graph.fingerprint:
                    raise RuntimeError('Dynamic graph detected in evaluation')
                metadata['gnn_graph_mapping'] = audit.report()
                stream.write(json.dumps(dict(seed=seed, version=VERSION,
                    checkpoint=str(ckpt), environment=metadata), default=str)+'\n')
                stream.flush()
                print(f'[B3-UOE-V2 EVAL] seed={seed} '+
                    f'jobDone={metadata.get("statistics", {}).get("jobDone", 0)} '+
                    f'candidate_map={metadata["gnn_graph_mapping"]["candidate_map_rate"]:.2%}', flush=True)
    finally:
        if env is not None:
            env.close()
    print('[B3-UOE-V2 EVAL] Saved:', outfile)


if __name__ == '__main__':
    main()
