"""Read-only B3-UOE-V3.1 evaluation + Routing Decision Trace.

Loads an existing *V2* checkpoint, evaluates using its unchanged policy,
and writes one Routing decision JSONL per held-out seed. NO TRAINING.
DO NOT use a V1 checkpoint, or train and test on the same seeds.
"""
from __future__ import annotations

import os
os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'operation_first'
import argparse
import hashlib
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
from train_B3_GNN_UOE_v2 import VERSION as V2_VERSION, build_v2_model, v1
from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
from thesis.mappo.routing_guidance_v121 import GuidedEventCollector
from thesis.mappo.routing_trace_uoe_v3 import RoutingDecisionTrace

VERSION = 'B3_UOE_V3_1_DIAGNOSTIC_ONLY'


def _sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for buf in iter(lambda: f.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True,
                   help='V2 GAT/MP checkpoint; V3.1 does NOT alter its weights')
    p.add_argument('--seeds', type=int, nargs='+', default=[1001, 1002, 1003])
    p.add_argument('--sim-seconds', type=int, default=3600)
    p.add_argument('--port', type=int, default=51705)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', type=Path, required=True,
                   help='NEW, empty results directory (never overwrite old logs)')
    p.add_argument('--radius', type=float, default=6.0,
                   help='meters: last-node proximity proxy, NOT collision geometry')
    p.add_argument('--java-exe', default=r'C:\Program Files\AnyLogic 8.9 Personal Learning Edition\jre\bin\java.exe')
    p.add_argument('--startup-timeout', type=int, default=180)
    p.add_argument('--lock-timeout', type=int, default=180)
    args = p.parse_args()
    checkpoint = args.checkpoint.resolve()
    config_file = checkpoint.parent/'run_config.json'
    if not checkpoint.is_file() or not config_file.is_file():
        p.error('V2 checkpoint and its run_config.json must both exist')
    config = json.loads(config_file.read_text(encoding='utf-8'))
    if config.get('version') != V2_VERSION:
        p.error('Only V2 checkpoints supported, got '+str(config.get('version')))
    if len(args.seeds) != len(set(args.seeds)) or not args.seeds or args.sim_seconds <= 0:
        p.error('Require distinct held-out seeds and positive sim-seconds')
    if set(args.seeds).intersection(range(config['seed'], config['seed']+config['episodes'])):
        p.error('Held-out seeds overlap training seeds')
    if not 0 < args.radius < 100000:
        p.error('Radius must be positive')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        p.error('Refusing to overwrite existing output '+str(output))
    output.mkdir(parents=True, exist_ok=True)

    env_args = SimpleNamespace(
        model=config['model_file'], port=args.port, sim_seconds=args.sim_seconds,
        max_events=config['max_events'], fleetsize=config['fleetsize'],
        max_fleetsize=config['max_fleetsize'],
        single_dispatch_policy=config['single_dispatch_policy'],
        routing_backbone='pooled', dispatch_backbone='pooled',
        stop_mask_mode=config['stop_mask_mode'], env_type='matrix',
        seed=args.seeds[0], hidden=config['hidden'],
        graph_layers=config['graph_layers'],
        routing_guide_strength=config['routing_guide_strength'],
        routing_encoder=config['routing_encoder'], gat_heads=config['gat_heads'],
        gamma=config['gamma'], gae_lambda=config['gae_lambda'], clip=config['clip'],
        lr=config['lr'], epochs=config['epochs'], mini_batches=config['mini_batches'],
        device=args.device, startup_timeout=args.startup_timeout,
        lock_timeout=args.lock_timeout, java_exe=args.java_exe)
    if not Path(env_args.model).is_file():
        p.error('AnyLogic model ZIP missing: '+env_args.model)
    random.seed(env_args.seed)
    np.random.seed(env_args.seed)
    torch.manual_seed(env_args.seed)

    manifest = dict(version=VERSION, source_version=V2_VERSION,
        checkpoint=str(checkpoint), checkpoint_sha256=_sha256(checkpoint),
        graph_fingerprint=config['graph_fingerprint'],
        router=config['routing_encoder'], dispatch='B2 Operation-first non-GNN',
        seeds=args.seeds, sim_seconds=args.sim_seconds, local_radius_m=args.radius,
        train_seed=config['seed'], train_episodes=config['episodes'],
        original_model_file=env_args.model, no_training=True,
        warning='V3.1 reports observational associations only; no causal block attribution')
    (output/'diagnostic_run_config.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    env = None
    try:
        env = v1.baseline.build_env(env_args)
        model, graph, _ = build_v2_model(env, env_args)
        if graph.fingerprint != config['graph_fingerprint']:
            raise RuntimeError('Evaluation graph fingerprint != training graph')
        model.load(str(checkpoint))  # Load only checkpoints generated/trusted locally.
        audit = RoutingGraphAudit(model.roles['routing'].actor,
                                  sample_limit=config.get('diagnostic_samples', 2048))
        collector = GuidedEventCollector(model, env,
            stop_mask_mode=config['stop_mask_mode'],
            routing_guide_strength=config['routing_guide_strength'])
        tracer = RoutingDecisionTrace(graph, local_radius=args.radius)
        results = output/'evaluation_v3_diagnostic.jsonl'
        with results.open('x', encoding='utf-8') as f:
            for seed in args.seeds:
                audit.reset()
                trace_file = output / f'routing_trace_seed{seed}.jsonl'
                with tracer.attach(model, env, trace_file):
                    _, metadata = collector.episode(seed=seed, train=False,
                                                     max_event_steps=env_args.max_events)
                actual_graph = GraphSpec.from_hive(env.agent_hive)
                if actual_graph.fingerprint != graph.fingerprint:
                    raise RuntimeError('Topology changed inside evaluation')
                metadata['gnn_graph_mapping'] = audit.report()
                metadata['routing_trace_v3'] = tracer.report()
                (output/f'routing_trace_summary_seed{seed}.json').write_text(
                    json.dumps(tracer.report(), indent=2), encoding='utf-8')
                f.write(json.dumps(dict(seed=seed, version=VERSION,
                    source_checkpoint=str(checkpoint), environment=metadata),
                    default=str)+'\n')
                f.flush()
                kp = metadata.get('statistics', {})
                rt = metadata['routing_trace_v3']
                print(f'[V3.1 EVAL] seed={seed} '
                    f'jobDone={kp.get("jobDone", 0)} block={kp.get("block", 0)} '
                    f'liveLock={kp.get("liveLock", 0)} '
                    f'routing_decisions={rt.get("routing_decisions", 0)} '
                    f'guide_follow_rate={rt["guide_follow_rate"]:.1%}', flush=True)
    finally:
        if env is not None:
            env.close()
    print('[V3.1 EVAL] Read-only output:', output, flush=True)
    print('[V3.1 EVAL] Next: python scripts/analyze_B3_UOE_v3_trace.py --folder', output, flush=True)


if __name__ == '__main__':
    main()
