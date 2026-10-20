"""Safe B3-GNN V2 experiment: optionally use directed GAT for Routing only.

Reuses EXISTING MAPPO, Dijkstra, AnyLogic, collector and B2 Operation-first.
Adds uoe-agents-inspired audit and experiment manifest; fresh outputs only.
"""
from __future__ import annotations

import os
os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'operation_first'
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch
import train_mappo_long_B3_GNN_v121 as v1
from thesis.mappo.gnn_attention_uoe_v2 import enable_gat_routing
from thesis.mappo.gnn_audit_uoe_v2 import (
    RoutingGraphAudit, verify_ppo_rollout_parity, verify_dispatch_backbone)
from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
from thesis.mappo.routing_guidance_v121 import GuidedEventCollector

VERSION = 'B3_GNN_UOE_v2_routing_only'


def build_v2_model(env, args):
    # V1 creates directed MP GNN, topology, Dijkstra bias; B2 dispatch unchanged.
    model, spec = v1.build_gnn_model(env, args)
    if args.routing_encoder == 'gat':
        attention_meta = enable_gat_routing(model, heads=args.gat_heads)
    else:
        attention_meta = dict(encoder='directed_mp', graph_layers=args.graph_layers)
    dispatch_roles = verify_dispatch_backbone(model)
    print('[B3-UOE-V2] Routing:', attention_meta,
          '| Dispatching NON-GNN:', dispatch_roles, flush=True)
    return model, spec, attention_meta


def args_parser():
    p = v1.args_parser()
    p.description = __doc__
    p.set_defaults(output=str(ROOT / 'mappo_runs' / 'B3_UOE_v2_GAT_smoke_seed43'))
    p.add_argument('--routing-encoder', choices=('mp', 'gat'), default='gat')
    p.add_argument('--gat-heads', type=int, default=2)
    p.add_argument('--diagnostic-samples', type=int, default=2048)
    p.add_argument('--parity-samples', type=int, default=64)
    p.add_argument('--max-parity-error', type=float, default=1e-4)
    p.add_argument('--min-candidate-map-rate', type=float, default=0.0,
                   help='If >0, stop before PPO update when legal-node mapping is below this fraction')
    return p


def main():
    parser = args_parser()
    args = parser.parse_args()
    if args.episodes < 1 or args.sim_seconds <= 0 or args.max_events <= 0:
        parser.error('Require positive episodes, sim-seconds, max-events')
    if args.epochs < 1 or args.graph_layers < 1 or args.mini_batches < 1:
        parser.error('Require positive epochs, graph layers and minibatches')
    if not 0 <= args.min_candidate_map_rate <= 1 or args.max_parity_error <= 0:
        parser.error('Invalid diagnostic threshold')
    if args.gat_heads < 1 or args.hidden % args.gat_heads:
        parser.error('--hidden must be divisible by --gat-heads')
    if not Path(args.model).is_file():
        parser.error('Missing AnyLogic model ZIP: ' + args.model)
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        parser.error('Refusing to overwrite B2/B3 or previous V2 outputs: '+str(out))
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = None
    try:
        env = v1.baseline.build_env(args)
        model, spec, extra = build_v2_model(env, args)
        auditor = RoutingGraphAudit(model.roles['routing'].actor,
                                    sample_limit=args.diagnostic_samples)
        manifest = dict(version=VERSION, routing_backbone='directed_graph_gnn',
                        routing_encoder=args.routing_encoder, gat_heads=args.gat_heads,
                        dispatch_backbone='pooled', dispatch_mask_mode='operation_first',
                        graph_fingerprint=spec.fingerprint, graph_nodes=spec.n,
                        graph_edges=len(spec.arcs), graph_layers=args.graph_layers,
                        routing_guide_strength=args.routing_guide_strength,
                        stop_mask_mode=args.stop_mask_mode,
                        model_file=str(Path(args.model).resolve()),
                        episodes=args.episodes, seed=args.seed,
                        sim_seconds=args.sim_seconds, max_events=args.max_events,
                        fleetsize=args.fleetsize, max_fleetsize=args.max_fleetsize,
                        single_dispatch_policy=args.single_dispatch_policy,
                        hidden=args.hidden, gamma=args.gamma, gae_lambda=args.gae_lambda,
                        clip=args.clip, lr=args.lr, epochs=args.epochs,
                        mini_batches=args.mini_batches,
                        diagnostic_samples=args.diagnostic_samples,
                        parity_samples=args.parity_samples,
                        max_parity_error=args.max_parity_error,
                        min_candidate_map_rate=args.min_candidate_map_rate,
                        torch_version=torch.__version__, architecture_metadata=extra,
                        origin='EPyMARL diagnostics + TA-RWARE-inspired state checks; no upstream source copied')
        (out/'run_config.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        collector = GuidedEventCollector(model, env,
            stop_mask_mode=args.stop_mask_mode,
            routing_guide_strength=args.routing_guide_strength)
        with (out/'metrics.jsonl').open('x', encoding='utf-8') as stream:
            for ep in range(args.episodes):
                auditor.reset()
                before = env._mappo_source_counts.copy()
                trajectories, metadata = collector.episode(
                    seed=args.seed+ep, train=True, max_event_steps=args.max_events)
                current = GraphSpec.from_hive(env.agent_hive)
                if current.fingerprint != spec.fingerprint:
                    raise RuntimeError('Topology changed during training; cannot reuse frozen directed graph')
                graph_report = auditor.report()  # BEFORE PPO-parity evaluation
                metadata['gnn_graph_mapping'] = graph_report
                metadata['station_state_sources'] = {
                    k:int(env._mappo_source_counts[k] - before[k])
                    for k in sorted(env._mappo_source_counts)}
                if metadata.get('events', 0) and metadata.get('routing_guidance', {}).get('guided', 0) == 0:
                    raise RuntimeError('No Dijkstra guidance was used; stop to inspect graph mapping')
                if (args.min_candidate_map_rate > 0 and
                    graph_report['candidate_map_rate'] < args.min_candidate_map_rate):
                    (out/f'failed_graph_mapping_ep_{ep+1:04d}.json').write_text(
                        json.dumps(graph_report, indent=2), encoding='utf-8')
                    raise RuntimeError('Legal candidate mapping rate under threshold; PPO NOT updated')
                parity = verify_ppo_rollout_parity(model, trajectories,
                    max_samples=args.parity_samples, atol=args.max_parity_error)
                metadata['ppo_preupdate_audit'] = parity
                update = model.train(trajectories)
                checkpoint = out / f'checkpoint_ep_{ep+1:04d}.pt'
                tmp = out / f'checkpoint_ep_{ep+1:04d}.pt.tmp'
                model.save(str(tmp))
                os.replace(tmp, checkpoint)
                shutil.copy2(checkpoint, out/'latest_mappo.pt')
                stream.write(json.dumps(dict(episode=ep+1,
                    environment_seed=args.seed+ep, environment=metadata,
                    policy_updates=update, version=VERSION), default=str)+'\n')
                stream.flush()
                kp = metadata.get('statistics', {})
                print(f"[B3-UOE-V2 EP {ep+1}/{args.episodes}] "
                      f"reward={metadata.get('delivered_reward',0):.2f} "
                      f"jobDone={kp.get('jobDone',0)} wrongTarget={kp.get('wrongTarget',0)} "
                      f"ego_map={graph_report['ego_map_rate']:.2%} "
                      f"target_map={graph_report['target_map_rate']:.2%} "
                      f"candidate_map={graph_report['candidate_map_rate']:.2%} "
                      f"PPO_logp_err={parity['max_logprob_error']:.2g}", flush=True)
    finally:
        if env is not None:
            env.close()
    print('[B3-UOE-V2] Saved experiment to:', out, flush=True)


if __name__ == '__main__':
    main()
