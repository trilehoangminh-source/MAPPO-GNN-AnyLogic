"""V4: Edge-aware residual GAT Routing + annealed Dijkstra prior + MAPPO.

Adds new components; reuses the existing Alpyne/AnyLogic, B2 MLP Dispatching,
Dijkstra teacher, PPO/CTDE critic. Fresh checkpoints; never resume blindly.
"""
from __future__ import annotations

import os
os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'operation_first'
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))

import numpy as np
import torch
import train_mappo_long_B3_GNN_v121 as v1
from thesis.mappo.edge_gat_routing_v4 import enable_edge_gat_routing
from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
from thesis.mappo.routing_guidance_v121 import GuidedEventCollector
from thesis.mappo.gnn_audit_uoe_v2 import (
    RoutingGraphAudit, verify_ppo_rollout_parity, verify_dispatch_backbone)

VERSION = 'B3_EDGE_GAT_MAPPO_V4'


def make_v4_model(env, args):
    # Builds original MAPPO+CTDE and installs GNN only for Routing.
    model, graph = v1.build_gnn_model(env, args)
    edge = enable_edge_gat_routing(model, heads=args.edge_heads)
    dispatch = verify_dispatch_backbone(model)
    print('[EDGE-GAT-V4] graph=', graph.n, 'nodes,', len(graph.arcs), 'arcs;'
          'heads=', args.edge_heads, 'dispatch non-GNN=', dispatch, flush=True)
    return model, graph, edge


def guide_schedule(ep: int, start: float, end: float, decay_episodes: int) -> float:
    """Zero-based episode schedule, explicitly STORED with trajectory bias.

    A change of strength occurs between rollouts only (NEVER during PPO update).
    """
    if decay_episodes <= 1:
        return float(end)
    alpha = min(1.0, max(0.0, ep / (decay_episodes-1)))
    return float(start + (end - start)*alpha)


def args_parser():
    p = v1.args_parser()
    p.description = __doc__
    p.set_defaults(output=str(ROOT/'mappo_runs'/'B3_EDGE_GAT_V4_smoke_seed43'),
                   episodes=2, sim_seconds=720, routing_guide_strength=0.5)
    p.add_argument('--edge-heads', type=int, default=2)
    p.add_argument('--guide-final', type=float, default=0.2,
                   help='Final Dijkstra logit bonus as annealing finishes')
    p.add_argument('--guide-decay-episodes', type=int, default=10)
    p.add_argument('--diagnostic-samples', type=int, default=1024)
    p.add_argument('--parity-samples', type=int, default=64)
    p.add_argument('--min-candidate-map-rate', type=float, default=0.99)
    return p


def main():
    p = args_parser()
    a = p.parse_args()
    if a.episodes < 1 or a.max_events < 1 or a.sim_seconds < 1 or a.epochs < 1 or a.mini_batches < 1:
        p.error('Require positive episodes/sim_seconds/max_events/epochs/minibatches')
    if a.graph_layers < 1 or a.hidden < 4 or a.hidden % a.edge_heads:
        p.error('hidden must be >=4, multiple of edge-heads, graph-layers >=1')
    if not 0 <= a.guide_final <= a.routing_guide_strength <= 20:
        p.error('Require 0 <= guide-final <= routing-guide-strength <=20')
    if a.guide_decay_episodes < 1 or not 0 <= a.min_candidate_map_rate <= 1:
        p.error('Invalid schedule or graph map rate')
    if not Path(a.model).is_file():
        p.error('AnyLogic .zip not found: '+a.model)
    out = Path(a.output).resolve()
    if out.exists() and any(out.iterdir()):
        p.error('Refusing to overwrite output: '+str(out))
    out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    env = None
    try:
        env = v1.baseline.build_env(a)
        model, spec, edge = make_v4_model(env, a)
        auditor = RoutingGraphAudit(model.roles['routing'].actor,
                                    sample_limit=a.diagnostic_samples)
        config = dict(version=VERSION, routing_encoder='residual_edge_gat',
                      routing_backbone='directed_graph_gnn',
                      dispatch_backbone='pooled', dispatch_mask_mode='operation_first',
                      graph_fingerprint=spec.fingerprint, graph_nodes=spec.n,
                      graph_edges=len(spec.arcs), graph_layers=a.graph_layers,
                      heads=a.edge_heads, model_file=str(Path(a.model).resolve()),
                      episodes=a.episodes, sim_seconds=a.sim_seconds, max_events=a.max_events,
                      seed=a.seed, fleetsize=a.fleetsize, max_fleetsize=a.max_fleetsize,
                      single_dispatch_policy=a.single_dispatch_policy,
                      stop_mask_mode=a.stop_mask_mode, hidden=a.hidden,
                      gamma=a.gamma, gae_lambda=a.gae_lambda, clip=a.clip,
                      lr=a.lr, epochs=a.epochs, mini_batches=a.mini_batches,
                      guide_start=a.routing_guide_strength, guide_final=a.guide_final,
                      guide_decay_episodes=a.guide_decay_episodes,
                      diagnostic_samples=a.diagnostic_samples, parity_samples=a.parity_samples,
                      min_candidate_map_rate=a.min_candidate_map_rate,
                      final_used_guide_strength=guide_schedule(a.episodes-1,
                          a.routing_guide_strength, a.guide_final,a.guide_decay_episodes),
                      architecture=edge, torch_version=torch.__version__,
                      sources=['Lei-Kun/DRL-and-graph-neural-network-for-routing-problems',
                               'aalp75/neural-execution-graph-algorithms'],
                      note='Independent adaptation; no upstream code copied; no claimed KPI gain.')
        (out/'run_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        collector = GuidedEventCollector(model, env, stop_mask_mode=a.stop_mask_mode,
                                         routing_guide_strength=a.routing_guide_strength)
        with (out/'metrics.jsonl').open('x',encoding='utf-8') as f:
            for ep in range(a.episodes):
                strength = guide_schedule(ep, a.routing_guide_strength,a.guide_final,a.guide_decay_episodes)
                collector.guide.strength = strength
                auditor.reset()
                trajectories, meta = collector.episode(
                    seed=a.seed+ep, train=True,max_event_steps=a.max_events)
                live = GraphSpec.from_hive(env.agent_hive)
                if live.fingerprint != spec.fingerprint:
                    raise RuntimeError('Graph topology changed: stop PPO before update')
                graph_report = auditor.report()
                meta['gnn_graph_mapping'] = graph_report
                meta['dijkstra_strength_this_episode'] = strength
                if graph_report['candidate_map_rate'] < a.min_candidate_map_rate:
                    (out/f'failed_mapping_ep_{ep+1:04d}.json').write_text(
                        json.dumps(graph_report,indent=2),encoding='utf-8')
                    raise RuntimeError('Unmapped candidates! Aborting before PPO update')
                parity = verify_ppo_rollout_parity(model, trajectories,max_samples=a.parity_samples)
                meta['ppo_preupdate_audit'] = parity
                # PPO reuses routing_guide_bias STORED per observation in rollout.
                updates = model.train(trajectories)
                ckpt = out/f'checkpoint_ep_{ep+1:04d}.pt'
                tmp = out/f'checkpoint_ep_{ep+1:04d}.tmp'
                model.save(str(tmp))
                os.replace(tmp,ckpt)
                shutil.copy2(ckpt,out/'latest_mappo.pt')
                f.write(json.dumps(dict(version=VERSION,episode=ep+1,
                    environment_seed=a.seed+ep,environment=meta,
                    policy_updates=updates),default=str)+'\n')
                f.flush()
                stats = meta.get('statistics', {})
                print(f'[EDGE-GAT-V4 EP {ep+1}/{a.episodes}] guide={strength:.3f} '
                      f'reward={meta.get("delivered_reward",0):.2f} '
                      f'jobDone={stats.get("jobDone",0)} block={stats.get("block",0)} '
                      f'liveLock={stats.get("liveLock",0)} '
                      f'candidate_map={graph_report["candidate_map_rate"]:.1%} '
                      f'PPO_logp_err={parity["max_logprob_error"]:.2g}', flush=True)
    finally:
        if env is not None:
            env.close()
    print('[EDGE-GAT-V4] Training saved:',out)


if __name__=='__main__':
    main()
