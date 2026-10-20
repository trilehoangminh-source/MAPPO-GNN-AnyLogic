"""Train NEW B3-GNN (directed message passing + B3 Dijkstra + B2 dispatch).

Additive trainer: relies on EXISTING B3 make_model/build_env, guided collector,
MAPPO training, CTDE critic, operation_first hook; changes routing actor only.
Fresh runs only (no silent resume). This script creates a distinct output folder.
"""
from __future__ import annotations

import os
os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'operation_first'
import argparse
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

import numpy as np
import torch
import train_mappo_long_B3_v121 as baseline
from thesis.mappo.gnn_routing_B3_v121 import GraphSpec, upgrade_guided_mappo_to_gnn
from thesis.mappo.routing_guidance_v121 import GuidedEventCollector

VERSION = 'B3_GNN_directed_mappo_v1'


def args_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=str(ROOT / 'envs' / 'Matrix_critic_fresh.zip'))
    p.add_argument('--port', type=int, default=51690)
    p.add_argument('--episodes', type=int, default=2)
    p.add_argument('--sim-seconds', type=int, default=720)
    p.add_argument('--max-events', type=int, default=20000)
    p.add_argument('--fleetsize', type=int, default=8)
    p.add_argument('--max-fleetsize', type=int, default=20)
    p.add_argument('--seed', type=int, default=43)
    p.add_argument('--output', default=str(ROOT/'mappo_runs'/'B3_GNN_smoke_seed43'))
    p.add_argument('--device', default='cpu')
    p.add_argument('--hidden', type=int, default=32)
    p.add_argument('--graph-layers', type=int, default=2)
    p.add_argument('--routing-guide-strength', type=float, default=1.5)
    p.add_argument('--gamma', type=float, default=.90)
    p.add_argument('--gae-lambda', type=float, default=.95)
    p.add_argument('--clip', type=float, default=.1)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--mini-batches', type=int, default=1)
    p.add_argument('--startup-timeout', type=int, default=180)
    p.add_argument('--lock-timeout', type=int, default=180)
    p.add_argument('--java-exe', default=r'C:\Program Files\AnyLogic 8.9 Personal Learning Edition\jre\bin\java.exe')
    # Baseline B3 creates the pre-upgrade Routing actor as pooled.
    p.set_defaults(routing_backbone='pooled', dispatch_backbone='pooled',
                   env_type='matrix', single_dispatch_policy=False,
                   stop_mask_mode='forbid_when_move')
    return p


def build_gnn_model(env, args):
    # Original make_model resets env and builds actual role-specific CTDE critics.
    model = baseline.make_model(env, args)
    graph = upgrade_guided_mappo_to_gnn(model, env.agent_hive,
                                        hidden=args.hidden, layers=args.graph_layers)
    print(f'[B3-GNN] REAL graph actor: nodes={graph.n} edges={len(graph.arcs)} '
          f'fingerprint={graph.fingerprint[:12]}; Dijkstra strength={args.routing_guide_strength}', flush=True)
    return model, graph


def main():
    p = args_parser()
    args = p.parse_args()
    if args.episodes < 1 or args.sim_seconds <= 0 or args.max_events < 1:
        p.error('Require positive episodes/sim-seconds/max-events')
    if args.max_fleetsize < args.fleetsize or args.graph_layers < 1 or args.hidden < 4:
        p.error('Invalid fleetsize, hidden, or graph-layers')
    if not 0 <= args.routing_guide_strength <= 20:
        p.error('--routing-guide-strength must be within [0, 20]')
    if args.epochs < 1 or args.mini_batches < 1:
        p.error('epochs and mini-batches must be positive')
    if not Path(args.model).is_file():
        p.error('Missing AnyLogic model ZIP: '+args.model)
    out = Path(args.output).resolve()
    if out.exists() and any(out.iterdir()):
        p.error(f'Output directory nonempty; WILL NOT overwrite: {out}')
    out.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    env = None
    try:
        env = baseline.build_env(args)
        model, graph = build_gnn_model(env, args)
        config = dict(version=VERSION, routing_backbone='directed_graph_gnn',
                      dispatch_backbone='pooled', dispatch_mask_mode='operation_first',
                      stop_mask_mode=args.stop_mask_mode,
                      routing_guide_strength=args.routing_guide_strength,
                      graph_fingerprint=graph.fingerprint, graph_nodes=graph.n,
                      graph_edges=len(graph.arcs), graph_layers=args.graph_layers,
                      model_file=str(Path(args.model).resolve()),
                      episodes=args.episodes, seed=args.seed,
                      sim_seconds=args.sim_seconds, max_events=args.max_events,
                      fleetsize=args.fleetsize, max_fleetsize=args.max_fleetsize,
                      single_dispatch_policy=args.single_dispatch_policy,
                      hidden=args.hidden, gamma=args.gamma, gae_lambda=args.gae_lambda,
                      clip=args.clip, lr=args.lr, epochs=args.epochs,
                      mini_batches=args.mini_batches,
                      torch_version=str(torch.__version__))
        (out/'run_config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        collector = GuidedEventCollector(model, env, stop_mask_mode=args.stop_mask_mode,
                                         routing_guide_strength=args.routing_guide_strength)
        with (out/'metrics.jsonl').open('x', encoding='utf-8') as stream:
            for ep in range(args.episodes):
                before = env._mappo_source_counts.copy()
                trajectories, metadata = collector.episode(seed=args.seed+ep, train=True,
                                                            max_event_steps=args.max_events)
                # Abort if AnyLogic model changes topology on reset, before PPO update.
                current = GraphSpec.from_hive(env.agent_hive)
                if current.fingerprint != graph.fingerprint:
                    raise RuntimeError('LIVE Hive graph changed during train: cannot use stale GNN topology')
                metadata['station_state_sources'] = {
                    k: int(env._mappo_source_counts[k] - before[k])
                    for k in sorted(env._mappo_source_counts)}
                if metadata.get('events', 0) and metadata.get('routing_guidance', {}).get('guided', 0) == 0:
                    raise RuntimeError('No Dijkstra guidance; inspect live graph mapping')
                update = model.train(trajectories)
                ckpt = out / f'checkpoint_ep_{ep+1:04d}.pt'
                temp = out / f'checkpoint_ep_{ep+1:04d}.pt.tmp'
                model.save(str(temp))
                os.replace(temp, ckpt)
                shutil.copy2(ckpt, out/'latest_mappo.pt')
                stream.write(json.dumps(dict(episode=ep+1, environment_seed=args.seed+ep,
                                             environment=metadata, policy_updates=update,
                                             version=VERSION), default=str)+'\n')
                stream.flush()
                kpi = metadata.get('statistics', {})
                rg = metadata.get('routing_guidance', {})
                print(f'[B3-GNN EP {ep+1}/{args.episodes}] reward={metadata.get("delivered_reward",0):.2f} '
                      f'jobDone={kpi.get("jobDone",0)} wrongTarget={kpi.get("wrongTarget",0)} '
                      f'guide={rg.get("guided",0)} guide_hit={rg.get("selected_guide",0)} '
                      f'mask_violations={metadata.get("action_diagnostics",{}).get("selected_mask_violations", "?")} '
                      f'closed={metadata.get("closed_transitions",0)}', flush=True)
    finally:
        if env is not None:
            env.close()
    print('[B3-GNN] Training complete; outputs:', out)


if __name__ == '__main__':
    main()
