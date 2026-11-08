"""Construct full-scratch V14 GAT4 / candidate-MLP / 3-layer critics.

Only V8 run_config.json is read as an environment/topology blueprint. Neither
V8 nor V11/V12/V13 weights/optimizer state are deserialized at EP0.
"""
from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace


def build_v14(cfg, device, port, sim_seconds, max_events, *,
              gat_layers=4, gat_heads=2, dispatch_width=128, embedding=64,
              station_id_dim=16, checkpoint=None):
    import train_B3_MAPPO_v8_overnight as v8
    import train_mappo_long_B3_v121 as baseline
    from thesis.mappo.mappo_stability_v7 import V7Options, enable_v7
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.edge_gat_routing_v4 import EdgeAwareLayer
    from thesis.mappo.congestion_edgegat_v12_clean import (
        install_v12, CongestionResidualEdgeGAT, TRAFFIC_NAMES)
    from thesis.mappo.v14_deep_candidate_mlp import (
        install_v14, CandidateAwareDispatchActor, DeepCentralizedCritic)
    from thesis.mappo.algorithm import MAPPO

    if (int(cfg['fleetsize']),int(cfg['max_fleetsize'])) != (20,20):
        raise ValueError('V14 requires 20 AGVs and 20 max fleet size')
    if (cfg.get('routing_encoder') != 'residual_edge_gat' or
        cfg.get('dispatch_backbone') != 'pooled' or
        cfg.get('dispatch_mask_mode') != 'operation_first'):
        raise ValueError('V14 requires original compatible V8 blueprint')
    if gat_layers != 4 or gat_heads < 1 or int(cfg['hidden']) % gat_heads:
        raise ValueError('This V14 experiment requires 4 GAT layers and valid attention head count')

    # Propagate override to train_args(), which reads cfg['graph_layers'].
    # NEVER modify the source config or environment topology fingerprint.
    effective_cfg=dict(cfg)
    effective_cfg['graph_layers']=int(gat_layers)
    effective_cfg['heads']=int(gat_heads)
    baseline.SIM_CONFIG['reward_part_completed']=2.0
    os.environ['MAPPO_DISPATCH_MASK_MODE']='operation_first'
    args=SimpleNamespace(
        model=Path(cfg['model_file']).resolve(),port_train=int(port),
        max_additional_episodes=10000,train_sim_seconds=int(sim_seconds),
        max_events=int(max_events),fleetsize=20,device=str(device),gamma=float(cfg['gamma']))
    if not args.model.is_file():
        raise FileNotFoundError(f'Original AnyLogic model ZIP does not exist: {args.model}')
    v7,v8args=v8.train_args(args,effective_cfg)
    if int(v8args.graph_layers)!=4 or int(v8args.edge_heads)!=gat_heads:
        raise RuntimeError('Effective GAT4 CLI arguments were not applied')
    env=v7.v1.baseline.build_env(v8args)
    try:
        model,spec,_=v7.make_v4_model(env,v8args)
        if GraphSpec.from_hive(env.agent_hive).fingerprint!=cfg['graph_fingerprint']:
            raise RuntimeError('Live AnyLogic graph fingerprint differs from blueprint')
        actor=model.roles['routing'].actor
        if len(actor.layers)!=gat_layers or not all(isinstance(l,EdgeAwareLayer) for l in actor.layers):
            raise RuntimeError('Routing architecture is not 4 Edge-GAT Message Passing layers')
        # Same stable PPO patches as previous V12/V13 experiments.
        enable_v7(model,V7Options(**cfg['ppo_v7_options']))
        v12=install_v12(model)
        actor=model.roles['routing'].actor
        if (len(actor.layers)!=gat_layers or
            not all(isinstance(l.attention,CongestionResidualEdgeGAT) for l in actor.layers)):
            raise RuntimeError('Exactly 4 congestion residual Edge-GAT layers were not installed')
        if not hasattr(actor,'v11_traffic') or hasattr(actor.v11_traffic,'next_hop'):
            raise RuntimeError('Observed-only traffic is missing or legacy planner still present')
        if actor.dist.__func__ is not type(actor).dist or model.act.__func__ is not MAPPO.act:
            raise RuntimeError('Unwanted Dijkstra guidance wrapper still active')
        role=model.roles['dispatch_front']
        feature_dim=int(role.actor.encoder.main[0].in_features)
        n_stations=int(role.actor.action_dim)
        if n_stations!=27 or role.actor.action_dim!=model.roles['dispatch_rear'].actor.action_dim:
            raise ValueError('Expected exactly 27 Dispatch action slots')
        role_info=install_v14(model,feature_dim,n_stations,
                               dispatch_width,embedding,station_id_dim)
        for name,role in model.roles.items():
            if not isinstance(role.critic,DeepCentralizedCritic):
                raise RuntimeError(f'{name} Critic not replaced')
            if name!='routing' and not isinstance(role.actor,CandidateAwareDispatchActor):
                raise RuntimeError(f'{name} actor not replaced')
        if checkpoint is not None:
            # Only a state-confirmed checkpoint FROM THIS SAME V14 scratch run.
            model.load(str(Path(checkpoint).resolve()))
        return env,model,spec,dict(routing_layers=4,routing_heads=gat_heads,
             traffic_channels=list(TRAFFIC_NAMES),no_dijkstra=bool(v12['guidance_removed']),
             dispatch_arch=role_info,n_station_slots=n_stations,feature_dim=feature_dim)
    except Exception:
        env.close()
        raise
