"""Build V15 from scratch using the same AnyLogic/role/PPO infrastructure as V14.

Unlike build_v14 this module does not call GuidedMAPPO, install_v12, install_v14
or any routing/reward wrapper. GAT has no Dijkstra guidance. PPO unchanged.

For phase dispatch8 ONLY, Matrix(pseudo_routing=True) uses Tobias's
native CollisionFreeRouting/CollisionFreeRoutingHive. This does its own
time-window reservation graph search, without teaching the GAT via Dijkstra.
"""
from __future__ import annotations
from pathlib import Path
import os

# Native Tobias MultiDispAgent dispatch mask: union of eligible GEO and
# processing stations plus ALL buffer nodes. Do not impose 'operation_first'.
TOBIAS_DISPATCH_MASK_MODE = 'legacy'

# Explicit joint-training reward config from Tobias's training_disp_rout.ipynb.
# reward_part_completed is COMMENTED OUT there, so deliberately omitted here.
TOBIAS_JOINT_REWARDS=dict(
    dispatch=True, routing_ma=True, dispatching_ma=True,
    reward_reached_target=10, reward_wrong_target=-1,
    reward_removed_for_block=-20, reward_target_distance=0,
    reward_invalid=-0.1, reward_duration=-0.5, reward_pass=0.2,
    block_timeout=60, station_separate=False,
    reward_accepted_in_station=0.5,
    reward_declined_in_station=-0.25, reward_geo_operation=0.25,
    reward_reduce=-0.005, routing_interval=2, dispatching_interval=360,
    io_quote=0.9, availability=0.9, mttr=5,
    fixed_fleets=True, remove_all_blocked=False,
)


def build_v15(cfg, *, model_path, fleet_size, port, device, seed, max_events,
              sim_seconds=3600, checkpoint=None, phase='routing8'):
    """Construct V15; dispatch8 uses native Tobias pseudo-routing, all other phases GAT."""
    from v15_staged_collector import PHASES
    if phase not in PHASES or int(fleet_size)!=PHASES[phase][0]:
        raise ValueError('V15 phase/fleet mismatch')
    use_conventional_routing = (phase == 'dispatch8')
    if sim_seconds != 3600 or fleet_size not in (8,12,16,20):
        raise ValueError('V15 supports 3600s and fleet size 8,12,16,20 only')
    if int(cfg['max_fleetsize'])!=20:
        raise ValueError('Blueprint must have max_fleetsize=20')
    if int(max_events)<1: raise ValueError('max_events must be positive')
    from thesis.envs.matrix_v121 import Matrix
    from thesis.mappo.algorithm import MAPPO,MAPPOConfig
    from thesis.mappo.mappo_stability_v7 import enable_v7,V7Options
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.v15_tobias_arch import install_v15

    os.environ['MAPPO_DISPATCH_MASK_MODE']=TOBIAS_DISPATCH_MASK_MODE
    env=Matrix(model_path=str(Path(model_path).resolve()),
               startport=int(port),fleetsize=int(fleet_size),fleetsize_upper=None,
               max_fleetsize=20,sim_config=dict(TOBIAS_JOINT_REWARDS),
               max_seconds=int(sim_seconds),max_steps=int(max_events),
               pseudo_routing=use_conventional_routing,pseudo_dispatcher=False,
               routing_agent_death=True,death_on_target=False,
               transform_dispatching_partobs=True,direction_reward=-0.1)
    try:
        env.reset(seed=int(seed))
        # Assert native Tobias conventional Routing is truly active in dispatch8.
        # Its actions are executed internally in BaseAlpyneZoo._collect():
        # CollisionFreeRouting.is_for_learning() returns False.
        if use_conventional_routing:
            from thesis.envs.behaviors_v121 import CollisionFreeRouting, CollisionFreeRoutingHive
            if not isinstance(env.agent_hive, CollisionFreeRoutingHive):
                raise RuntimeError('dispatch8 requires CollisionFreeRoutingHive')
            if any(not isinstance(env.agent_behaviors.get(str(i)), CollisionFreeRouting)
                   for i in range(int(fleet_size))):
                raise RuntimeError('dispatch8 requires the native Tobias router for each AGV')
            if any(str(a).isdigit() and int(a)<1000 for a in env.agents):
                raise RuntimeError('dispatch8 Routing unexpectedly marked for PPO training')
        elif bool(env.pseudo_routing):
            raise RuntimeError('Learned GAT phase cannot use conventional routing')
        graph=GraphSpec.from_hive(env.agent_hive)
        if cfg.get('graph_fingerprint') and cfg['graph_fingerprint']!=graph.fingerprint:
            raise ValueError('Live AnyLogic graph differs from blueprint')
        obs_space=env.observation_space(env.agent_selection)
        gs=env.get_mappo_state()
        if int(obs_space['agvs'].shape[0])!=20 or int(gs['agvs'].shape[0])!=20:
            raise ValueError('Expected 20 padded physical AGV slots across all curriculum stages')
        if int(obs_space['stat'].shape[0])!=int(gs['stat'].shape[0]):
            raise ValueError('Station actor/global state row count mismatch')
        station_slots=int(gs['stat'].shape[0]);feat=int(gs['agvs'].shape[-1])
        roles={'routing':(obs_space,5),
               'dispatch_front':(obs_space,station_slots),
               'dispatch_rear':(obs_space,station_slots)}
        # The original MAPPO implementation and its config remain untouched.
        # Construct with lightweight placeholder actors, replace before EP0.
        mcfg=MAPPOConfig(gamma=float(cfg['gamma']),gae_lambda=float(cfg['gae_lambda']),
             clip_epsilon=float(cfg['clip']),actor_lr=float(cfg['lr']),
             critic_lr=float(cfg['lr']),epochs=int(cfg['epochs']),
             mini_batches=int(cfg['mini_batches']),hidden=32,
             routing_backbone='pooled',dispatch_backbone='pooled',
             env_type='matrix',two_fleets=True)
        model=MAPPO(roles,feat,mcfg,device=str(device))
        arch=install_v15(model,obs_space,feat,graph)
        enable_v7(model,V7Options(**dict(cfg['ppo_v7_options'])))
        if checkpoint is not None: model.load(str(Path(checkpoint).resolve()))
        if getattr(model.roles['routing'].actor,'_b3_guide_dist',False):
            raise RuntimeError('Unexpected Dijkstra policy wrapper')
        if type(model).__name__!='MAPPO':
            raise RuntimeError('V15 must use UNGUIDED MAPPO, not GuidedMAPPO')
        return env,model,graph,dict(arch,station_slots=station_slots,global_feature_dim=feat,
                                    padded_agv_slots=20,sim_seconds=3600,guide_strength=0.,
                                    native_conventional_routing=use_conventional_routing,
                                    dispatch_mask_mode=TOBIAS_DISPATCH_MASK_MODE,
                                    routing_controller=('tobias_collision_free_reservation'
                                                        if use_conventional_routing else 'unguided_gatv2'))
    except BaseException:
        env.close()
        raise
