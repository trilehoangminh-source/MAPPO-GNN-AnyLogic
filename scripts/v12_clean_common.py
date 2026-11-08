"""V12 builder + strict checkpoint migration. V8/V11 artifacts are read-only."""
from __future__ import annotations
import os
from pathlib import Path
from types import SimpleNamespace
from v11_common import validate_v8_cfg


def load_clean_checkpoint(model, checkpoint: Path, *, source_v11: bool):
    """Load 100% V11 trainable actor/critic/optimizer state and strip ONLY
    the stale static next-hop buffers from the legacy V11 source checkpoint.
    Use only checkpoints you created/trust: torch.load deserializes Python data.
    """
    import torch
    data = torch.load(str(checkpoint), map_location=model.device, weights_only=False)
    expected = set(model.roles)
    if set(data['roles']) != expected:
        raise RuntimeError('Checkpoint MAPPO role set mismatch')
    for name, role_data in data['roles'].items():
        role = model.roles[name]
        actor_state = dict(role_data['actor'])
        old_key = 'v11_traffic.next_hop'
        if name == 'routing':
            if source_v11:
                if old_key not in actor_state:
                    raise RuntimeError('Expected original V11 next-hop buffer for source conversion')
                actor_state.pop(old_key)
            elif old_key in actor_state:
                raise RuntimeError('V12 checkpoint still contains shortest-path buffer')
        role.actor.load_state_dict(actor_state, strict=True)
        role.critic.load_state_dict(role_data['critic'], strict=True)
        role.actor_optimizer.load_state_dict(role_data['actor_optimizer'])
        role.critic_optimizer.load_state_dict(role_data['critic_optimizer'])
        role.valuenorm.load_state_dict(role_data['valuenorm'])
    if any(not p.requires_grad for role in model.roles.values()
           for net in (role.actor, role.critic) for p in net.parameters()):
        raise RuntimeError('After checkpoint load, trainable parameters were frozen')
    actor_state = model.roles['routing'].actor.state_dict()
    if any('next_hop' in k for k in actor_state):
        raise RuntimeError('Shortest-path buffer survived V12 migration')
    return {'source_v11':source_v11,'removed_buffers':1 if source_v11 else 0}


def build_v12(cfg, source_checkpoint, device, port, sim_seconds, max_events,
              checkpoint=None, source_v11=False):
    import torch
    import train_B3_MAPPO_v8_overnight as v8
    import train_mappo_long_B3_v121 as baseline
    from thesis.mappo.mappo_stability_v7 import V7Options, enable_v7
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.congestion_edgegat_v12_clean import install_v12

    validate_v8_cfg(cfg)  # Validates historical V8 SOURCE, not a V12 guide.
    baseline.SIM_CONFIG['reward_part_completed'] = 2.0
    os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'operation_first'
    a = SimpleNamespace(model=Path(cfg['model_file']).resolve(),port_train=port,
        max_additional_episodes=10000, train_sim_seconds=sim_seconds,
        max_events=max_events, fleetsize=20, device=device, gamma=float(cfg['gamma']))
    if not a.model.is_file():
        raise FileNotFoundError('Missing AnyLogic ZIP: '+str(a.model))
    v7, args = v8.train_args(a, cfg)
    env = v7.v1.baseline.build_env(args)
    try:
        model, spec, _ = v7.make_v4_model(env,args)
        if GraphSpec.from_hive(env.agent_hive).fingerprint != cfg['graph_fingerprint']:
            raise RuntimeError('AnyLogic graph differs from V8/V11')
        enable_v7(model, V7Options(**cfg['ppo_v7_options']))
        model.load(str(source_checkpoint))    # original V8; strict
        installation = install_v12(model)     # NO shortest-path planning, NO action prior
        if checkpoint is not None:
            load_clean_checkpoint(model, checkpoint, source_v11=source_v11)
        from thesis.mappo.algorithm import MAPPO
        actor = model.roles['routing'].actor
        if actor.dist.__func__ is not type(actor).dist or model.act.__func__ is not MAPPO.act:
            raise RuntimeError('Legacy path-guidance wrapper still active')
        return env, model, spec, installation
    except Exception:
        env.close()
        raise
