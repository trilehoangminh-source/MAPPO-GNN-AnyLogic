"""V10-D true full-scratch wiring: no pretrained checkpoint is loaded."""
from __future__ import annotations
import os
from pathlib import Path
from v10_pure_common import make_run_args


def setup_from_scratch(cfg, device: str, port: int, sim_seconds: int,
                       max_events: int, hidden: int, heads: int, layers: int,
                       checkpoint=None):
    """Create EVERY actor and critic afresh; optionally restore this run's own checkpoint.

    We use the old V8 run_config.json only as an ENV/ARCHITECTURE BLUEPRINT.
    Neither source V8 weights nor V8 optimizer/ValueNorm states are read.
    """
    import train_mappo_long_B3_v121 as baseline
    from thesis.mappo.mappo_stability_v7 import V7Options, enable_v7
    from thesis.mappo.v10_pure_transformer import install_v10_pure
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    baseline.SIM_CONFIG['reward_part_completed'] = 2.0
    os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'operation_first'
    v7, args = make_run_args(cfg, device, port, sim_seconds, max_events)
    env = v7.v1.baseline.build_env(args)
    try:
        # This builds randomly initialized Residual Edge-GAT and pooled Dispatch nets.
        model, spec, _ = v7.make_v4_model(env, args)
        if spec.fingerprint != cfg['graph_fingerprint']:
            raise RuntimeError('AnyLogic graph fingerprint differs from baseline topology')
        enable_v7(model, V7Options(**cfg['ppo_v7_options']))
        # Replace BOTH random Dispatch Actors/Critics with NEW pure Transformer nets.
        # Critic ValueNorm and optimizers are freshly initialized too.
        feature_dim = int(model.roles['dispatch_front'].actor.encoder.main[0].in_features)
        n_stations = int(model.roles['dispatch_front'].actor.action_dim)
        if n_stations != int(model.roles['dispatch_rear'].actor.action_dim):
            raise RuntimeError('Dispatch action dimensions differ')
        install_v10_pure(model, feature_dim, n_stations, 'both', hidden, heads, layers)
        if checkpoint is not None:
            cp = Path(checkpoint).resolve()
            # Only checkpoint from this SAME scratch run is ever supplied here.
            model.load(str(cp))
        return env, model, spec
    except Exception:
        env.close()
        raise
