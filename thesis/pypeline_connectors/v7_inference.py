"""Checkpoint-backed inference for MAPPO V7 with no Alpyne server startup.

Requires local thesis/mappo modules from the original training repository.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch


def verify_checkpoint(checkpoint: Path, run_config: Path):
    checkpoint = Path(checkpoint)
    run_config = Path(run_config)
    cfg = json.loads(run_config.read_text(encoding="utf-8-sig"))
    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    roles = raw.get("roles")
    expected = {"routing", "dispatch_front", "dispatch_rear"}
    if not isinstance(roles, dict) or set(roles) != expected:
        raise ValueError("Expected 3 V7 roles: routing, dispatch_front, dispatch_rear")
    if cfg.get("routing_encoder") != "residual_edge_gat":
        raise ValueError("run_config is not a residual_edge_gat model")
    if cfg.get("dispatch_backbone") != "pooled" or cfg.get("dispatch_mask_mode") != "operation_first":
        raise ValueError("Mismatch with trained dispatch MLP/operation-first policy")
    if cfg.get("stop_mask_mode") != "forbid_when_move":
        raise ValueError("Unexpected Stop mask configuration")
    if cfg.get("graph_nodes") != 105 or cfg.get("graph_edges") != 328:
        raise ValueError("Unexpected graph topology")
    routing = roles["routing"]["actor"]
    xy, adjacency = routing["node_xy"], routing["outgoing_average"]
    if tuple(xy.shape) != (105, 2) or tuple(adjacency.shape) != (105, 105):
        raise ValueError("Checkpoint graph buffers have unexpected shapes")
    if int((adjacency > 0).sum()) != 328:
        raise ValueError("Graph adjacency edge count mismatch")
    if not all(f"layers.{layer}.attention.edge_score.weight" in routing for layer in (0, 1)):
        raise ValueError("Missing trained Edge-GAT attention layer weights")
    for role in ("dispatch_front", "dispatch_rear"):
        if tuple(roles[role]["actor"]["policy.2.weight"].shape) != (27, 64):
            raise ValueError(f"Dispatch head mismatch: {role}")
    return cfg, raw


def assert_live_graph_compatible(hive, run_cfg, raw):
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec

    live = GraphSpec.from_hive(hive)
    if live.fingerprint != run_cfg["graph_fingerprint"]:
        raise RuntimeError(
            "STOP: AnyLogic Desktop graph differs from trained graph. "
            f"Live={live.fingerprint[:12]}, trained={run_cfg['graph_fingerprint'][:12]}. "
            "Do not infer on a different model or layout."
        )
    if live.n != run_cfg["graph_nodes"] or len(live.arcs) != run_cfg["graph_edges"]:
        raise RuntimeError("LIVE topology dimensions differ from checkpoint")
    return live


def make_inference_policy(checkpoint: Path, run_config: Path, hive):
    """Reconstruct exact trained actor layout from checkpoint, not stale routing_backbone=pooled."""
    from thesis.mappo.algorithm import MAPPOConfig
    from thesis.mappo.routing_guidance_v121 import GuidedMAPPO, enable_routing_distribution_prior
    from thesis.mappo.gnn_routing_B3_v121 import GraphRoutingActor
    from thesis.mappo.edge_gat_routing_v4 import enable_edge_gat_routing
    from thesis.mappo.networks import Actor

    cfg, raw = verify_checkpoint(checkpoint, run_config)
    spec = assert_live_graph_compatible(hive, cfg, raw)

    feature_dim = int(raw["roles"]["routing"]["actor"]["entity_encoder.main.0.weight"].shape[1])
    # Input spaces need only .shape for building Actor, no simulated environment.
    space = {
        "agvs": SimpleNamespace(shape=(int(cfg["max_fleetsize"]), feature_dim)),
        "stat": SimpleNamespace(shape=(27, feature_dim)),
    }
    kwargs = dict(
        hidden=int(cfg["hidden"]), env_type="matrix", routing_backbone="pooled",
        dispatch_backbone="pooled", two_fleets=True,
        gamma=float(cfg["gamma"]), gae_lambda=float(cfg["gae_lambda"]),
        clip_epsilon=float(cfg["clip"]), actor_lr=float(cfg["lr"]),
        critic_lr=float(cfg["lr"]), epochs=int(cfg["epochs"]),
        mini_batches=int(cfg["mini_batches"]),
    )
    model = GuidedMAPPO({
        "routing": (space, 5), "dispatch_front": (space, 27),
        "dispatch_rear": (space, 27),
    }, feature_dim, MAPPOConfig(**kwargs), device="cpu")
    routing = model.roles["routing"]
    routing.actor = GraphRoutingActor(
        space, spec, hidden=int(cfg["hidden"]), layers=int(cfg["graph_layers"])
    )
    enable_edge_gat_routing(model, heads=int(cfg["heads"]))
    enable_routing_distribution_prior(model)

    for role_name, role in model.roles.items():
        role.actor.load_state_dict(raw["roles"][role_name]["actor"], strict=True)
        role.actor.eval()
    return model


@torch.inference_mode()
def greedy_action(model, role: str, obs: dict) -> int:
    """Use actor-only distribution; no critic or optimizer is needed in Desktop demo."""
    from thesis.mappo.algorithm import as_actor, batchify

    if role not in model.roles:
        raise ValueError(f"Unrecognized role {role}")
    actor = model.roles[role].actor
    actor.eval()
    dist = actor.dist(batchify(as_actor(obs, torch.device('cpu'))))
    valid = torch.as_tensor(obs['action_mask']) > 0.5
    result = int(dist.probs.argmax(dim=-1).item())
    if not bool(valid[result]):
        raise RuntimeError(f"Actor selected forbidden action {result}")
    return result
