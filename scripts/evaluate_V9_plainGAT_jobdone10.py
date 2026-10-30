"""Held-out evaluation for FRESH V9 plain-GAT JobDone+10 checkpoints.

Must use this evaluator, not evaluate_B3_MAPPO_v7.py (which builds Residual-GAT).
No optimizer updates or checkpoint writes occur here.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import torch
import train_B3_MAPPO_v7 as v7
from train_V9_plainGAT_jobdone10_5h import VERSION, sha256
from thesis.mappo.plain_gat_routing_v9 import enable_plain_gat_routing
from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit, verify_dispatch_backbone
from thesis.mappo.routing_guidance_v121 import GuidedEventCollector


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[900001, 900002, 900003, 900004, 900005])
    p.add_argument("--sim-seconds", type=int, default=1800)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--port", type=int, default=51881)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--java-exe", default=r"C:\Program Files\AnyLogic 8.9 Personal Learning Edition\jre\bin\java.exe")
    p.add_argument("--startup-timeout", type=int, default=180)
    p.add_argument("--lock-timeout", type=int, default=180)
    o = p.parse_args()
    if not o.checkpoint.is_file():
        p.error("Checkpoint does not exist")
    config_file = o.checkpoint.resolve().parent / "run_config.json"
    if not config_file.is_file():
        p.error("Checkpoint must be beside V9 run_config.json")
    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    if cfg.get("version") != VERSION or cfg.get("reward_part_completed") != 10.0:
        p.error("Must evaluate a V9 plain-GAT JobDone+10 run")
    if cfg.get("routing_encoder") != "plain_directed_gat_no_residual_no_edge_features":
        p.error("Wrong encoder; this evaluator does not handle Residual-GAT")
    if cfg.get("graph_nodes") != 105 or cfg.get("graph_edges") != 328:
        p.error("Unexpected saved graph size")
    if not o.seeds or len(o.seeds) != len(set(o.seeds)) or o.sim_seconds <= 0:
        p.error("Require unique seeds and positive duration")
    state_file = o.checkpoint.resolve().parent / "state.json"
    last_episode = 0
    if state_file.is_file():
        last_episode = int(json.loads(state_file.read_text(encoding="utf-8")).get("last_episode",0))
    if any(cfg["seed"] <= seed < cfg["seed"] + last_episode for seed in o.seeds):
        p.error("Do not evaluate on a training seed")
    if o.output.exists():
        p.error("Refusing to overwrite evaluation output")
    model_path = Path(cfg["model_file"])
    if not model_path.is_file() or sha256(model_path) != cfg["model_sha256"]:
        p.error("Missing or changed AnyLogic ZIP")

    a = SimpleNamespace(
        model=str(model_path), port=o.port, sim_seconds=o.sim_seconds,
        max_events=cfg["max_events"], fleetsize=20, max_fleetsize=20,
        single_dispatch_policy=cfg["single_dispatch_policy"],
        routing_backbone="pooled", dispatch_backbone="pooled",
        stop_mask_mode=cfg["stop_mask_mode"], env_type="matrix",
        seed=o.seeds[0], hidden=cfg["hidden"], graph_layers=cfg["graph_layers"],
        edge_heads=cfg["heads"], routing_guide_strength=1.5,
        gamma=cfg["gamma"], gae_lambda=cfg["gae_lambda"], clip=cfg["clip"],
        lr=cfg["lr"], epochs=cfg["epochs"], mini_batches=cfg["mini_batches"],
        device=o.device, startup_timeout=o.startup_timeout,
        lock_timeout=o.lock_timeout, java_exe=o.java_exe,
    )
    import train_mappo_long_B3_v121 as baseline
    original = baseline.SIM_CONFIG
    baseline.SIM_CONFIG = dict(original, reward_part_completed=10.0)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    env = None
    try:
        env = baseline.build_env(a)
        model, graph = v7.v1.build_gnn_model(env, a)
        arch = enable_plain_gat_routing(model, heads=a.edge_heads)
        verify_dispatch_backbone(model)
        if graph.fingerprint != cfg["graph_fingerprint"] or arch != cfg["architecture"]:
            raise RuntimeError("Evaluation graph/architecture differs from training")
        model.load(str(o.checkpoint.resolve()))
        collector = GuidedEventCollector(model, env,
            stop_mask_mode=cfg["stop_mask_mode"], routing_guide_strength=1.5)
        audit = RoutingGraphAudit(model.roles["routing"].actor)
        o.output.parent.mkdir(parents=True, exist_ok=True)
        with o.output.open("x", encoding="utf-8") as f:
            for seed in o.seeds:
                audit.reset()
                _, meta = collector.episode(seed=seed, train=False,
                    max_event_steps=cfg["max_events"])
                if GraphSpec.from_hive(env.agent_hive).fingerprint != graph.fingerprint:
                    raise RuntimeError("Graph changed during held-out evaluation")
                meta["gnn_graph_mapping"] = audit.report()
                meta["reward_part_completed_config"] = 10.0
                f.write(json.dumps({"version":VERSION,"seed":seed,"evaluation":True,
                    "checkpoint":str(o.checkpoint.resolve()),"environment":meta},default=str)+"\n")
                f.flush()
                stats = meta.get("statistics", {})
                print(f"[V9 EVAL] seed={seed} jobDone={stats.get('jobDone',0)} "
                      f"wrong={stats.get('wrongTarget',0)} block={stats.get('block',0)} "
                      f"livelock={stats.get('liveLock',0)} "
                      f"reward={meta.get('delivered_reward',0):.2f} "
                      f"mapping={meta['gnn_graph_mapping']['candidate_map_rate']:.1%}",flush=True)
    finally:
        if env is not None:
            env.close()
        baseline.SIM_CONFIG = original
    print("[V9 EVAL] Saved:",o.output)


if __name__ == "__main__":
    main()
