"""Fresh 20-AGV MAPPO + plain GAT + JobDone +10, wall-clock limited training.

Run from repository ROOT. Never loads V7/V8 checkpoints. Saves one complete
checkpoint each episode, alternating two rolling slots; resumption is explicit.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
os.environ["MAPPO_DISPATCH_MASK_MODE"] = "operation_first"

import numpy as np
import torch
import train_B3_MAPPO_v7 as v7
from thesis.mappo.plain_gat_routing_v9 import enable_plain_gat_routing
from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
from thesis.mappo.gnn_audit_uoe_v2 import (
    RoutingGraphAudit, verify_ppo_rollout_parity, verify_dispatch_backbone
)
from thesis.mappo.routing_guidance_v121 import GuidedEventCollector
from thesis.mappo.mappo_stability_v7 import V7Options, enable_v7, audit_rollout_roles

VERSION = "MAPPO_V9_PLAIN_GAT_JOBDONE10_FRESH"


def atomic_json(path, data):
    path = Path(path)
    temp = Path(str(path) + ".tmp")
    temp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    os.replace(temp, path)


def atomic_model_save(model, dest):
    temp = Path(str(dest) + ".tmp")
    model.save(str(temp))
    os.replace(temp, dest)


def sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_args():
    parser = v7.args_parser()
    parser.description = __doc__
    parser.set_defaults(
        model=str(ROOT / "envs" / "Matrix_RL_debug.zip"),
        output=str(ROOT / "mappo_runs" / "V9_PLAIN_GAT_JOBDONE10_FRESH20_5H"),
        episodes=100000, sim_seconds=1800, max_events=20000,
        fleetsize=20, max_fleetsize=20, seed=43,
        device="cuda:0", hidden=32, graph_layers=2, edge_heads=2,
        gamma=0.9, gae_lambda=0.95, clip=0.1, lr=0.0001,
        epochs=5, mini_batches=1, routing_guide_strength=1.5,
        guide_final=1.5, guide_decay_episodes=1,
        ppo_v7_mode="stable", v7_value_loss="clipped_huber",
        v7_value_clip=0.2, v7_huber_delta=1.0, v7_target_kl=0.03,
        v7_time_unit_seconds=0.0, stop_mask_mode="forbid_when_move",
    )
    parser.add_argument("--hours", type=float, default=5.0)
    parser.add_argument("--snapshot-every", type=int, default=10)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--reward-jobdone", type=float, default=10.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    a = parser.parse_args()
    if not 0 < a.hours <= 72 or not 1 <= a.cpu_threads <= 32:
        parser.error("--hours must be (0,72], --cpu-threads 1..32")
    if a.snapshot_every < 1 or a.episodes < 1:
        parser.error("--snapshot-every and --episodes must be positive")
    if a.fleetsize != 20 or a.max_fleetsize != 20:
        parser.error("V9 experiment is fixed to 20 AGVs")
    if a.reward_jobdone != 10.0:
        parser.error("V9 experiment requires --reward-jobdone 10")
    if not a.model or not Path(a.model).is_file():
        parser.error("Missing AnyLogic ZIP: " + str(a.model))
    if a.hidden % a.edge_heads or a.graph_layers < 1:
        parser.error("GAT head count must divide hidden; layers>=1")
    if a.guide_final != 1.5 or a.routing_guide_strength != 1.5 or a.guide_decay_episodes != 1:
        parser.error("This experiment holds the Dijkstra strength fixed at 1.5")
    if a.ppo_v7_mode != "stable":
        parser.error("This experiment requires stable V7 PPO")
    return a


def main():
    a = get_args()
    out = Path(a.output).resolve()
    model_path = Path(a.model).resolve()
    import train_mappo_long_B3_v121 as baseline
    original_config = baseline.SIM_CONFIG
    # Isolated Reward V9; do not touch V8 reward or change the source file.
    baseline.SIM_CONFIG = dict(original_config, reward_part_completed=10.0)
    assert baseline.SIM_CONFIG["reward_part_completed"] == 10.0

    opts = V7Options(
        value_loss=a.v7_value_loss,
        value_clip_epsilon=a.v7_value_clip,
        huber_delta=a.v7_huber_delta,
        target_kl=a.v7_target_kl,
        time_unit_seconds=a.v7_time_unit_seconds,
    )
    settings = {
        "version": VERSION, "model_file": str(model_path),
        "model_sha256": sha256(model_path), "seed": a.seed,
        "sim_seconds": a.sim_seconds, "max_events": a.max_events,
        "fleetsize": a.fleetsize, "max_fleetsize": a.max_fleetsize,
        "hidden": a.hidden, "graph_layers": a.graph_layers, "heads": a.edge_heads,
        "gamma": a.gamma, "gae_lambda": a.gae_lambda, "clip": a.clip,
        "lr": a.lr, "epochs": a.epochs, "mini_batches": a.mini_batches,
        "guide_start": 1.5, "guide_final": 1.5,
        "guide_decay_episodes": 1, "stop_mask_mode": a.stop_mask_mode,
        "single_dispatch_policy": a.single_dispatch_policy,
        "ppo_v7_mode": "stable", "ppo_v7_options": asdict(opts),
        "reward_part_completed": 10.0,
        "routing_encoder": "plain_directed_gat_no_residual_no_edge_features",
        "dispatch_backbone": "pooled",
        "dispatch_mask_mode": "operation_first",
    }

    if a.dry_run:
        print("[DRY RUN] FRESH MAPPO V9: EP1, no prior checkpoint")
        print("[DRY RUN] Node-only GAT; NO residual; no edge attributes")
        print("[DRY RUN] JobDone reward=+10, 20 AGVs, 5h wall clock")
        print("[DRY RUN] AnyLogic:", model_path)
        print("[DRY RUN] Output:", out, "resume=", a.resume)
        if a.device.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; install CUDA-enabled PyTorch or use --device cpu")
            print("[DRY RUN] GPU:", torch.cuda.get_device_name(torch.device(a.device)))
        return

    cfg_path = out / "run_config.json"
    state_path = out / "state.json"
    metrics_path = out / "metrics.jsonl"
    if a.resume:
        if not cfg_path.is_file() or not state_path.is_file():
            raise RuntimeError("Resume requires existing V9 run_config.json and state.json")
        previous = json.loads(cfg_path.read_text(encoding="utf-8"))
        # Compare only immutable configuration; hours and max episodes can change.
        for key, value in settings.items():
            if key not in ("graph_fingerprint", "graph_nodes", "graph_edges", "architecture") and previous.get(key) != value:
                raise RuntimeError("Resume mismatch: " + key)
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not 1 <= int(state.get("last_episode", -1)):
            raise RuntimeError("No committed episode to resume")
        ckpt = Path(state["checkpoint"])
        if ckpt.parent != out or ckpt.name not in ("rolling_slot_0.pt", "rolling_slot_1.pt") or not ckpt.is_file():
            raise RuntimeError("Missing/foreign rolling checkpoint")
        if not metrics_path.exists():
            raise RuntimeError("Missing committed metrics")
        rows = [json.loads(x) for x in metrics_path.read_text(encoding="utf-8").splitlines() if x.strip()]
        committed = [r for r in rows if int(r["episode"]) <= int(state["last_episode"])]
        ep_nums = [int(r["episode"]) for r in committed]
        if not ep_nums or ep_nums != list(range(1, state["last_episode"] + 1)):
            raise RuntimeError("Missing/duplicate episodes in metrics; inspect before resume")
        if len(committed) < len(rows):
            metrics_path.write_text("".join(json.dumps(r) + "\n" for r in committed), encoding="utf-8")
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError("Refusing nonempty output; new run required: " + str(out))
        out.mkdir(parents=True, exist_ok=True)
        state = {"last_episode": 0, "checkpoint": "", "snapshots": [], "run_seconds": 0.0}
        atomic_json(state_path, state)
    torch.set_num_threads(a.cpu_threads)
    os.environ["OMP_NUM_THREADS"] = str(a.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(a.cpu_threads)
    if a.device.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; install CUDA PyTorch or pass --device cpu")
        print("[V9 GPU]", a.device, torch.cuda.get_device_name(torch.device(a.device)))
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    env = None
    start = time.monotonic()
    last_duration = 0.0
    try:
        env = baseline.build_env(a)
        # B3 builds the actor fresh (random init); this replaces ONLY Routing MP.
        model, spec = v7.v1.build_gnn_model(env, a)
        arch = enable_plain_gat_routing(model, heads=a.edge_heads)
        verify_dispatch_backbone(model)
        enable_v7(model, opts)
        settings.update({"graph_fingerprint": spec.fingerprint,
                         "graph_nodes": spec.n, "graph_edges": len(spec.arcs),
                         "architecture": arch})
        if a.resume:
            for key in ("graph_fingerprint", "graph_nodes", "graph_edges", "architecture"):
                if previous.get(key) != settings[key]:
                    raise RuntimeError("Graph/architecture changed: " + key)
            model.load(str(ckpt))
            print("[V9] RESUMED after episode", state["last_episode"])
        else:
            atomic_json(cfg_path, settings)
            print("[V9] FRESH model, EP1; no checkpoint load")
        auditor = RoutingGraphAudit(model.roles["routing"].actor, sample_limit=a.diagnostic_samples)
        collector = GuidedEventCollector(model, env,
            stop_mask_mode=a.stop_mask_mode, routing_guide_strength=1.5)
        while int(state["last_episode"]) < a.episodes:
            elapsed = time.monotonic() - start
            if elapsed + max(last_duration, 0.0) >= a.hours * 3600:
                print("[V9] Wall-clock limit reached between complete episodes")
                break
            ep = int(state["last_episode"]) + 1
            ts = time.monotonic()
            random.seed(a.seed + ep)
            np.random.seed(a.seed + ep)
            torch.manual_seed(a.seed + ep)
            auditor.reset()
            trajectories, meta = collector.episode(
                seed=a.seed + ep - 1, train=True, max_event_steps=a.max_events
            )
            if GraphSpec.from_hive(env.agent_hive).fingerprint != spec.fingerprint:
                raise RuntimeError("Graph topology changed: no PPO update")
            report = auditor.report()
            meta["gnn_graph_mapping"] = report
            meta["dijkstra_strength_this_episode"] = 1.5
            meta["reward_part_completed_config"] = 10.0
            if report["candidate_map_rate"] < a.min_candidate_map_rate:
                raise RuntimeError("Candidate map rate failed: " + str(report))
            meta["ppo_preupdate_audit"] = verify_ppo_rollout_parity(
                model, trajectories, max_samples=a.parity_samples
            )
            meta["ppo_all_roles_preupdate_audit"] = audit_rollout_roles(
                model, trajectories, max_per_role=a.parity_samples
            )
            updates = model.train(trajectories)
            slot = out / f"rolling_slot_{ep % 2}.pt"
            atomic_model_save(model, slot)
            item = {"version": VERSION, "episode": ep,
                    "environment_seed": a.seed + ep - 1,
                    "environment": meta, "policy_updates": updates}
            with metrics_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(item, default=str) + "\n")
                file.flush()
                os.fsync(file.fileno())
            state["last_episode"] = ep
            state["checkpoint"] = str(slot)
            state["run_seconds"] += time.monotonic() - ts
            if ep % a.snapshot_every == 0:
                snap = out / f"snapshot_ep_{ep:04d}.pt"
                shutil.copy2(slot, Path(str(snap) + ".tmp"))
                os.replace(str(snap) + ".tmp", snap)
                state["snapshots"].append(str(snap))
            atomic_json(state_path, state)
            # Latest is a convenience copy only. state.json + rolling slot is canonical.
            shutil.copy2(slot, out / "latest_mappo.pt.tmp")
            os.replace(out / "latest_mappo.pt.tmp", out / "latest_mappo.pt")
            st = meta.get("statistics", {})
            last_duration = time.monotonic() - ts
            print(f"[V9 EP {ep}] jobDone={st.get('jobDone',0)} "
                  f"wrongTarget={st.get('wrongTarget',0)} "
                  f"block={st.get('block',0)} liveLock={st.get('liveLock',0)} "
                  f"reward={meta.get('delivered_reward',0):.2f} "
                  f"map={report['candidate_map_rate']:.1%} "
                  f"elapsed={(time.monotonic()-start)/60:.1f}min", flush=True)
    except KeyboardInterrupt:
        print("[V9] Interrupted; committed episode", state["last_episode"], flush=True)
    finally:
        if env is not None:
            env.close()
        baseline.SIM_CONFIG = original_config
    print("[V9] Last completed episode:", state["last_episode"])
    print("[V9] Checkpoint:", state["checkpoint"])
    print("[V9] Output:", out)


if __name__ == "__main__":
    main()
