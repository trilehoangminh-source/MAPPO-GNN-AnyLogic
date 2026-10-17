
"""Long, single-run training of paper-inspired MAPPO B2.

50 episodes update one continuous model (not 50 independent models).
Only adds a new script and new output files. Safe episode-level checkpoints.
"""
from __future__ import annotations

import argparse
import json
import os
os.environ["MAPPO_DISPATCH_MASK_MODE"] = "operation_first"  # B2 is opt-in; no effect on A/B
import random
import sys
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from thesis.mappo import MAPPO, MAPPOConfig
from thesis.mappo.collector_stop_ab import EventCollector


# Original combined-training environment configuration
SIM_CONFIG = dict(
    dispatch=True,
    routing_ma=True,
    dispatching_ma=True,
    reward_reached_target=10,
    reward_wrong_target=-1,
    reward_removed_for_block=-20,
    reward_target_distance=0,
    reward_invalid=-0.1,
    reward_duration=-0.5,
    reward_pass=0.2,
    block_timeout=60,
    station_separate=False,
    reward_accepted_in_station=0.5,
    reward_declined_in_station=-0.25,
    reward_geo_operation=0.25,
    reward_reduce=-0.005,
    routing_interval=2,
    dispatching_interval=360,
    io_quote=0.9,
    availability=0.9,
    mttr=5,
    fixed_fleets=True,
    remove_all_blocked=False,
)


# Uses alpyne121_bridge.py; no monkeypatch of Alpyne itself.

# ============================================================
# ENVIRONMENT
# ============================================================

def build_env(args):

    # The bridge creates AnyLogicSim with Java 17; no legacy startup monkeypatch.
    os.environ["MAPPO_SERVER_TIMEOUT"] = str(args.startup_timeout)
    os.environ["MAPPO_LOCK_TIMEOUT"] = str(args.lock_timeout)
    if args.java_exe:
        os.environ["MAPPO_JAVA_EXE"] = args.java_exe

    from thesis.envs.matrix_v121 import Matrix

    env = Matrix(
        model_path=str(Path(args.model).resolve()),
        startport=args.port,
        fleetsize=args.fleetsize,
        fleetsize_upper=None,
        max_fleetsize=args.max_fleetsize,
        sim_config=SIM_CONFIG.copy(),
        max_seconds=args.sim_seconds,
        max_steps=args.max_events,
        pseudo_routing=False,
        pseudo_dispatcher=False,
        routing_agent_death=True,
        death_on_target=False,
        transform_dispatching_partobs=True,
        direction_reward=-0.1,
    )
    env._mappo_source_counts = Counter()
    capture_original = env._capture_mappo_state

    def count_station_freshness(alpyne_obs):
        capture_original(alpyne_obs)
        env._mappo_source_counts[env.get_mappo_station_source()] += 1

    env._capture_mappo_state = count_station_freshness
    return env


# ============================================================
# BUILD MAPPO MODEL
# ============================================================

def make_model(env, args):

    from thesis.mappo.algorithm import MAPPO

    cfg = MAPPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip,
        actor_lr=args.lr,
        critic_lr=args.lr,
        epochs=args.epochs,
        mini_batches=args.mini_batches,
        hidden=args.hidden,
        env_type=args.env_type,
        routing_backbone=args.routing_backbone,
        dispatch_backbone=args.dispatch_backbone,
        two_fleets=not args.single_dispatch_policy,
    )

    env.reset(seed=args.seed)

    sample = env.last()[0]

    obs_space = env.observation_space(
        env.agent_selection
    )

    critic_state = env.get_mappo_state()

    n_station = critic_state["stat"].shape[0]
    feat = critic_state["agvs"].shape[1]

    roles = {
        "routing": (obs_space, 5)
    }

    if cfg.two_fleets:
        roles["dispatch_front"] = (
            obs_space,
            n_station
        )
        roles["dispatch_rear"] = (
            obs_space,
            n_station
        )
    else:
        roles["dispatching"] = (
            obs_space,
            n_station
        )

    return MAPPO(
        roles,
        feat,
        cfg,
        device=args.device
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--stop-mask-mode", choices=["baseline", "forbid_when_move"],
        default="forbid_when_move", help="A=baseline; B=mask Stop only when a real move is available",
    )

    parser.add_argument(
        "--model",
        default=str(ROOT / "envs" / "Matrix_critic_fresh.zip")
    )

    parser.add_argument(
        "--env-type",
        default="matrix",
        choices=["matrix", "minimatrix"]
    )

    parser.add_argument(
        "--port",
        type=int,
        default=51550
    )

    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=180
    )

    parser.add_argument("--java-exe", default=r"C:\Program Files\AnyLogic 8.9 Personal Learning Edition\jre\bin\java.exe")
    parser.add_argument("--lock-timeout", type=int, default=180)

    parser.add_argument(
        "--fleetsize",
        type=int,
        default=8
    )

    parser.add_argument(
        "--max-fleetsize",
        type=int,
        default=20
    )

    parser.add_argument(
        "--sim-seconds",
        type=int,
        default=720
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=20000
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=50
    )

    parser.add_argument(
        "--episodes-per-update",
        type=int,
        default=1
    )

    parser.add_argument(
        "--routing-backbone",
        choices=["original_gnn", "pooled"],
        default="pooled"
    )

    parser.add_argument(
        "--dispatch-backbone",
        choices=["original_linear", "pooled"],
        default="pooled"
    )

    parser.add_argument(
        "--single-dispatch-policy",
        action="store_true"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.90
    )

    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95
    )

    parser.add_argument(
        "--clip",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--mini-batches",
        type=int,
        default=1
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--hidden",
        type=int,
        default=32
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=43
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--output",
        default=str(
            ROOT / "mappo_runs" / "long_B_seed43_50ep"
        )
    )

    parser.add_argument(
        "--resume-existing", action="store_true",
        help="Continue a PARTIAL run using checkpoint_ep_XXXX.pt in the same output folder.",
    )
    parser.add_argument(
        "--checkpoint-every", type=int, default=5,
        help="Print checkpoint milestones every N episodes; every episode is saved for safe recovery.",
    )

    args = parser.parse_args()

    if args.fleetsize > args.max_fleetsize:
        parser.error(
            "fleetsize must be <= max-fleetsize"
        )

    if args.startup_timeout < 1:
        parser.error(
            "startup-timeout must be positive"
        )

    if args.episodes_per_update < 1:
        parser.error(
            "episodes-per-update must be positive"
        )

    if args.episodes < 1 or args.sim_seconds < 1:
        parser.error("--episodes and --sim-seconds must be positive")
    if args.episodes_per_update != 1:
        parser.error("Safe episode-level checkpoints require --episodes-per-update 1")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if args.stop_mask_mode != "forbid_when_move":
        parser.error("This trainer is Stop-mask B only; use original long trainer for A.")
    if args.routing_backbone != "pooled" or args.dispatch_backbone != "pooled":
        parser.error("This validated long-run setup uses pooled actor backbones only.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output).resolve()
    metrics = out / "metrics.jsonl"
    manifest_path = out / "run_config.json"
    settings = {
        "version": "long_B2_operation_first_v1", "dispatch_mask_mode": "operation_first", "seed": args.seed,
        "episodes": args.episodes, "sim_seconds": args.sim_seconds,
        "fleetsize": args.fleetsize, "max_fleetsize": args.max_fleetsize,
        "stop_mask_mode": args.stop_mask_mode,
        "routing_backbone": args.routing_backbone,
        "dispatch_backbone": args.dispatch_backbone,
        "single_dispatch_policy": args.single_dispatch_policy,
        "max_events": args.max_events, "gamma": args.gamma,
        "gae_lambda": args.gae_lambda, "clip": args.clip,
        "lr": args.lr, "epochs": args.epochs,
        "mini_batches": args.mini_batches, "hidden": args.hidden,
        "model_file": str(Path(args.model).resolve()),
    }
    if not Path(args.model).is_file():
        parser.error(f"Missing model ZIP: {args.model}")
    out.mkdir(parents=True, exist_ok=True)
    start_episode = 0
    if args.resume_existing:
        if not manifest_path.is_file():
            parser.error("Cannot resume: run_config.json is missing")
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous != settings:
                parser.error("Run settings differ from run_config.json. Use the exact original command.")
            rows = ([json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
                    if metrics.is_file() else [])
        except (ValueError, OSError) as exc:
            parser.error(f"Cannot inspect existing run: {exc}")
        if not all(row.get("episode") == i for i, row in enumerate(rows, 1)):
            parser.error("Non-sequential metrics.jsonl; check interrupted output manually")
        start_episode = len(rows)
        if start_episode > args.episodes:
            parser.error("Existing log has more episodes than requested")
        if start_episode and not (out / f"checkpoint_ep_{start_episode:04d}.pt").is_file():
            parser.error("Last completed episode checkpoint missing; do not silently restart")
        if start_episode == args.episodes:
            print(f"[DONE] Already completed {args.episodes} episodes: {out}", flush=True)
            return
    else:
        if metrics.exists() or manifest_path.exists() or any(out.iterdir()):
            parser.error(f"Output is nonempty: {out}. Choose a NEW --output or use --resume-existing.")
        manifest_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    env = None
    try:
        print("[LONG] Building AnyLogic environment / MAPPO networks...", flush=True)
        env = build_env(args)
        model = make_model(env, args)
        if start_episode:
            ckpt = out / f"checkpoint_ep_{start_episode:04d}.pt"
            model.load(str(ckpt))
            print(f"[RESUME] Weights + optimizer restored from {ckpt.name}; "
                  "random-number state is NOT restored bit-for-bit.", flush=True)
        collector = EventCollector(model, env, stop_mask_mode=args.stop_mask_mode)
        print(f"[LONG] B2 operation_first dispatch; training episodes {start_episode+1}..{args.episodes}; "
              f"sim_seconds={args.sim_seconds}; episode_seed={args.seed}+i-1", flush=True)
        # Each episode receives its own exact seed but all policy updates affect
        # the SAME MAPPO model (unlike separate A/B multiseed experiments).
        with metrics.open("a", encoding="utf-8") as logfile:
            for e in range(start_episode, args.episodes):
                before_sources = env._mappo_source_counts.copy()
                pieces, meta = collector.episode(
                    seed=args.seed + e, train=True, max_event_steps=args.max_events,
                )
                meta["station_state_sources"] = {
                    k: int(env._mappo_source_counts[k] - before_sources[k])
                    for k in sorted(env._mappo_source_counts)
                }
                update = model.train(pieces)

                # Commit a distinct policy checkpoint for every episode.
                # On crashes, resume uses the last episode with BOTH a JSONL
                # entry and its corresponding checkpoint.
                ckpt = out / f"checkpoint_ep_{e+1:04d}.pt"
                tmp = ckpt.with_suffix(".pt.tmp")
                model.save(str(tmp))
                os.replace(tmp, ckpt)
                record = {
                    "episode": e + 1,
                    "environment_seed": args.seed + e,
                    "environment": meta,
                    "policy_updates": update,
                    "experiment": {
                        "stop_mask_mode": args.stop_mask_mode, "seed": args.seed,
                        "sim_seconds": args.sim_seconds,
                        "routing_backbone": args.routing_backbone,
                        "dispatch_backbone": args.dispatch_backbone,
                    },
                }
                logfile.write(json.dumps(record, default=str) + "\n")
                logfile.flush()
                # Convenience latest filename. The immutable ep checkpoint is
                # authoritative for recovery even if the copy is interrupted.
                shutil.copy2(ckpt, out / "latest_mappo.pt")
                kpi = meta.get("statistics", {})
                print(f"[EP {e+1:03d}/{args.episodes}] "
                      f"reward={meta.get('delivered_reward', 0.0):.2f} "
                      f"targetReached={kpi.get('targetReached', 0)} "
                      f"jobDone={kpi.get('jobDone', 0)} "
                      f"wrongTarget={kpi.get('wrongTarget', 0)} "
                      f"invalid={kpi.get('taken_invalid', 0)} "
                      f"events={meta.get('events', 0)} "
                      f"fresh_routing={meta['station_state_sources'].get('fresh_routing', 0)} "
                      f"updates={','.join(update)}", flush=True)
                if (e + 1) % args.checkpoint_every == 0:
                    print(f"[CHECKPOINT] episode={e+1}; {ckpt.name}", flush=True)
        print(f"[DONE] {args.episodes} episodes complete. Metrics: {metrics}", flush=True)
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
