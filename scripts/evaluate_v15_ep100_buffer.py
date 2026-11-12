#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Read-only EP100 V15 evaluation: Buffer assignment, station congestion, JobDone.

Copy this file into PROJECT_ROOT/scripts/evaluate_v15_ep100_buffer.py.
Requires the original V15 staged training sources, unchanged.

Example (Windows, from PROJECT_ROOT):
  .\.venv311\Scripts\python.exe scripts\evaluate_v15_ep100_buffer.py \
      --stage-dir mappo_runs\YOUR_RUN_NAME\joint16 --seeds 10 --device cuda:0

No source code, reward, model weights, or checkpoint is modified. The file logs
observations at policy decision events, not a continuous AnyLogic trace.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from collections import Counter, defaultdict
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

BUFFER_NODE_IDS = (95, 96, 97, 98, 99)  # Tobias MatrixPart.stat_node['puffer']


def read_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def same_xy(a, b, atol=1e-5):
    return len(a) >= 2 and np.allclose(a[:2], b[:2], atol=atol, rtol=0.0)


def numeric(x, default=0.0):
    try:
        val = float(x)
        return val if math.isfinite(val) else default
    except (ValueError, TypeError):
        return default


class TrafficProbe:
    """Passive policy hook: intercepts decisions, changes nothing."""

    def __init__(self, adapter, env, seed):
        self.inner = adapter
        self.roles = adapter.roles
        self.cfg = adapter.cfg
        self.env = env
        self.seed = seed
        self.total_dispatch = 0
        self.buffer_assignments = 0
        self.buffer_agvs = set()
        self.buffer_assignments_by_node = Counter()
        self.buffer_seen_at_node = set()  # (physical AGV, buffer node)
        self.buffer_next_nonbuffer_dispatch = 0
        self.buffer_next_dispatch_buffer = 0
        self.last_dispatch_was_buffer = {}
        self.same_station_target = 0
        self.station_target_already_occupied = 0
        self.dispatch_station_assignments = 0
        self.station_actions = Counter()
        self.station_occupied_actions = Counter()
        self.station_contested_actions = Counter()
        self.station_target_load_sum = Counter()
        self.station_target_load_samples = Counter()
        self.station_peak_target_load = Counter()
        self.congested_decision_events = 0
        self.decision_events = 0
        self.jobdone_first_buffer = None
        self.jobdone_last = 0.0
        self.trace_rows = []
        self.station_catalog = None
        self.station_xy = None

    def role_for(self, *args, **kwargs):
        return self.inner.role_for(*args, **kwargs)

    def value(self, role, state):
        return self.inner.value(role, state)

    def _build_catalog(self):
        hive = self.env.agent_hive
        stations = list(hive.stations.items())
        # OrderedDict order is exactly the dispatch action-index order.
        self.station_catalog = [int(i) for i, _ in stations]
        self.station_xy = {int(i): np.asarray(xy, dtype=float) for i, xy in stations}
        missing = set(BUFFER_NODE_IDS) - set(self.station_catalog)
        if missing:
            raise RuntimeError(f"Tobias Buffer IDs absent from dispatch stations: {sorted(missing)}. "
                               "Check the actual AnyLogic graph before changing BUFFER_NODE_IDS.")

    def _analyze_agvs(self, state):
        agvs = np.asarray(state["agvs"], dtype=float)
        # The original AnyLogic getObservation() defines:
        # [0]=in_system, [1]=moving, [2:4]=last node, [4:6]=next node,
        # [6:8]=assigned target node, when obs_include_agv_target=True.
        if agvs.ndim != 2 or agvs.shape[1] < 8:
            raise RuntimeError(f"Unexpected physical state AGV tensor shape: {agvs.shape}")
        active = np.flatnonzero(agvs[:, 0] > 0.5)
        targeting = Counter()
        occupied = defaultdict(set)
        for i in active:
            row = agvs[int(i)]
            # DO NOT claim an AGV is 'occupying' a station merely because it
            # is traveling there. Station occupation proxy = nonmoving AGV at
            # next_node == station node (observation-level approximation).
            for node, xy in self.station_xy.items():
                if same_xy(row[6:8], xy):
                    targeting[node] += 1
                if row[1] < 0.5 and same_xy(row[4:6], xy):
                    occupied[node].add(int(i))
                    if node in BUFFER_NODE_IDS:
                        self.buffer_seen_at_node.add((int(i), int(node)))
        return targeting, occupied

    def act(self, role, obs, state, greedy=False):
        if self.station_catalog is None:
            self._build_catalog()
        t = numeric(self.env.get_mappo_time())
        ego = int(state["ego"])
        targeting, occupied = self._analyze_agvs(state)
        self.decision_events += 1
        if any(node not in BUFFER_NODE_IDS and count >= 2
               for node, count in targeting.items()):
            self.congested_decision_events += 1
        for node, load in targeting.items():
            if node not in BUFFER_NODE_IDS:
                self.station_peak_target_load[node] = max(self.station_peak_target_load[node], load)
                self.station_target_load_sum[node] += load
                self.station_target_load_samples[node] += 1

        # Make the unmodified, original model decision FIRST.
        action, logp, value = self.inner.act(role, obs, state, greedy=greedy)
        if role not in ("dispatch_front", "dispatch_rear"):
            return action, logp, value
        action = int(action)
        if action < 0 or action >= len(self.station_catalog):
            raise RuntimeError(f"Dispatch index {action} outside station list ({len(self.station_catalog)})")
        node = self.station_catalog[action]
        is_buffer = node in BUFFER_NODE_IDS
        self.total_dispatch += 1
        if self.last_dispatch_was_buffer.get(ego, False):
            if is_buffer:
                self.buffer_next_dispatch_buffer += 1
            else:
                self.buffer_next_nonbuffer_dispatch += 1
        self.last_dispatch_was_buffer[ego] = is_buffer

        if is_buffer:
            self.buffer_assignments += 1
            self.buffer_agvs.add(ego)
            self.buffer_assignments_by_node[node] += 1
            if self.jobdone_first_buffer is None:
                self.jobdone_first_buffer = numeric((self.env.statistics or {}).get("jobDone"))
        else:
            self.dispatch_station_assignments += 1
            self.station_actions[node] += 1
            # Another AGV physically occupies this destination (proxy).
            busy = any(j != ego for j in occupied[node])
            # Another AGV already has this station as an assigned target.
            other_target_count = targeting[node]
            ego_state = np.asarray(state["agvs"])[ego]
            if same_xy(ego_state[6:8], self.station_xy[node]):
                other_target_count = max(0, other_target_count - 1)
            if busy:
                self.station_target_already_occupied += 1
                self.station_occupied_actions[node] += 1
            if other_target_count > 0:
                self.same_station_target += 1
                self.station_contested_actions[node] += 1

        self.jobdone_last = numeric((self.env.statistics or {}).get("jobDone"))
        self.trace_rows.append({
            "seed": self.seed, "time_s": round(t, 4), "agv": ego,
            "role": role, "dest_node": node,
            "dest_type": "buffer" if is_buffer else "station",
            "other_agvs_targeting_dest": "" if is_buffer else other_target_count,
            "occupied_by_other_proxy": "" if is_buffer else int(busy),
            "jobdone_global_at_decision": self.jobdone_last,
        })
        return action, logp, value

    def summary(self, stats, env_summary, checkpoint_hash):
        final_jobdone = numeric(stats.get("jobDone"))
        return {
            "seed": self.seed,
            "checkpoint_sha256": checkpoint_hash,
            "jobdone": final_jobdone,
            "block": numeric(stats.get("block")),
            "liveLock": numeric(stats.get("liveLock")),
            "targetReached": numeric(stats.get("targetReached")),
            "reward": numeric(env_summary.get("delivered_reward")),
            "dispatch_decisions": self.total_dispatch,
            "buffer_dispatches": self.buffer_assignments,
            "buffer_dispatch_percent": round(100 * self.buffer_assignments / self.total_dispatch, 3) if self.total_dispatch else 0,
            "agvs_ever_dispatched_to_buffer": len(self.buffer_agvs),
            "agv_buffer_node_pairs_seen_stopped": len(self.buffer_seen_at_node),
            "buffer_then_next_dispatch_station": self.buffer_next_nonbuffer_dispatch,
            "buffer_then_next_dispatch_buffer": self.buffer_next_dispatch_buffer,
            "station_dispatches": self.dispatch_station_assignments,
            "station_dispatch_already_targeted_by_another_agv": self.same_station_target,
            "station_dispatch_occupied_by_another_stopped_agv_proxy": self.station_target_already_occupied,
            "station_target_congested_event_pct": round(100 * self.congested_decision_events / self.decision_events, 3) if self.decision_events else 0,
            "jobdone_after_first_buffer_global_temporal_only": (
                final_jobdone - self.jobdone_first_buffer
                if self.jobdone_first_buffer is not None else None
            ),
            "end_reason": env_summary.get("end_reason"),
            "episode_complete": env_summary.get("end_reason") != "max_event_steps",
        }


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def describe(rows, key):
    x = [float(r[key]) for r in rows if r.get(key) is not None]
    if not x:
        return "n/a"
    return f"{mean(x):.3f} +/- {stdev(x) if len(x) >= 2 else 0:.3f} (n={len(x)})"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage-dir", type=Path, required=True,
                   help="Directory containing joint16/experiment.json, state.json and snapshot_ep_0100.pt")
    p.add_argument("--episode", type=int, default=100)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--first-seed", type=int, default=910001)
    p.add_argument("--port", type=int, default=51884)
    p.add_argument("--device", default=None,
                   help="Use training device by default (experiment.json device)")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.seeds < 1 or args.episode < 1:
        p.error("Episode/seeds must be positive")

    stage = args.stage_dir.resolve()
    meta = read_json(stage / "experiment.json")
    state = read_json(stage / "state.json")
    phase = meta["phase"]
    if phase != "joint16":
        raise RuntimeError(f"Expected joint16 (16 AGV), got {phase!r}")
    if int(state["last_episode"]) < args.episode:
        raise RuntimeError(f"Stage only trained to EP{state['last_episode']}")
    checkpoint = (stage / f"snapshot_ep_{args.episode:04d}.pt").resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"EP{args.episode} snapshot missing: {checkpoint}. "
                                "Do not substitute latest checkpoint; that is not EP100.")
    registered = {str(Path(x).resolve()) for x in state.get("snapshots", [])}
    if registered and str(checkpoint) not in registered:
        print("[WARNING] EP snapshot not present in state.json snapshots (check lineage).")
    ckhash = sha256(checkpoint)
    blueprint = Path(meta["blueprint_file"]).resolve()
    simulator = Path(meta["model_file"]).resolve()
    if sha256(blueprint) != meta["blueprint_sha256"]:
        raise RuntimeError("Blueprint changed since training")
    if sha256(simulator) != meta["anylogic_sha256"]:
        raise RuntimeError("AnyLogic model changed since training")
    device = args.device or meta["device"]
    if device != meta["device"]:
        raise RuntimeError("To match V15 evaluation, use training device from experiment.json")
    if "fingerprints" in meta:
        checks = {
            "architecture": ROOT / "thesis/mappo/v15_tobias_arch.py",
            "environment_builder": ROOT / "scripts/v15_tobias_build.py",
            "staged_policy": ROOT / "scripts/v15_staged_collector.py",
            "ppo_algorithm": ROOT / "thesis/mappo/algorithm.py",
            "ppo_stability": ROOT / "thesis/mappo/mappo_stability_v7.py",
            "event_collector": ROOT / "thesis/mappo/collector_stop_ab.py",
            "matrix_environment": ROOT / "thesis/envs/matrix_v121.py",
            "behaviors": ROOT / "thesis/envs/behaviors_v121.py",
            "alpyne_bridge": ROOT / "thesis/envs/alpyne121_bridge.py",
        }
        for name, path in checks.items():
            expected = meta["fingerprints"].get(name)
            if expected and sha256(path) != expected:
                raise RuntimeError(f"{name} source has changed since training: {path}")

    out = (args.output or (stage / f"buffer_eval_ep{args.episode:04d}")).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output already has files; pick an empty --output folder: {out}")
    print("EP", args.episode, "phase", phase, "seeds", args.seeds,
          "checkpoint_sha256", ckhash, "device", device)
    if args.dry_run:
        print("[DRY RUN OK] No AnyLogic simulator started, no files written.")
        return

    import torch
    from v15_tobias_build import build_v15, TOBIAS_DISPATCH_MASK_MODE
    from v15_staged_collector import StagePolicyAdapter, PHASES
    from thesis.mappo.collector_stop_ab import EventCollector

    os.environ["MAPPO_DISPATCH_MASK_MODE"] = TOBIAS_DISPATCH_MASK_MODE
    os.environ["MAPPO_MAX_MINIBATCH"] = str(meta["max_minibatch"])
    torch.set_num_threads(int(meta["cpu_threads"]))
    random.seed(meta["base_seed"])
    np.random.seed(meta["base_seed"])
    torch.manual_seed(meta["base_seed"])
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    env = None
    report = []
    raw_events = []
    per_station = Counter()
    try:
        env, model, graph, arch = build_v15(
            read_json(blueprint), model_path=simulator,
            fleet_size=PHASES[phase][0], port=args.port, device=device,
            seed=meta["base_seed"], max_events=meta["max_events"],
            sim_seconds=3600, checkpoint=checkpoint, phase=phase)
        if not bool(env.config.obs_include_agv_target):
            raise RuntimeError("Cannot measure inbound station load without obs_include_agv_target")
        adapter = StagePolicyAdapter(model, phase, meta["base_seed"])
        for seed in range(args.first_seed, args.first_seed + args.seeds):
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            adapter.rng = np.random.default_rng(seed)
            probe = TrafficProbe(adapter, env, seed)
            collector = EventCollector(probe, env, stop_mask_mode="baseline")
            with torch.no_grad():
                _, env_summary = collector.episode(
                    seed=seed, train=False, max_event_steps=meta["max_events"])
            if env_summary["end_reason"] == "max_event_steps":
                raise RuntimeError(f"Seed {seed} truncated at event limit; don't analyze partial episode")
            record = probe.summary(env_summary.get("statistics", {}), env_summary, ckhash)
            report.append(record)
            raw_events.extend(probe.trace_rows)
            for node, count in probe.buffer_assignments_by_node.items():
                per_station[(seed, node, "buffer_assignment")] += count
            for node, count in probe.station_actions.items():
                per_station[(seed, node, "station_assignment")] += count
            for node, count in probe.station_contested_actions.items():
                per_station[(seed, node, "already_targeted")] += count
            for node, count in probe.station_occupied_actions.items():
                per_station[(seed, node, "occupied_proxy")] += count
            for node, count in probe.station_peak_target_load.items():
                per_station[(seed, node, "peak_target_load")] = count
            print(f"[EP{args.episode} seed={seed}] JobDone={record['jobdone']:.0f} "
                  f"Block={record['block']:.0f} LiveLock={record['liveLock']:.0f} "
                  f"Buffer={record['buffer_dispatches']}/{record['dispatch_decisions']} "
                  f"ToOccupiedProxy={record['station_dispatch_occupied_by_another_stopped_agv_proxy']} "
                  f"ToContested={record['station_dispatch_already_targeted_by_another_agv']}", flush=True)
    finally:
        if env is not None:
            env.close()
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "summary_by_seed.csv", report)
    write_csv(out / "dispatch_event_trace.csv", raw_events)
    write_csv(out / "station_counts_by_seed.csv", [
        {"seed": seed, "node": node, "metric": metric, "count": val}
        for (seed, node, metric), val in sorted(per_station.items())])
    (out / "methodology.json").write_text(json.dumps({
        "phase": phase, "episode": args.episode, "ck_sha256": ckhash,
        "seeds": list(range(args.first_seed, args.first_seed + args.seeds)),
        "sim_seconds": 3600, "greedy": True, "ppo_updates": 0,
        "buffer_node_ids": BUFFER_NODE_IDS,
        "definitions": {
            "buffer_dispatches": "Number of Dispatch policy actions selecting a Tobias buffer node (not proof of arrival)",
            "agv_buffer_node_pairs_seen_stopped": "Distinct AGVs observed stopped at a buffer node during policy events (not guaranteed complete arrival count)",
            "station_dispatch_already_targeted_by_another_agv": "At dispatch time, another AGV has the station as assigned target",
            "station_dispatch_occupied_by_another_stopped_agv_proxy": "Another AGV is stationary and its next-node matches station coordinate; approximate occupation",
            "station_target_congested_event_pct": "Fraction of decision events where >=2 AGVs target same nonbuffer station (event-sampled, not time-weighted)",
            "jobdone_after_first_buffer_global_temporal_only": "Global jobs completed after first buffer assignment; NOT product-level buffer-to-job attribution",
        },
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\nRESULTS (held-out seeds)")
    for key in ["jobdone", "block", "liveLock", "buffer_dispatches", "buffer_dispatch_percent",
                "agvs_ever_dispatched_to_buffer", "agv_buffer_node_pairs_seen_stopped",
                "buffer_then_next_dispatch_station", "station_dispatch_already_targeted_by_another_agv",
                "station_dispatch_occupied_by_another_stopped_agv_proxy",
                "station_target_congested_event_pct"]:
        print(f"  {key:56} {describe(report, key)}")
    xs = np.asarray([v["buffer_dispatch_percent"] for v in report], dtype=float)
    ys = np.asarray([v["jobdone"] for v in report], dtype=float)
    if len(xs) >= 3 and xs.std() > 1e-12 and ys.std() > 1e-12:
        print("  Corr(buffer_dispatch_pct, JobDone) (descriptive):", round(float(np.corrcoef(xs, ys)[0, 1]), 3))
    print("\nSaved:", out)
    print("NOTE: No product IDs are tracked. True buffer->same-product->JobDone requires product-level AnyLogic trace.")

if __name__ == "__main__":
    main()
