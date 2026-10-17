"""Aggregate Stop A/B MAPPO results by paired seed, not by individual events.

Outputs paired_seed_summary.csv and aggregate_summary.csv.
Supports legacy seed 43 runs via --include-seed43.
Requires no extra packages.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent


def load_run(path: Path, expected_seed: int, expected_mode: str):
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics: {path}")
    data = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not data:
        raise ValueError(f"Empty metrics: {path}")
    for i, row in enumerate(data, 1):
        exp = row.get("experiment", {})
        if (exp.get("seed") != expected_seed or exp.get("stop_mask_mode") != expected_mode
                or row.get("episode") != i):
            raise ValueError(f"Inconsistent experiment metadata: {path} episode {i}")
    return data


def get_stat(r, key):
    return float(r["environment"].get("statistics", {}).get(key, 0.0))


def get_diag(r, key):
    return float(r["environment"].get("action_diagnostics", {}).get(key, 0.0))


def stop_percentage(r):
    raw = r["environment"].get("action_diagnostics", {}).get("actions_by_role", {}).get("routing", {})
    total = sum(int(x) for x in raw.values())
    return 100 * int(raw.get("0", 0)) / total if total else 0.0


METRICS = {
    "reward": lambda r: float(r["environment"]["delivered_reward"]),
    "jobDone": lambda r: get_stat(r, "jobDone"),
    "targetReached": lambda r: get_stat(r, "targetReached"),
    "wrongTarget": lambda r: get_stat(r, "wrongTarget"),
    "taken_invalid": lambda r: get_stat(r, "taken_invalid"),
    "block": lambda r: get_stat(r, "block"),
    "liveLock": lambda r: get_stat(r, "liveLock"),
    "routing_action0_pct": stop_percentage,
    "events": lambda r: float(r["environment"].get("events", 0)),
    "fresh_routing": lambda r: float(r["environment"].get("station_state_sources", {}).get("fresh_routing", 0)),
    "masked_decisions": lambda r: get_diag(r, "routing_stop_masked_decisions"),
    "mask_violations": lambda r: get_diag(r, "selected_mask_violations"),
    "routing_updates_transitions": lambda r: float(r.get("policy_updates", {}).get("routing", {}).get("transitions", 0)),
    "dispatch_front_updates_transitions": lambda r: float(r.get("policy_updates", {}).get("dispatch_front", {}).get("transitions", 0)),
    "dispatch_rear_updates_transitions": lambda r: float(r.get("policy_updates", {}).get("dispatch_rear", {}).get("transitions", 0)),
}
# +1 when HIGHER is better; -1 when LOWER is better; 0 diagnostic only.
DIRECTION = {"reward": 1, "jobDone": 1, "targetReached": 1, "wrongTarget": -1,
             "taken_invalid": -1, "block": -1, "liveLock": -1,
             "routing_action0_pct": 0, "events": 0, "fresh_routing": 0,
             "masked_decisions": 0, "mask_violations": -1,
             "routing_updates_transitions": 0, "dispatch_front_updates_transitions": 0,
             "dispatch_rear_updates_transitions": 0}


def write_csv(path: Path, header: list[str], data: list[dict]):
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        writer.writerows(data)


def main(argv=None):
    p = argparse.ArgumentParser(description="Compare seed-paired MAPPO Stop A/B runs")
    p.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46])
    p.add_argument("--root", type=Path, default=Path("mappo_runs/stop_ab_multiseed"))
    p.add_argument("--include-seed43", action="store_true", help="Include existing legacy seed 43 A/B runs")
    args = p.parse_args(argv)
    seeds = ([43] if args.include_seed43 else []) + args.seeds
    if len(seeds) != len(set(seeds)):
        p.error("seeds must be unique (do not include seed 43 twice)")
    root = args.root if args.root.is_absolute() else ROOT / args.root
    paired = []
    per_seed = []
    target_episodes = None
    target_seconds = None
    for seed in seeds:
        if seed == 43 and args.include_seed43:
            paths = (ROOT / "mappo_runs" / "stop_ab_A_seed43" / "metrics.jsonl",
                     ROOT / "mappo_runs" / "stop_ab_B_seed43" / "metrics.jsonl")
        else:
            paths = (root / f"seed_{seed}" / "A" / "metrics.jsonl",
                     root / f"seed_{seed}" / "B" / "metrics.jsonl")
        a = load_run(paths[0], seed, "baseline")
        b = load_run(paths[1], seed, "forbid_when_move")
        if len(a) != len(b):
            raise ValueError(f"A/B episode count mismatch for seed={seed}: {len(a)} / {len(b)}")
        seconds = (a[0]["experiment"].get("sim_seconds"), b[0]["experiment"].get("sim_seconds"))
        if seconds[0] != seconds[1] or (target_seconds is not None and target_seconds != seconds[0]):
            raise ValueError(f"simulation length mismatch at seed {seed}: {seconds}")
        if target_episodes is not None and target_episodes != len(a):
            raise ValueError(f"episode count mismatch at seed {seed}")
        target_episodes, target_seconds = len(a), seconds[0]
        if any(r["experiment"].get("routing_backbone") != "pooled" or
               r["experiment"].get("dispatch_backbone") != "pooled" for r in a + b):
            raise ValueError(f"Backbone mismatch at seed {seed}")
        # Mean within seed, then aggregate across seeds so episodes aren't treated as independent trials.
        aa = {key: mean(fun(r) for r in a) for key, fun in METRICS.items()}
        bb = {key: mean(fun(r) for r in b) for key, fun in METRICS.items()}
        for key in METRICS:
            record = {"seed": seed, "metric": key, "A": aa[key], "B": bb[key], "B_minus_A": bb[key] - aa[key]}
            paired.append(record)
        row = {"seed": seed}
        for key in ("reward", "jobDone", "targetReached", "wrongTarget", "taken_invalid", "block", "liveLock"):
            row[key + "_A"] = round(aa[key], 3)
            row[key + "_B"] = round(bb[key], 3)
            row[key + "_delta"] = round(bb[key] - aa[key], 3)
        per_seed.append(row)
    root.mkdir(parents=True, exist_ok=True)
    columns = ["seed"] + [f"{metric}_{suffix}" for metric in
                           ("reward", "jobDone", "targetReached", "wrongTarget", "taken_invalid", "block", "liveLock")
                           for suffix in ("A", "B", "delta")]
    write_csv(root / "paired_seed_summary.csv", columns, per_seed)
    aggregate = []
    print(f"[AB] Paired seeds: {seeds}; {target_episodes} episodes/arm/seed; {target_seconds}s/episode")
    print(f"{'Metric':35} {'A mean':>12} {'B mean':>12} {'B - A':>12} {'B better':>10}")
    print("-" * 86)
    for key in METRICS:
        records = [r for r in paired if r["metric"] == key]
        am = mean(r["A"] for r in records)
        bm = mean(r["B"] for r in records)
        d = mean(r["B_minus_A"] for r in records)
        sample_sd = stdev(r["B_minus_A"] for r in records) if len(records) >= 2 else float("nan")
        better = sum(DIRECTION[key] * r["B_minus_A"] > 0 for r in records) if DIRECTION[key] else None
        same = sum(r["B_minus_A"] == 0 for r in records)
        aggregate.append({"metric": key, "A_mean": am, "B_mean": bm, "delta": d,
                          "paired_delta_sd": sample_sd, "B_better_seeds": better,
                          "ties": same, "seeds": len(records)})
        win = f"{better}/{len(records)}" if better is not None else "--"
        print(f"{key:35} {am:12.2f} {bm:12.2f} {d:12.2f} {win:>10}")
    write_csv(root / "aggregate_summary.csv", list(aggregate[0]), aggregate)
    print("\n[FILES] " + str(root / "paired_seed_summary.csv"))
    print("[FILES] " + str(root / "aggregate_summary.csv"))
    print("[NOTE] Seed-level descriptive comparison only; no significance claim. "
          "B removes Stop by design; fewer invalid actions alone do not prove better logistics performance.")


if __name__ == "__main__":
    main()
