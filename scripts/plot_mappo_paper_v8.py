#!/usr/bin/env python3
"""Plot MAPPO V8 real logs; no re-training or checkpoint loading.

Run from thesis repository root:
python scripts/plot_mappo_paper_v8.py --folder mappo_runs/V8_15AGV_fromV7_5h
"""
import argparse
import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Plot in VS Code/headless without opening a window
import matplotlib.pyplot as plt

METRICS = [
    ("reward", "Episode Reward", "Reward"),
    ("targetReached", "Targets Reached", "Targets Reached"),
    ("block", "Number of Blocks", "Blocks"),
    ("jobDone", "Products Completed", "Products"),
    ("wrongTarget", "Wrong Targets", "Wrong Targets"),
    ("liveLock", "Livelock", "Livelock"),
]

ALIASES = {
    "episode": ("episode", "ep", "training_episode", "episode_number", "episode_idx"),
    "seed": ("seed", "evaluation_seed", "environment_seed", "env_seed"),
    "reward": ("delivered_reward", "episode_reward", "total_reward", "reward", "return"),
    "targetReached": ("targetReached", "targetsReached", "targets_reached", "target_reached"),
    "block": ("block", "blocks", "blocked"),
    "jobDone": ("jobDone", "job_done", "jobs_completed", "products_completed"),
    "wrongTarget": ("wrongTarget", "wrong_target", "wrong_targets"),
    "liveLock": ("liveLock", "livelock", "live_lock"),
}

# Preferred locations reflect actual JSON examples from MAPPO/Alpyne logs.
PARENTS = {
    "episode": ((), ("training",), ("metadata",)),
    "seed": ((), ("environment",), ("evaluation",)),
    "reward": (("environment",), (), ("environment", "statistics"), ("statistics",)),
}
STAT_PARENTS = (
    ("environment", "statistics"), ("statistics",), ("environment", "stats"),
    ("stats",), ("environment",), (), ("metrics",), ("results",),
)


def nested(obj, path):
    for piece in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(piece)
    return obj


def as_number(value):
    if isinstance(value, bool) or isinstance(value, (dict, list)) or value is None:
        return None
    try:
        number = float(value)
        if math.isfinite(number):
            return number
    except (TypeError, ValueError):
        pass
    return None


def find_number(obj, name):
    parents = PARENTS.get(name, STAT_PARENTS)
    for parent in parents:
        node = nested(obj, parent)
        if isinstance(node, dict):
            for alias in ALIASES[name]:
                number = as_number(node.get(alias))
                if number is not None:
                    return number
    return None


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[WARN] {path.name}:{line_no}: invalid JSON: {exc}")
                continue
            if not isinstance(record, dict):
                print(f"[WARN] {path.name}:{line_no}: JSON is not an object; skipped")
                continue
            records.append(record)
    return records


def normalize(record, row_index):
    result = {name: find_number(record, name) for name in ALIASES}
    # We do NOT invent episode numbers: use a row index only to label a missing value.
    if result["episode"] is None:
        result["episode"] = float(row_index)
        result["episode_inferred"] = True
    else:
        result["episode_inferred"] = False
    return result


def rolling_mean(values, window):
    values = list(values)
    result = []
    for idx in range(len(values)):
        block = [val for val in values[max(0, idx - window + 1):idx + 1] if val is not None]
        result.append(statistics.mean(block) if block else None)
    return result


def plotted_xy(rows, name):
    pairs = [(r["episode"], r[name]) for r in rows if r[name] is not None]
    pairs.sort(key=lambda item: item[0])
    return [p[0] for p in pairs], [p[1] for p in pairs]


def style_ax(ax):
    ax.grid(axis="y", color="#bbbbbb", alpha=0.48, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_color("#888888")
    ax.spines["right"].set_color("#888888")
    ax.tick_params(axis="both", labelsize=9)


def save_figure(fig, target_stem):
    target_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_stem.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor="white")
    fig.savefig(target_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVED] {target_stem.with_suffix('.png')}")
    print(f"[SAVED] {target_stem.with_suffix('.pdf')}")


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_training(folder, output_dir, smooth):
    file = folder / "metrics.jsonl"
    if not file.is_file():
        print(f"[WARN] Training file missing: {file}")
        return False
    raw = read_jsonl(file)
    rows = [normalize(r, i + 1) for i, r in enumerate(raw)]
    if not rows:
        print("[WARN] Training metrics.jsonl empty/invalid")
        return False
    rows.sort(key=lambda item: item["episode"])
    present = [key for key, _, _ in METRICS if any(r[key] is not None for r in rows)]
    if not present:
        print("[WARN] Training file has no recognizable KPI fields; first-row top-level keys:")
        print("       ", list(raw[0].keys()))
        if isinstance(raw[0].get("environment"), dict):
            print("        environment keys:", list(raw[0]["environment"].keys()))
        return False
    print(f"[TRAIN] {len(rows)} rows, episodes {rows[0]['episode']:g}..{rows[-1]['episode']:g}; metrics found: {present}")
    if any(r["episode_inferred"] for r in rows):
        print("[WARN] Missing episode field in some rows; row position was used instead. Verify X axis.")

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.4), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes.ravel(), METRICS):
        x, y = plotted_xy(rows, key)
        if x:
            ax.plot(x, y, linewidth=0.7, alpha=0.27, color="#a03048", label="Episode raw")
            smooth_y = rolling_mean(y, smooth)
            ax.plot(x, smooth_y, linewidth=1.8, color="#b52146", label=f"Rolling mean ({smooth})")
            ax.legend(frameon=True, fontsize=8, loc="best")
        else:
            ax.text(0.5, 0.5, "Not logged", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training Episode")
        ax.set_ylabel(ylabel)
        style_ax(ax)
    fig.suptitle(f"MAPPO learning curves — {folder.name} (training episodes)", fontsize=14)
    save_figure(fig, output_dir / "Figure_Training_6panels")
    fields = ["episode", "episode_inferred", "reward", "targetReached", "block", "jobDone", "wrongTarget", "liveLock", "seed"]
    write_csv(output_dir / "normalized_training.csv", rows, fields)
    return True


def plot_evaluations(folder, output_dir):
    candidates = [
        ("V7 source on 15 AGVs", folder / "eval_source_v7_evaluated_same_fleet.jsonl"),
        ("V8 episode 96", folder / "eval_snapshot_ep_0096.jsonl"),
        ("V8 episode 144", folder / "eval_latest_ep_0144.jsonl"),
    ]
    colors = ("#b52146", "#d9a000", "#67429b")
    data = []
    exported = []
    for label, path in candidates:
        if not path.is_file():
            print(f"[WARN] Evaluation file missing: {path}")
            continue
        raw = read_jsonl(path)
        rows = [normalize(r, i + 1) for i, r in enumerate(raw)]
        if not rows:
            continue
        for r in rows:
            exported.append({"candidate": label, **r})
        seeds = [int(r["seed"]) for r in rows if r["seed"] is not None]
        print(f"[EVAL] {label}: {len(rows)} rows; seeds={seeds or 'not logged'}")
        data.append((label, rows, colors[candidates.index((label, path))]))

    if not data:
        print("[WARN] No evaluation data; skipping evaluation figure")
        return False

    seed_sets = [set(r["seed"] for r in rows if r["seed"] is not None) for _, rows, _ in data]
    if len(seed_sets) > 1 and all(seed_sets) and len({tuple(sorted(s)) for s in seed_sets}) > 1:
        print("[CAUTION] Different evaluation seeds between candidates; comparisons are not matched!")

    fig, axes = plt.subplots(3, 2, figsize=(14.5, 13), constrained_layout=True)
    for ax, (key, title, ylabel) in zip(axes.ravel(), METRICS):
        for j, (label, rows, color) in enumerate(data):
            values = [r[key] for r in rows if r[key] is not None]
            if not values:
                continue
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) >= 2 else 0.0
            ax.bar(j, mean, width=0.65, color=color, alpha=0.78)
            # This is SD across evaluation seeds, not a confidence interval.
            if len(values) >= 2:
                ax.errorbar(j, mean, yerr=sd, fmt="none", ecolor="#202020", capsize=4, linewidth=1.1)
            for idx, v in enumerate(values):
                offset = (idx - (len(values) - 1) / 2) * 0.075
                ax.scatter(j + offset, v, s=20, color="#101010", alpha=0.7, zorder=4)
            ax.annotate(f"{mean:.1f}\nn={len(values)}", (j, mean), xytext=(0, 4),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(data)), [label.replace(" ", "\n", 1) for label, _, _ in data], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        style_ax(ax)
    fig.suptitle("Evaluation candidates — check equal fleetsize, runtime and matched seeds", fontsize=13)
    save_figure(fig, output_dir / "Figure_Evaluation_6panels")
    write_csv(output_dir / "normalized_evaluation.csv", exported,
              ["candidate", "seed", "episode", "reward", "targetReached", "block", "jobDone", "wrongTarget", "liveLock"])
    return True


def main():
    parser = argparse.ArgumentParser(description="Paper-style plots directly from MAPPO V8 logs")
    parser.add_argument("--folder", default="mappo_runs/V8_15AGV_fromV7_5h", help="V8 run folder")
    parser.add_argument("--out", default=None, help="Figure folder (default figures/<run-name>)")
    parser.add_argument("--window", type=int, default=7, help="Rolling mean over episodes (default 7)")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be >= 1")
    folder = Path(args.folder)
    if not folder.is_dir():
        parser.error(f"Run folder not found: {folder.resolve()}")
    out = Path(args.out) if args.out else Path("figures") / folder.name
    out.mkdir(parents=True, exist_ok=True)
    config = folder / "run_config.json"
    if config.exists():
        try:
            obj = json.loads(config.read_text(encoding="utf-8-sig"))
            fleet = nested(obj, ("fleetsize",))
            if fleet is None:
                fleet = nested(obj, ("env", "fleetsize"))
            print(f"[CONFIG] fleetsize={fleet if fleet is not None else 'not found'}")
        except Exception as exc:
            print(f"[WARN] Cannot read run_config.json: {exc}")
    train_ok = plot_training(folder, out, args.window)
    eval_ok = plot_evaluations(folder, out)
    if not (train_ok or eval_ok):
        raise SystemExit("No plots generated. Please share first 2 lines of metrics.jsonl/evaluation file (hide personal paths).")
    print(f"\n[DONE] Output directory: {out.resolve()}")
    print("NOTE: episode X-axis is not training steps; evaluation error bars = sample SD.")
    print("NOTE: V7 8-AGV vs V8 15-AGV is NOT a controlled performance comparison.")


if __name__ == "__main__":
    main()
