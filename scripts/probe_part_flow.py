
"""Diagnosis-only Part-stage snapshot from Tobias Matrix AnyLogic observations.

Evaluates an existing MAPPO checkpoint with greedy actions.
No PPO training and no changes to the AnyLogic model.
Only observes Parts carried by AGVs at dispatch decision times.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import train_mappo as trainer


# Original Tobias Part.initRespot() rules.
# Each operation:
# [name, work, after_geo, finish_before_geo, rework_before_geo]
RESPOT_RULES = {
    "HC_V1": [
        ["HSN1", 22, 1, 4, 5],
        ["FLS", 9, 1, 3, 5],
        ["WPS", 29, 2, 5, 5],
        ["HSN2", 18, 2, 3, 5],
        ["WPS", 29, 3, 4, 5],
        ["HSN1", 22, 3, 4, 5],
        ["HSN2", 18, 4, 5, 5],
        ["IMP", 13, 4, 5, 5],
    ],
    "HC_V2": [
        ["HSN1", 22, 1, 2, 5],
        ["FLS", 9, 1, 3, 5],
        ["WPS", 29, 2, 3, 5],
        ["HSN2", 6, 2, 4, 5],
        ["WPS", 29, 3, 5, 5],
        ["HSN2", 18, 3, 4, 5],
        ["HSN1", 22, 3, 4, 5],
        ["IMP", 13, 4, 5, 5],
        ["WPS", 34, 4, 5, 5],
    ],
    "VB_V1": [
        ["WPS", 29, 1, 5, 6],
        ["HSN2", 18, 1, 2, 6],
        ["HSN1", 11, 2, 4, 6],
        ["HSN2", 18, 2, 6, 6],
        ["WPS", 29, 3, 5, 6],
        ["HSN1", 11, 3, 5, 6],
        ["FLS", 4, 4, 6, 6],
        ["IMP", 20, 4, 5, 6],
        ["FLS", 4, 5, 6, 6],
        ["HSN2", 18, 5, 6, 6],
        ["WPS", 29, 5, 6, 6],
    ],
    "VB_V2": [
        ["WPS", 29, 1, 3, 6],
        ["HSN2", 18, 1, 3, 6],
        ["HSN1", 11, 2, 5, 6],
        ["WPS", 18, 3, 6, 6],
        ["FLS", 4, 4, 5, 6],
        ["IMP", 20, 4, 5, 6],
        ["FLS", 4, 5, 6, 6],
        ["HSN2", 18, 5, 6, 6],
        ["WPS", 46, 5, 6, 6],
    ],
}

STAGES = [
    "g01", "g12", "g23", "g34",
    "g45", "g56", "g6d", "finalState"
]

VARIANTS = ["VB_V1", "VB_V2", "HC_V1", "HC_V2"]

PART_BEGIN = 18
PART_VARIANTS = 4
PART_STATES = 8
PART_END = 53


def decode_part(raw_row, agv_slot, time, caller):

    part = np.asarray(
        raw_row[PART_BEGIN:PART_END],
        dtype=np.float32
    )

    if part.size != PART_END - PART_BEGIN:
        raise ValueError(
            f"Incorrect Part observation width: {part.size}"
        )

    var_flags = part[:PART_VARIANTS]

    stage_flags = part[
        PART_VARIANTS:PART_VARIANTS + PART_STATES
    ]

    var_indices = np.flatnonzero(var_flags > 0.5)
    stage_indices = np.flatnonzero(stage_flags > 0.5)

    if not len(var_indices) and not len(stage_indices):
        return None

    if len(var_indices) != 1 or len(stage_indices) != 1:
        return {
            "time": time,
            "caller": caller,
            "agv_slot": agv_slot,
            "variant": "UNDECODABLE",
            "stage": "UNDECODABLE",
            "pending_before_next_geo": "",
            "unfinished_operations": "",
            "failed_rework_due": "",
            "note": "Unexpected one-hot: "
                    + repr(part[:12].tolist())
        }

    variant = VARIANTS[int(var_indices[0])]
    stage = STAGES[int(stage_indices[0])]

    rules = RESPOT_RULES.get(variant, [])

    flag_block = part[
        PART_VARIANTS + PART_STATES:
    ]

    done = [
        bool(flag_block[2 * i] > 0.5)
        for i in range(len(rules))
    ]

    nio = [
        bool(flag_block[2 * i + 1] > 0.5)
        for i in range(len(rules))
    ]

    remaining = [
        f"{i+1}:{r[0]}"
        for i, r in enumerate(rules)
        if not done[i]
    ]

    due = []
    rework_due = []

    if (
        stage.startswith("g")
        and stage[1:].isdigit()
        and len(stage) == 3
    ):
        next_geo = int(stage[-1])

        if variant.startswith("HC") and stage == "g56":
            next_geo = None

        if stage == "g6d":
            next_geo = None

        if next_geo is not None:

            due = [
                f"{i+1}:{r[0]}"
                for i, r in enumerate(rules)
                if r[3] <= next_geo and not done[i]
            ]

            rework_due = [
                f"{i+1}:{r[0]}"
                for i, r in enumerate(rules)
                if r[4] <= next_geo and nio[i]
            ]

    return {
        "time": time,
        "caller": caller,
        "agv_slot": agv_slot,
        "variant": variant,
        "stage": stage,
        "pending_before_next_geo": ", ".join(due),
        "unfinished_operations": ", ".join(remaining),
        "failed_rework_due": ", ".join(rework_due),
        "note": ""
    }


def main():

    ap = argparse.ArgumentParser(
        description="Evaluate MAPPO checkpoint and inspect Parts"
    )

    ap.add_argument(
        "--model",
        default="envs/Matrix_python39.zip"
    )

    ap.add_argument(
        "--checkpoint",
        default="mappo_runs/check_720/latest_mappo.pt"
    )

    ap.add_argument("--port", type=int, default=51270)
    ap.add_argument("--startup-timeout", type=int, default=180)
    ap.add_argument("--fleetsize", type=int, default=8)
    ap.add_argument("--max-fleetsize", type=int, default=20)
    ap.add_argument("--sim-seconds", type=int, default=720)
    ap.add_argument("--max-events", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=45)

    ap.add_argument(
        "--output",
        default="mappo_runs/part_probe"
    )

    ap.add_argument("--device", default="cpu")
    ap.add_argument("--env-type", default="matrix")

    ap.add_argument(
        "--routing-backbone",
        default="original_gnn"
    )

    ap.add_argument(
        "--dispatch-backbone",
        default="original_linear"
    )

    ap.add_argument(
        "--single-dispatch-policy",
        action="store_true"
    )

    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--mini-batches", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=32)

    args = ap.parse_args()

    checkpoint = Path(args.checkpoint).resolve()

    if not checkpoint.is_file():
        ap.error(f"Checkpoint not found: {checkpoint}")

    if not Path(args.model).is_file():
        ap.error(f"Model not found: {args.model}")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    snapshots = []
    observed_dispatches = [0]
    env = None

    try:

        print(
            "[PROBE] Starting AnyLogic Matrix...",
            flush=True
        )

        env = trainer.build_env(args)
        original = env._capture_mappo_state

        def monitor_dispatch(obs):

            original(obs)

            caller = int(obs.caller)

            if not 1000 <= caller < 2000:
                return

            raw = np.asarray(
                obs.obs,
                dtype=np.float32
            )

            if (
                raw.ndim != 2
                or raw.shape[1] < PART_END
                or raw.shape[0] < args.fleetsize
            ):
                raise ValueError(
                    "Unexpected dispatch observation shape: "
                    + str(raw.shape)
                )

            observed_dispatches[0] += 1

            sim_time = round(
                float(env.get_mappo_time()),
                3
            )

            for slot in range(args.fleetsize):

                row = decode_part(
                    raw[slot],
                    slot,
                    sim_time,
                    caller
                )

                if row is not None:
                    snapshots.append(row)

        env._capture_mappo_state = monitor_dispatch

        print(
            "[PROBE] Building MAPPO network "
            "and loading checkpoint...",
            flush=True
        )

        model = trainer.make_model(env, args)
        model.load(checkpoint)

        snapshots.clear()
        observed_dispatches[0] = 0

        print(
            "[PROBE] Checkpoint loaded. "
            "Greedy policy evaluation, NO training.",
            flush=True
        )

        from thesis.mappo import EventCollector

        collector = EventCollector(model, env)

        _, meta = collector.episode(
            seed=args.seed,
            train=False,
            max_event_steps=args.max_events
        )

        # ================================================
        # NEW: GREEDY ROUTING ACTION DIAGNOSTICS
        # ================================================

        diagnostics = meta.get(
            "action_diagnostics", {}
        )

        routing_actions = (
            diagnostics
            .get("actions_by_role", {})
            .get("routing", {})
        )

        routing_total = sum(
            int(v) for v in routing_actions.values()
        )

        routing_stops = int(
            routing_actions.get("0", 0)
        )

        stop_percent = (
            100.0 * routing_stops / routing_total
            if routing_total else 0.0
        )

        print(
            "[PROBE] Greedy routing actions:",
            routing_actions,
            flush=True
        )

        print(
            f"[PROBE] Routing action 0: "
            f"{routing_stops}/{routing_total} "
            f"({stop_percent:.1f}%)",
            flush=True
        )

        print(
            "[PROBE] Action mask violations:",
            diagnostics.get(
                "selected_mask_violations",
                "not available"
            ),
            flush=True
        )

        # ================================================
        # SAVE PART SNAPSHOTS
        # ================================================

        keys = [
            "time",
            "caller",
            "agv_slot",
            "variant",
            "stage",
            "pending_before_next_geo",
            "unfinished_operations",
            "failed_rework_due",
            "note"
        ]

        with (
            out / "part_snapshots.csv"
        ).open(
            "w",
            encoding="utf-8-sig",
            newline=""
        ) as fp:

            wr = csv.DictWriter(
                fp,
                fieldnames=keys
            )

            wr.writeheader()
            wr.writerows(snapshots)

        stat = meta.get("statistics", {})

        work = {
            k: v
            for k, v in stat.items()
            if k.startswith("_jobnum_")
        }

        counted = collections.Counter(
            r["stage"] for r in snapshots
        )

        variants = collections.Counter(
            r["variant"] for r in snapshots
        )

        blocked = collections.Counter(
            item.strip().split(":", 1)[-1]
            for r in snapshots
            for item in r[
                "pending_before_next_geo"
            ].split(", ")
            if item.strip()
        )

        summary = {
            "notes": [
                "No PPO training; greedy evaluation.",
                "Snapshots are not unique Parts.",
                "Only Parts onboard AGVs are observable.",
                "Parts inside stations remain invisible."
            ],
            "events": meta.get("events"),
            "jobDone": stat.get("jobDone"),
            "targetReached": stat.get("targetReached"),
            "newPart": stat.get("newPart"),
            "registered_station_work": work,
            "dispatch_observations": observed_dispatches[0],
            "onboard_part_snapshots": len(snapshots),
            "stage_sample_counts": dict(counted),
            "variant_sample_counts": dict(variants),
            "prerequisite_missing_sample_counts": dict(blocked),

            # NEW: GREEDY ACTION RESULTS
            "greedy_routing_actions": routing_actions,
            "routing_action_0_percent": round(
                stop_percent, 2
            ),
            "action_mask_violations": diagnostics.get(
                "selected_mask_violations"
            )
        }

        (
            out / "part_summary.json"
        ).write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
                default=str
            ),
            encoding="utf-8"
        )

        print(
            "[PROBE] Done. Events:",
            summary["events"],
            "dispatch snapshots:",
            observed_dispatches[0],
            "onboard Parts observed:",
            len(snapshots),
            flush=True
        )

        print(
            "[PROBE] Stage sample counts:",
            dict(counted),
            flush=True
        )

        print(
            "[PROBE] Missing prerequisite sample counts:",
            dict(blocked),
            flush=True
        )

        print(
            "[PROBE] JobDone:",
            summary["jobDone"],
            flush=True
        )

        print(
            "[PROBE] Results:",
            out / "part_snapshots.csv",
            "and",
            out / "part_summary.json",
            flush=True
        )

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
