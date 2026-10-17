"""Run paired A/B Stop-mask MAPPO experiments, sequentially, using the current v1 trainer.

Only adds NEW output directories. Existing complete runs are validated and skipped.
Existing partial/mismatched runs are never overwritten.
Run from project root using .venv311 Python.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINER = ROOT / "scripts" / "train_mappo_stop_ab.py"


def load_records(metrics: Path):
    try:
        lines = metrics.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read metrics: {metrics}: {exc}") from exc


def validate_completed(output: Path, seed: int, mode: str, episodes: int, seconds: int):
    metrics = output / "metrics.jsonl"
    if not metrics.is_file():
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Nonempty output without completed metrics: {output}. "
                               "Move it aside or select a new --output-root; never overwrite experimental runs.")
        return False
    records = load_records(metrics)
    if len(records) != episodes:
        raise RuntimeError(f"Partial output {metrics}: {len(records)} episodes, expected {episodes}. "
                           "Move it aside or select a new --output-root.")
    for i, rec in enumerate(records, 1):
        exp = rec.get("experiment", {})
        if (rec.get("episode") != i or exp.get("seed") != seed
                or exp.get("stop_mask_mode") != mode or exp.get("sim_seconds") != seconds
                or exp.get("routing_backbone") != "pooled"
                or exp.get("dispatch_backbone") != "pooled"):
            raise RuntimeError(f"Existing output does not match settings: {metrics}, episode {i}. "
                               "Use a NEW --output-root.")
    return True


def run_one(*, seed: int, mode: str, output: Path, port: int, args):
    if validate_completed(output, seed, mode, args.episodes, args.sim_seconds):
        print(f"[SKIP] seed={seed} mode={mode}: verified {args.episodes} episodes in {output}", flush=True)
        return
    if not TRAINER.is_file():
        raise FileNotFoundError(f"Missing {TRAINER}. Copy the Stop Action A/B v1 files first.")
    output.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, "-u", str(TRAINER),
        "--model", str(ROOT / args.model),
        "--port", str(port),
        "--sim-seconds", str(args.sim_seconds),
        "--episodes", str(args.episodes),
        "--seed", str(seed),
        "--routing-backbone", "pooled",
        "--dispatch-backbone", "pooled",
        "--stop-mask-mode", mode,
        "--output", str(output),
    ]
    log_path = output / "console.log"
    print(f"\n[RUN] seed={seed} {'A baseline' if mode == 'baseline' else 'B forbid_when_move'} "
          f"port={port} episodes={args.episodes} seconds={args.sim_seconds}", flush=True)
    print(f"[LOG] {log_path.relative_to(ROOT) if log_path.is_relative_to(ROOT) else log_path}", flush=True)
    warn_count = 0
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + subprocess.list2cmdline(cmd) + "\n\n")
        log.flush()
        with subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        ) as proc:
            assert proc.stdout is not None
            last_lines = []
            for line in proc.stdout:
                log.write(line)
                last_lines.append(line.rstrip())
                if len(last_lines) > 25:
                    last_lines.pop(0)
                if "observation.networkcontext = None" in line:
                    warn_count += 1
                elif line.startswith(("[MAPPO]", "[AB]", "Episode ", "Traceback", "RuntimeError", "Error")):
                    print("   " + line.rstrip(), flush=True)
            rc = proc.wait()
    if warn_count:
        print(f"[INFO] {warn_count} networkcontext warnings retained in console.log (not fixed or suppressed).")
    if rc != 0:
        raise RuntimeError(f"Trainer exited {rc} for seed={seed}, mode={mode}. "
                           f"Check {log_path}. Final output:\n" + "\n".join(last_lines[-18:]))
    if not validate_completed(output, seed, mode, args.episodes, args.sim_seconds):
        raise RuntimeError(f"Trainer exited 0 without metrics: {output}")
    print(f"[OK] seed={seed} mode={mode}; complete", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Paired multi-seed Stop action A/B runs (no overwrites)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[44, 45, 46])
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--sim-seconds", type=int, default=720)
    parser.add_argument("--port-base", type=int, default=51420)
    parser.add_argument("--model", default="envs/Matrix_critic_fresh.zip")
    parser.add_argument("--output-root", type=Path, default=Path("mappo_runs/stop_ab_multiseed"))
    args = parser.parse_args(argv)
    if args.episodes < 1 or args.sim_seconds < 1 or len(args.seeds) != len(set(args.seeds)):
        parser.error("episodes/sim-seconds must be positive, and seeds must be unique")
    if not (1024 <= args.port_base and args.port_base + 2 * len(args.seeds) - 1 <= 65535):
        parser.error("port range must be between 1024 and 65535")
    if not (ROOT / args.model).is_file():
        parser.error(f"AnyLogic model not found: {ROOT / args.model}")
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    print(f"[PLAN] {len(args.seeds)} paired seeds={args.seeds}, "
          f"{args.episodes} episodes per arm, {args.sim_seconds}s each, sequential.", flush=True)
    print(f"[PYTHON] {sys.executable}", flush=True)
    print(f"[OUTPUT] {output_root}", flush=True)
    for n, seed in enumerate(args.seeds):
        # Alternate A/B run order to reduce systematic hardware/time-of-run effects.
        order = ("baseline", "forbid_when_move") if n % 2 == 0 else ("forbid_when_move", "baseline")
        for arm_pos, mode in enumerate(order):
            port = args.port_base + 2 * n + arm_pos
            output = output_root / f"seed_{seed}" / ("A" if mode == "baseline" else "B")
            run_one(seed=seed, mode=mode, output=output, port=port, args=args)
    print("\n[DONE] All paired experiments complete. Run summarize_stop_ab_multiseed.py.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, FileNotFoundError) as exc:
        print(f"[STOP] {exc}", file=sys.stderr)
        sys.exit(1)
