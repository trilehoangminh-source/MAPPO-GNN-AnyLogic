"""Safely recover an incomplete V6 C/D evaluation JSONL without rerunning training.

Read and validate existing evaluation prefix; run ONLY missing evaluation seeds
using a separate port and output file; retain original backup; atomically merge.
Requires original scripts/run_B3_V6_CD_replication.py and V4 evaluator.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_B3_V6_CD_replication import valid_eval, valid_train  # noqa: E402
from v6_cd_common import CONDITIONS, condition_dir  # noqa: E402


def load_jsonl(path: Path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def validate_rows(rows, seeds, condition, checkpoint, *, allow_prefix=False):
    observed = [x.get("seed") for x in rows]
    if observed != list(seeds[:len(rows)]) if allow_prefix else observed != list(seeds):
        raise RuntimeError(f"Invalid or out-of-order evaluation seeds: {observed}; expected {seeds}")
    if len(rows) > len(seeds):
        raise RuntimeError("Unexpected extra evaluation rows")
    for r in rows:
        e = r.get("environment", {})
        if (r.get("evaluation") is not True
            or r.get("version") != "B3_EDGE_GAT_MAPPO_V4"
            or not e.get("reached_terminal", False)
            or not math.isclose(float(e.get("dijkstra_strength_eval", -1)), condition.guide, abs_tol=1e-10)
            or Path(r.get("checkpoint", "")).resolve() != checkpoint.resolve()):
            raise RuntimeError(f"Bad record for evaluation seed {r.get('seed')}")


def prepare(folder: Path, code: str, train_seed: int):
    manifest_file = folder / "experiment_manifest.json"
    if not manifest_file.is_file():
        raise RuntimeError(f"Experiment manifest missing: {manifest_file}")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if manifest.get("protocol") != "V6_CD_EdgeGAT_constant_guide_v1":
        raise RuntimeError("Expected original V6 C/D experiment protocol")
    if train_seed not in manifest["train_seeds"]:
        raise RuntimeError(f"Seed {train_seed} not among training seeds {manifest['train_seeds']}")
    condition = next(c for c in CONDITIONS if c.code == code)
    expected_conds = [{"code": x.code, "encoder": "residual_edge_gat", "guide": x.guide} for x in CONDITIONS]
    if manifest.get("conditions") != expected_conds:
        raise RuntimeError("Training conditions do not match original V6")
    train_folder = condition_dir(folder, condition, train_seed)
    checkpoint = train_folder / f"checkpoint_ep_{manifest['train_episodes']:04d}.pt"
    outfile = train_folder / f"evaluation_{manifest['eval_sim_seconds']}.jsonl"
    fake_args = SimpleNamespace(train_episodes=manifest["train_episodes"],
                                train_sim_seconds=manifest["train_sim_seconds"],
                                max_events=manifest["max_events"], hidden=manifest["hidden"],
                                graph_layers=manifest["graph_layers"], heads=manifest["heads"],
                                epochs=manifest["epochs"], eval_seeds=manifest["eval_seeds"])
    if not valid_train(train_folder, condition, train_seed, fake_args, Path(manifest["model_file"])):
        raise RuntimeError("Checkpoint/config/metrics incomplete. Refusing to alter any output")
    if not outfile.is_file():
        raise RuntimeError(f"Partial evaluation file missing: {outfile}")
    rows = load_jsonl(outfile)
    validate_rows(rows, manifest["eval_seeds"], condition, checkpoint, allow_prefix=True)
    return manifest, condition, checkpoint, outfile, rows, fake_args


def port_busy(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1)
        return s.connect_ex(("127.0.0.1", port)) == 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="mappo_runs/B3_V6_CD_30ep_720s")
    parser.add_argument("--condition", choices=["C", "D"], default="C")
    parser.add_argument("--train-seed", type=int, default=143)
    parser.add_argument("--port", type=int, default=51832, help="Fresh, unused local AnyLogic port")
    parser.add_argument("--dry-run", action="store_true")
    a = parser.parse_args(argv)
    if not 1 <= a.port <= 65535:
        parser.error("port outside 1..65535")
    folder = (ROOT / a.folder).resolve()
    manifest, cond, ckpt, out, rows, fake_args = prepare(folder, a.condition, a.train_seed)
    missing = manifest["eval_seeds"][len(rows):]
    print(f"[V6 RECOVER] condition={cond.code}, training_seed={a.train_seed}", flush=True)
    print(f"[V6 RECOVER] completed={len(rows)}/{len(manifest['eval_seeds'])}; missing={missing}", flush=True)
    if not missing:
        assert valid_eval(out, cond, fake_args, ckpt)
        print("[V6 RECOVER] Already complete: no files changed")
        return
    if a.dry_run:
        print("[V6 RECOVER] DRY RUN: existing data untouched. Would evaluate only:", missing)
        return
    if port_busy(a.port):
        raise RuntimeError(f"Port {a.port} already in use. Choose another with --port")
    evaluator = ROOT / "scripts" / "evaluate_B3_edgeGAT_MAPPO_v4.py"
    if not evaluator.is_file():
        raise FileNotFoundError(evaluator)
    token = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    recovery = folder / "recovery_logs"
    recovery.mkdir(exist_ok=True)
    delta_file = recovery / f"{a.condition}_train{a.train_seed}_missing_{token}.jsonl"
    process_log = recovery / f"{a.condition}_train{a.train_seed}_retry_{token}.log"
    cmd = [sys.executable, "-u", str(evaluator), "--checkpoint", str(ckpt),
           "--seeds", *[str(x) for x in missing],
           "--sim-seconds", str(manifest["eval_sim_seconds"]),
           "--port", str(a.port), "--output", str(delta_file)]
    print("[V6 RECOVER] Running:", subprocess.list2cmdline(cmd), flush=True)
    with process_log.open("x", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        assert proc.stdout
        for line in proc.stdout:
            print(line, end="", flush=True)
            handle.write(line)
            handle.flush()
        code = proc.wait()
        handle.write(f"\nEXIT_CODE={code}\n")
    if code:
        raise RuntimeError(f"AnyLogic evaluation retry failed. Existing 4 rows remain untouched. Log: {process_log}")
    extra = load_jsonl(delta_file)
    validate_rows(extra, missing, cond, ckpt)
    # A recheck detects unforeseen changes to the original file during retry.
    original = load_jsonl(out)
    if original != rows:
        raise RuntimeError("Original evaluation changed while retry ran. Refusing merge")
    combined = original + extra
    validate_rows(combined, manifest["eval_seeds"], cond, ckpt)
    backup = recovery / f"{a.condition}_train{a.train_seed}_partial_backup_{token}.jsonl"
    shutil.copy2(out, backup)
    stage = recovery / f"{a.condition}_train{a.train_seed}_staged_{token}.jsonl"
    with stage.open("x", encoding="utf-8") as handle:
        for x in combined:
            handle.write(json.dumps(x, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    if not valid_eval(stage, cond, fake_args, ckpt):
        raise RuntimeError(f"Merged data failed official V6 validation. Original kept. Stage: {stage}")
    os.replace(stage, out)
    print(f"[V6 RECOVER] SUCCESS: {len(combined)} valid rows -> {out}")
    print(f"[V6 RECOVER] Original 4-row backup: {backup}")
    print(f"[V6 RECOVER] Separate missing-seed output: {delta_file}")
    print("[V6 RECOVER] Next: rerun your ORIGINAL V6 command with --resume")


if __name__ == "__main__":
    main()
