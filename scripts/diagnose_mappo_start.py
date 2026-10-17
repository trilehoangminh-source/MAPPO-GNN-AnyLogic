"""Diagnose AnyLogic's MODEL_ERROR during Tobias MAPPO environment startup.

Place this file next to scripts/train_mappo.py, then run it with the same
CLI arguments. It preserves MAPPO's training code, intercepts only the
Matrix startup exception, and captures Java diagnostics before re-raising.

Windows PowerShell:
  python -u scripts/diagnose_mappo_start.py --model envs/Matrix.zip --port 51160 --fleetsize 8 --sim-seconds 120 --episodes 2 --output mappo_runs/smoke
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path

# Make both repo root (for thesis.*) and this directory (for train_mappo)
# importable, regardless of the current working directory.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


def _tail(path: Path, max_lines: int = 60) -> str:
    try:
        if not path.is_file():
            return "(not found)"
        with path.open("r", encoding="utf-8", errors="replace") as stream:
            lines = stream.readlines()
        return "".join(lines[-max_lines:]).strip() or "(empty)"
    except Exception as exc:
        return f"(could not read {path}: {exc})"


def _decode(raw) -> str:
    if raw is None:
        return "(stream not captured)"
    if isinstance(raw, bytes):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


def _capture_java(proc) -> None:
    if proc is None:
        print("[DIAG] No Java process handle was saved.", flush=True)
        return

    print(f"[DIAG] Java PID={proc.pid}; exit-code-before-stop={proc.poll()}", flush=True)
    # The simulation already returned MODEL_ERROR. Stop this Java process so
    # communicate() can consume stderr/stdout without waiting forever.
    if proc.poll() is None:
        try:
            proc.terminate()
        except Exception as exc:
            print(f"[DIAG] Java terminate warning: {exc}", flush=True)

    try:
        stdout, stderr = proc.communicate(timeout=12)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=12)
    except Exception as exc:
        print(f"[DIAG] Java output capture failed: {exc}", flush=True)
        return

    for title, blob in (("JAVA STDERR", stderr), ("JAVA STDOUT", stdout)):
        body = _decode(blob)
        print(f"\n========== {title} (last 16000 characters) ==========", flush=True)
        print(body[-16000:] or "(empty)", flush=True)


def main() -> None:
    from thesis.envs.matrix import Matrix
    import train_mappo

    original_start = Matrix._start

    def _start_with_diagnostics(self):
        try:
            return original_start(self)
        except Exception as exc:
            print("\n" + "=" * 62, flush=True)
            print("[DIAG] MATRIX/ANYLOGIC STARTUP FAILED", flush=True)
            print(f"[DIAG] Python error: {type(exc).__name__}: {exc}", flush=True)
            print("[DIAG] This error occurred before the MAPPO trainer started.", flush=True)
            print("=" * 62, flush=True)

            # These are best-effort diagnostics, not an attempted repair.
            client = getattr(self, "client", None)
            proc = getattr(client, "_proc", None)
            try:
                _capture_java(proc)
            except Exception as capture_exc:
                print(f"[DIAG] Java diagnostics failed: {capture_exc}", flush=True)

            for path in (ROOT / "alpyne.log", ROOT / "model.log"):
                print(f"\n========== LOG: {path} (last 60 lines) ==========", flush=True)
                print(_tail(path), flush=True)

            print("\n[DIAG] Please share the JAVA STDERR/STDOUT and LOG sections.", flush=True)
            raise

    Matrix._start = _start_with_diagnostics
    train_mappo.main()


if __name__ == "__main__":
    main()
