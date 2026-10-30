"""V8 Residual Edge-GAT: override JobDone reward only for this process.

Does not modify global SIM_CONFIG on disk or reload any checkpoint itself.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Must be set BEFORE V8 imports baseline.build_env through v7/v4 modules.
import train_mappo_long_B3_v121 as baseline
baseline.SIM_CONFIG["reward_part_completed"] = 2.0
print("[V8 REWARD ABLATION] reward_part_completed=+2.0; residual_edge_gat; no architecture change", flush=True)

from run_v8_quiet import main

if __name__ == "__main__":
    main()
