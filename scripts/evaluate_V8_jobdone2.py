"""Evaluate V8 Reward+2 with the SAME environment reward configuration."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import train_mappo_long_B3_v121 as baseline
baseline.SIM_CONFIG['reward_part_completed'] = 2.0
import evaluate_B3_MAPPO_v7 as evaluator

if __name__ == '__main__':
    print('[V8 EVAL] reward_part_completed=+2.0, V8 residual Edge-GAT', flush=True)
    evaluator.evaluator.main()
