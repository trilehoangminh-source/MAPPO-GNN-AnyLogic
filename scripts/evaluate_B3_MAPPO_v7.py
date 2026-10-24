"""Evaluate V7 MAPPO checkpoint using the SAME Edge-GAT V4 evaluator.

Only its version guard is changed for V7 checkpoint directories. All actor
construction, GraphSpec checks, masks and greedy policy remain identical.
"""
from __future__ import annotations
import evaluate_B3_edgeGAT_MAPPO_v4 as evaluator

evaluator.VERSION = 'B3_MAPPO_STABILITY_V7'

if __name__ == '__main__':
    evaluator.main()
