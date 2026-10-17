# MAPPO B2: Operation-first dispatch decoder (experiment-only)

## Goal
The validated live action mapping gave action 21 = puffer node 95 and action 22 = puffer node 96. B checkpoint always chose these buffers; jobDone stayed 0. This ZIP implements a **controlled experimental intervention**, not a claim of superior performance.

Method adapted from:
- Y. Li et al. (2025), *Real-Time Scheduling for Flexible Job Shop With AGVs Using Multiagent Reinforcement Learning and Efficient Action Decoding*, IEEE TSMC: https://doi.org/10.1109/TSMC.2024.3520381. **MARL, not specifically verified as MAPPO.**
- L. Zheng et al. (2026), *Real-time dynamic integrated process planning and scheduling with reconfigurable manufacturing cells via multi-agent reinforcement learning*, Journal of Manufacturing Systems: https://doi.org/10.1016/j.jmsy.2026.01.004 (operation-aware multi-agent scheduling; reward sharing is not implemented in this B2 experiment).
- S. Peng (2023), *Multi-Robot Path Planning Combining Heuristics and Multi-Agent Reinforcement Learning*, https://arxiv.org/abs/2306.01270 (Routing guidance concept for a **separate future** experiment, not implemented here).
- C. Yu et al. (2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, https://arxiv.org/abs/2103.01955 (MAPPO controls, action masking, centralized critic).

## Files added
- `thesis/envs/dispatch_paper_mask_v121.py`: operation-first decoder.
- `scripts/apply_dispatch_paper_patch_v121.py`: one-time, exact-anchor patch installer with backup; **the ZIP does not overwrite `behaviors_v121.py`**.
- `scripts/train_mappo_long_B2_v121.py`: B2 training, own manifest/weights.
- `scripts/evaluate_mappo_long_B2_v121.py`: held-out greedy B2 eval, requires B2 checkpoint.
- `tests/test_dispatch_paper_mask_v121.py`: unit tests no Java required.
- `scripts/compare_mappo_B_B2_v121.py`: paired evaluation table/CSV.

## Install
Extract this ZIP **into** `C:\MARL_AnyLogic\MAPPO_NEW\thesis` (the project root containing `scripts` and `thesis`). Existing files are preserved; files named in this ZIP are new. Ensure existing `scripts/train_mappo_long_B_v121.py` remains intact.

Activate `.venv311`, then:
```powershell
python scripts/apply_dispatch_paper_patch_v121.py
python tests/test_dispatch_paper_mask_v121.py
```
The installer edits `thesis/envs/behaviors_v121.py` in place **only if its exact original anchor matches**, and saves `behaviors_v121.py.paper_b2_backup` first. To roll back the hook safely, restore the backup after saving any subsequent edits. By default A/B behavior remains *legacy*; only B2 sets `MAPPO_DISPATCH_MASK_MODE=operation_first`.

## B2 smoke (start small)
```powershell
python -u scripts/train_mappo_long_B2_v121.py --model envs/Matrix_critic_fresh.zip --port 51670 --episodes 2 --sim-seconds 720 --seed 43 --routing-backbone pooled --dispatch-backbone pooled --stop-mask-mode forbid_when_move --output mappo_runs/B2_operation_first_smoke
```
Watch `[DISPATCH_B2] ... reason=... names=... allowed=... buffer_fallback=...` and check for `fresh_routing>0`, `closed_transitions>0` and 0 mask violations. Run A/B from their original scripts unchanged to confirm the default remains legacy.

## B2 50 episodes (do **not** use smoke checkpoint)
```powershell
python -u scripts/train_mappo_long_B2_v121.py --model envs/Matrix_critic_fresh.zip --port 51671 --episodes 50 --sim-seconds 720 --seed 43 --routing-backbone pooled --dispatch-backbone pooled --stop-mask-mode forbid_when_move --output mappo_runs/long_B2_seed43_50ep
```
```powershell
python -u scripts/evaluate_mappo_long_B2_v121.py --checkpoint mappo_runs/long_B2_seed43_50ep/checkpoint_ep_0050.pt --seeds 1001 1002 1003 --sim-seconds 3600 --port 51672 --output mappo_runs/long_B2_seed43_50ep/evaluation_3600.jsonl
```
Create a paired comparison report (after B2 evaluation):
```powershell
python scripts/compare_mappo_B_B2_v121.py --b mappo_runs/long_B_seed43_50ep/evaluation_3600.jsonl --b2 mappo_runs/long_B2_seed43_50ep/evaluation_3600.jsonl --out mappo_runs/B_vs_B2_eval_3600.csv
```
Compare **both** B and B2 with same held-out seeds, horizon and budget. Inspect `jobDone`, per-process completions, fraction of wrongTarget among arrivals, blocked/livelock counts, destination distribution. Don't interpret buffer=0 as proof of improvement: mask enforces it.

## Limitations and follow-ups
- This experiment does not check Station occupancy/breakdowns or whether an AGV *should* wait in a buffer during congestion. It allows buffer if no stage-eligible station is found, not whenever stations happen to be busy; thus it could **worsen congestion**. Always track `block`, `liveLock`.
- It follows Tobias `CleverMatrixDispatcher` precedence (rework -> GEO -> operations) but not the full task/production physics. Stage semantics are from `MatrixPart.translate` (original source); verify against AnyLogic state machine before claiming correct processing.
- No Dijkstra, reward shaping, RNN, PopArt, async credit assignment, or changes to Java are included in this patch. Those require separate experiments with direct observation validation.
- Tests verify logic and syntax offline only; live AnyLogic and checkpoint performance must be tested on user's Windows machine.
