# V11 — Congestion-Aware Residual Edge-GAT (20 AGVs)

This branch is reserved for the V11 experiment, based on the original V8 EP304 checkpoint. It is separate from `main` (V8) and `feature/v10-transformer`.

**Algorithm:** MAPPO + Residual Edge-GAT Routing with six actor-observation traffic features (`edge_occupancy`, shortest-path `predicted_flow`, `upstream_waiting`, `downstream_load`, `pressure`, `contraflow`). Directed static graph and Dijkstra action guidance **strength 1.5** remain enabled in V11. Dispatch Front/Rear use V8 MLP (not Transformer). This is **not V12**.

**Training provenance:**
- Source: V8 EP304, trained with reward part completed +2 and 20 AGVs.
- EP305–EP325: Routing-only updates.
- EP326 onward: `--train-roles all` updates Routing and both Dispatch policies/critics; `--max-minibatch 64`.
- Latest examined external metrics file: EP529 (NOT an uploaded checkpoint).
- V11 model checkpoints, metric JSONL, and AnyLogic runtime model are local-only and are not committed here.

**Important:** This branch currently includes the V11 traffic model. The actual runner, shared builder, evaluator and tests should be pushed from the **verified local working tree** so that code exactly matches the version that produced EP529. Do not assume this GitHub branch is runnable until those exact local files are present.

Suggested source paths:
```
thesis/mappo/congestion_edgegat_v11.py
scripts/v11_common.py
scripts/train_v11_congestion.py
scripts/evaluate_v11_congestion.py
tests/test_v11_traffic_offline.py
tests/test_v11_role_change.py
tests/test_v11_minibatch_change.py
```

Exclude `mappo_runs/`, `*.pt`, `*.pth`, `.venv311/`, logs and trained model artifacts. Always validate `git diff --cached --stat` before committing.
