# Tobias MAPPO-B3 — operation-first Dispatching + directed-graph-guided Routing

**Reference:** Shaoming Peng, *Multi-Robot Path Planning Combining Heuristics and Multi-Agent Reinforcement Learning*, 2023, MAPPOHR, https://arxiv.org/abs/2306.01270.

**What this is:** A narrow, independently developed MAPPOHR-inspired **soft shortest-path logit prior** adapted to the original Tobias directed network graph. **It is not a reproduction** of MAPPOHR's dynamic D* Lite/local time-window planner, and does **not** claim collision-free paths or improved performance before evaluation.

### Safety and scope

- Only **new** files. No overwriting `algorithm.py`, `networks.py`, `behaviors_v121.py`, AnyLogic Java, B2 checkpoints or previous runs. B2 hook **must already be installed**, as in your existing project.
- Original B2 Dispatching operation-first mask unchanged. Routing Stop mask `forbid_when_move` unchanged. Reward, PPO hyperparameters, station critic and model pooled backbone unchanged.
- `HiveContext.nodes` and `HiveContext.paths` are the **live directed graph** populated by Tobias's `networkcontext`. The target and four candidate next nodes are read from ego Actor observation columns `[6:8]` and `[8:16]`. This does not expose critic-only information to Actor.
- On every routing decision, distance from **each legal adjacent next node** to target is computed by reversed Dijkstra. The legal shortest move gets an additive logit prior `+beta` (`beta=1.5` default). It is **not a hard mask**; collision handling is still learned by MAPPO. When no path/graph match is possible, fall back to existing policy and count failure.
- This prior is stored **inside each rollout observation** as `routing_guide_bias` and is used in both `model.act()` and `MAPPO.train()`; PPO likelihood ratios use the same action distribution. A unit test checks this invariant. No *post-sampling* forced action correction.
- Checkpoint neural parameters are the usual MAPPO ones; B3 evaluation MUST load with the B3 script or the policy's action distribution will change.
- Counter metrics: `guided`, `no_legal_path`, `unmapped_coordinates`, `missing_graph`, `guided_actions`, `selected_guide`. If no paths were guided in a whole episode, training stops with an explicit exception: do not waste 50 episodes on a nonfunctional guide.

### Install into existing project

Extract ZIP into `C:\MARL_AnyLogic\MAPPO_NEW\thesis` (the folder containing `scripts` and the inner package `thesis`). Use `.venv311`.

```powershell
python scripts/check_mappo_B3_setup_v121.py
python tests/test_routing_guidance_B3_v121.py
```

If setup reports B2 hook missing, do **not** install B3 over your source; restore your B2 installation first. If unit test succeeds, run one controlled smoke test:

```powershell
python -u scripts/train_mappo_long_B3_v121.py --model envs/Matrix_critic_fresh.zip --port 51680 --episodes 2 --sim-seconds 720 --seed 43 --routing-backbone pooled --dispatch-backbone pooled --stop-mask-mode forbid_when_move --routing-guide-strength 1.5 --output mappo_runs/B3_routing_guide_smoke
```

Inspect `[EP ... guide=N guide_hit=M]` in terminal and `routing_guidance` in metrics. Need: `guided > 0`, `missing_graph=0` (ideally), significant guided fraction, zero mask violations. `selected_guide` should not be assumed to equal all decisions. The soft prior does not force every move.

### Controlled 50-episode B2 vs B3 comparison

Train B3 fresh from scratch with **seed 43**, **50 episodes**, **720 seconds**; leave B2 50-episode baseline untouched:

```powershell
python -u scripts/train_mappo_long_B3_v121.py --model envs/Matrix_critic_fresh.zip --port 51681 --episodes 50 --sim-seconds 720 --seed 43 --routing-backbone pooled --dispatch-backbone pooled --stop-mask-mode forbid_when_move --routing-guide-strength 1.5 --output mappo_runs/B3_graph_guided_seed43_50ep
```

Evaluate B3 and B2 on identical **held-out** seeds 1001/1002/1003 and horizons. These commands DO NOT train and DO NOT overwrite B2:

```powershell
python -u scripts/evaluate_mappo_long_B3_v121.py --checkpoint mappo_runs/B3_graph_guided_seed43_50ep/checkpoint_ep_0050.pt --seeds 1001 1002 1003 --sim-seconds 720 --port 51682 --output mappo_runs/B3_graph_guided_seed43_50ep/evaluation_720.jsonl
python -u scripts/evaluate_mappo_long_B3_v121.py --checkpoint mappo_runs/B3_graph_guided_seed43_50ep/checkpoint_ep_0050.pt --seeds 1001 1002 1003 --sim-seconds 3600 --port 51683 --output mappo_runs/B3_graph_guided_seed43_50ep/evaluation_3600.jsonl
```

**Before comparing**, ensure B2 baseline evaluation is run using `evaluate_mappo_long_B2_v121.py` for the B2 checkpoint, with the same 1001/1002/1003 and 3600 seconds.

```powershell
python scripts/compare_mappo_B2_B3_v121.py --b2 mappo_runs/B2_operation_first_50ep/evaluation_3600.jsonl --b3 mappo_runs/B3_graph_guided_seed43_50ep/evaluation_3600.jsonl --out mappo_runs/B2_vs_B3_3600.csv
```

If evaluation output files already exist, choose new filenames. Check `jobDone`, operation counts, target/wrong ratio, block/livelock, not just `guide_hit` or training reward. This is one 50-episode training seed, not proof of statistical significance. Subsequent paired seeds may be needed.

### Diagnostic limitations

This static global path guide may be unavailable for some graph states or targets; the adaptive policy must handle dynamic robot traffic. A strong beta can override useful collision-avoidance behavior even if actions are technically legal. Try lower beta 0.5 and 1.5 with fresh training runs if needed. If B3 fails to improve `jobDone` or operations, isolate Routing using Tobias's **existing** `CollisionFreeRouting` benchmark or compare direction-value/advantage data before attempting additional reward changes.
