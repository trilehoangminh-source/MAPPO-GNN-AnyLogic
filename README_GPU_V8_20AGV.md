# GPU for V8 20AGV — safe incremental patch

This patch replaces ONLY `scripts/train_B3_MAPPO_v8_overnight.py`, adds `scripts/check_B3_V8_gpu.py` and `tests/test_B3_V8_GPU_offline.py`. Preserve current `scripts/_mappo_v8_utils.py` **including** prior resume-metrics hotfix. Back up the existing runner before replacement.

## Critical distinction
- PyTorch MAPPO/EdgeGAT actor, critic and PPO optimizer can use NVIDIA CUDA when PyTorch CUDA build is installed.
- AnyLogic Java simulator, agent events, Alpyne HTTP still run on CPU. GPU may be slower for small networks; first test one complete episode.
- `--device cuda:0` fails if CUDA is not available and checks actor/critic tensors after model creation and checkpoint load. No silent fallback.
- `--cpu-threads 2` limits PyTorch CPU threads, **not Java's CPU usage**.

## Preparation
PowerShell in repo root with `.venv311`:
```powershell
python scripts/check_B3_V8_gpu.py
python tests/test_B3_V8_GPU_offline.py
```
CUDA AVAILABLE must be True and CUDA MATMUL PASS must display cuda:0. If false, run `nvidia-smi` or check Task Manager -> Performance -> GPU. NVIDIA GPU with drivers and CUDA-enabled PyTorch required; do not blindly change torch. See https://pytorch.org/get-started/locally/ .

## Preflight
Do not edit any checkpoint/state/metrics. Install previous resume-metrics hotfix first if not yet in place.

```powershell
python scripts/train_B3_MAPPO_v8_overnight.py `
  --from-previous-run mappo_runs/V8_15AGV_fromV7_5h `
  --fleetsize 20 `
  --hours 5 `
  --train-sim-seconds 1800 `
  --max-additional-episodes 500 `
  --snapshot-every 10 `
  --eval-reserve-minutes 45 `
  --eval-seeds 3301 3302 `
  --port-train 51880 `
  --port-eval 51881 `
  --output mappo_runs/V8_20AGV_from15_5h `
  --resume --device cuda:0 --cpu-threads 2 --dry-run
```
IMPORTANT: This `--dry-run` does not touch files and checks CUDA capability. For actual continuation, use the SAME original resume training/eval flags as when the run was created; add `--device cuda:0 --cpu-threads 2`. To limit the runtime safely, wait for the first committed episode, then interrupt with Ctrl+C. Keep source checkpoints. Do not run two processes concurrently.

## Post-launch proof
Expect lines `[V8 CUDA] cuda:0 | <name>` and `[V8 DEVICE CHECK] ... cuda:0 ...` and `MAPPO-V8 EP ...`. If training causes a CUDA OOM, existing committed episodes/checkpoints remain. If `model.load` does not support moving CPU optimizer state to CUDA, fail before update and inspect loader rather than editing checkpoint.

## Important caveat
Source snapshot is the V8 20AGV code version in this conversation; compare with your current runner if edited since then. Test coverage is offline only; actual CUDA and Java on user's Windows machine are untested.
