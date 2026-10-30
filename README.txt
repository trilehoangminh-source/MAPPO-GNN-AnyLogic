MAPPO V8 / Residual Edge-GAT / JobDone reward +2 / EP205 checkpoint

Extract to: C:\MARL_AnyLogic\MAPPO_NEW\thesis

FIRST: Stop V9 cleanly with Ctrl+C in the V9 training terminal. Keep its V9 checkpoints and metrics as an ablation reference.

Start in PowerShell:
  cd C:\MARL_AnyLogic\MAPPO_NEW\thesis
  python -u .\scripts\start_V8_jobdone2_from205.py --dry-run
  python -u .\scripts\start_V8_jobdone2_from205.py --hours 5 --device cuda:0

Resume this V8 reward+2 experiment for another 5h after interruption:
  python -u .\scripts\start_V8_jobdone2_from205.py --resume --hours 5 --device cuda:0

Evaluate latest checkpoint separately with THE MATCHING reward config:
  python -u .\scripts\evaluate_V8_jobdone2.py --checkpoint .\mappo_runs\V8_RESIDUAL_EDGEGAT_JOBDONE2_FROM205\latest_mappo.pt --seeds 900001 900002 900003 900004 900005 --sim-seconds 1800 --device cuda:0 --port 51881 --guide-strength 1.5 --output .\mappo_runs\V8_RESIDUAL_EDGEGAT_JOBDONE2_FROM205\heldout_5seeds.jsonl

Default PPO minibatches are UNCHANGED from V8 (MAPPO_MAX_MINIBATCH=0).
If CUDA OOM, restart the saved run with --resume --max-minibatch 128 (this will change PPO optimizer minibatch dynamics).
No global SIM_CONFIG is rewritten on disk.
