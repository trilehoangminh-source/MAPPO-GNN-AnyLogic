MAPPO V8 RESIDUAL EDGE-GAT — XEM ANIMATION 20 AGV TRONG ANYLOGIC

1. Giai nen ZIP tai thu muc C:\MARL_AnyLogic\MAPPO_NEW\thesis.
   File can co: thesis\pypeline_connectors\matrix_mappo_v8_20agv.py

2. Trong VS Code PowerShell, chay:
   cd C:\MARL_AnyLogic\MAPPO_NEW\thesis
   $run = ".\mappo_runs\V8_RESIDUAL_EDGEGAT_JOBDONE2_FROM205"
   $s = Get-Content "$run\state.json" -Raw | ConvertFrom-Json
   Write-Host "Last EP:" $s.last_episode
   if (!(Test-Path $s.checkpoint)) { throw "Khong tim thay checkpoint" }
   Copy-Item -LiteralPath $s.checkpoint -Destination "$run\desktop_brain.pt" -ErrorAction Stop
   .\.venv311\Scripts\python.exe -c "from thesis.pypeline_connectors.matrix_mappo_v8_20agv import get_config; print('OK: fleet =', get_config('fleetsize'))"

3. Mo .alp local (ban da dung va da test AnyLogic 8.9):
   C:\MARL_AnyLogic\MAPPO_NEW\thesis\envs\Matrix1\Matrix1_AGV17_trace.alp
   (hoac file .alp local hien tai cua ban). KHONG mo .zip RL export de xem animation.

4. Trong Main (model editor):
   RL_pypeline_path = "C:/MARL_AnyLogic/MAPPO_NEW/thesis"
   RL_pypeline_name = "matrix_mappo_v8_20agv"
   runmode = 3
   py/pythonExecPath = "C:/MARL_AnyLogic/MAPPO_NEW/thesis/.venv311/Scripts/python.exe"
   RL_reward_part_completed = 2 (neu muon reward thong ke khop)

5. Run experiment Simulation va bam Play. Xem trace Java:
   [V8 20AGV ANIMATION] LOADED: ...desktop_brain.pt
   [V8 20AGV ANIMATION] ... role=routing ...
   AnyLogic se hien 20 AGV chay theo action cua policy.

Luu y: Desktop animation la run moi, KHONG phai replay episode 342.
Bo nao PyTorch chay trong Python duoc AnyLogic goi qua Pypeline, khong
nap truc tiep .pt trong AnyLogic. Hoat dong inference, khong train tiep.

Neu phan mem bao ModuleNotFoundError / khong thay policy, kiem tra duong
dan Pypeline/ten module/Python exe. Neu bao Graph fingerprint mismatch,
model .alp hien tai co graph khac voi model da train.
