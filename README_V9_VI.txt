MAPPO V9 — TRAIN TU DAU: Plain GAT + Reward JobDone +10
=======================================================

GIAI NEN: CHO CAC FOLDER 'scripts' VA 'thesis' VAO PROJECT ROOT:
C:\MARL_AnyLogic\MAPPO_NEW\thesis

CAC FILE DUOC THEM MOI (KHONG GHI DE V8):
  thesis/mappo/plain_gat_routing_v9.py
  scripts/train_V9_plainGAT_jobdone10_5h.py
  scripts/evaluate_V9_plainGAT_jobdone10.py

YEU CAU:
  - Da co envs/Matrix_RL_debug.zip voi AnyLogic timeout 0.097
  - Da co cac modules V7/B3-GNN, PyTorch, Alpyne trong project
  - NVIDA CUDA san sang (hoac doi --device cpu)

MO POWERSHELL TRONG VS CODE:
  cd C:\MARL_AnyLogic\MAPPO_NEW\thesis

1. KIEM TRA TRUOC (KHONG TRAIN):
  python -u .\scripts\train_V9_plainGAT_jobdone10_5h.py --dry-run

2. BAT DAU FRESH TRAIN TOI DA 5 TIENG:
  python -u .\scripts\train_V9_plainGAT_jobdone10_5h.py --hours 5 --device cuda:0

3. DA CHAY DANG DO VA CAN TIEP TUC THEM (5 GIO LA NGAN SACH MOI):
  python -u .\scripts\train_V9_plainGAT_jobdone10_5h.py --resume --hours 5 --device cuda:0

4. KIEM TRA SAU TRAIN:
  Get-Content .\mappo_runs\V9_PLAIN_GAT_JOBDONE10_FRESH20_5H\state.json

5. DANH GIA 5 HELD-OUT SEEDS:
  python -u .\scripts\evaluate_V9_plainGAT_jobdone10.py --checkpoint .\mappo_runs\V9_PLAIN_GAT_JOBDONE10_FRESH20_5H\latest_mappo.pt --seeds 900001 900002 900003 900004 900005 --sim-seconds 1800 --device cuda:0 --output .\mappo_runs\V9_PLAIN_GAT_JOBDONE10_FRESH20_5H\heldout_5seeds.jsonl

CANH BAO:
  - V9 la mo hinh NGHIEM MOI, KHONG nap checkpoint 205/208.
  - GAT chi dung node features + directed arcs, no residual/edge attributes.
  - JobDone+10 duoc gan rieng cho V9 luc tao environment.
  - Dijkstra prior=1.5, MAPPO PPO stable, 20 AGV, seed huan luyen=43.
  - Mot job completion co the duoc broadcast den nhieu Dispatch agent;
    +10 khong tuong duong +10 trong delivered_reward tong.
  - Vi doi CA GAT va REWARD, khong the quy ket su khac biet chi do reward.
  - Neu ban da xoa cac file B3/V7 can thiet, khoi phuc tu GitHub truoc.

DON NOT DELETE checkpoint .pt: gitignore khong backup no len GitHub.
