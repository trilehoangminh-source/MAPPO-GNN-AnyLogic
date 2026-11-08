DANH GIA V8 EP304 VA V11 SAU TRAIN — 20 AGV, 5 HELD-OUT SEEDS

Giai nen ZIP vao goc C:\MARL_AnyLogic\MAPPO_NEW\thesis
-> them 2 file scripts/evaluate_v11_milestones.ps1 va scripts/compare_v8_v11_heldout.py
KHONG ghi de code V8/V11; KHONG can dat lai checkpoint.

STOP TRAIN, bao dam AnyLogic port 51881 chua duoc su dung.
Trong VS Code Terminal PowerShell dang o thu muc thesis:

1) Chi V8 EP304 vs V11 checkpoint moi nhat:
   & .\scripts\evaluate_v11_milestones.ps1

2) V8 EP304 vs V11 snapshot EP424,464,504,524 va checkpoint moi nhat:
   & .\scripts\evaluate_v11_milestones.ps1 -WithMilestones

3) Doc lai bang ket qua KHONG mo AnyLogic:
   python .\scripts\compare_v8_v11_heldout.py --directory .\mappo_runs\V11_CONGESTION_FROM_V8_EP304\eval_5seeds

KET QUA:
  mappo_runs/V11_CONGESTION_FROM_V8_EP304/eval_5seeds/heldout_*.jsonl
  mappo_runs/V11_CONGESTION_FROM_V8_EP304/eval_5seeds/heldout_comparison.csv

Script tu lay EP cuoi tu state.json, sao chep rolling checkpoint thanh
  eval_frozen_ep_XXXX.pt de giu lai ban danh gia, khong ghi de.
Neu output da ton tai, script se skip de bao ve ket qua (khong ghi de).
Khi muon evaluate lai cung EP nhung khac setup, dung output moi / doi ten output cu
sau khi sao luu. Dung 1800 sim-seconds, 5 seeds 900001..900005,
Dijkstra guide strength 1.5 va V8 reward JobDone +2.

CHU Y: 5 SEEDS chi la danh gia ban dau; nen danh gia 20+ seeds de bao cao
khoa hoc va giu tap test rieng neu co su dung seeds de chon checkpoint.
