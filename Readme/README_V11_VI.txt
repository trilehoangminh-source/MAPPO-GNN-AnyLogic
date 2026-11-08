V11 — Congestion-Aware Residual Edge-GAT (V8 EP304) — 20 AGV
==========================================================

GÓI PATCH CHỈ GỒM FILE MỚI. Chép cả thư mục thesis/, scripts/, tests/ vào
ROOT C:\MARL_AnyLogic\MAPPO_NEW\thesis. Không chép đè file V8/V10.

KHÁC BIỆT SO VỚI V8
- Routing: giữ V8 Residual Edge-GAT, thêm 6 kênh động từ actor observation:
  edge_occupancy, predicted_flow (Dijkstra đường ngắn nhất),
  upstream_waiting, downstream_load, signed_pressure, opposite_flow.
- 3 kênh tĩnh [edge_length,dx,dy] và node occupancy giữ nguyên.
- Dispatch Front/Rear: V8 MLP, KHÔNG Transformer.
- Routing Dijkstra logit guide strength 1.5; action mask; PPO V7 stable;
  reward JobDone +2; 20 AGV; train environment giữ 1800s.
- Ban đầu cả nhánh score/value traffic là số 0: V11 logits = V8 EP304 logits.
- Default TRAIN CHỈ ROUTING (MLP dispatch giữ nguyên trọng số EP304).
  Muốn train đồng thời Dispatch MLP thì chọn --train-roles all.
- Lưu ý: queue/pressure là PROXY từ observation, không phải dữ liệu cảm biến
  hay thời gian đặt chỗ / deadlock detection. Các trường của Java phải đúng
  nghĩa before real run. Chưa kiểm thử end-to-end AnyLogic tại đây.

QUAN TRỌNG: CHECKPOINT EP304
- Dùng snapshot_ep_0304.pt, checkpoint_ep_0304.pt, hoặc rolling_slot
  khi state.json cho thấy đúng last_episode=304.
- KHÔNG được nhầm checkpoint EP342 hoặc EP477 thành EP304!
- Mặc định --source-episode 304; có thể đổi sang episode khác bằng tham số này.
- Nếu snapshot EP304 nằm trong thư mục run V8 khác, dùng --source-run phù hợp.
- Nếu snapshot nằm ngoài source-run, cung cấp --source-checkpoint đường dẫn đầy đủ.
- Không đổi tên EP304 thành EP300, không chọn checkpoint của bản V9/V10.

1. TEST OFFLINE (mở PowerShell tại root của project):
  .\.venv311\Scripts\python.exe -m unittest discover -s tests -p "test_v11_traffic_offline.py" -v

2. PRE-FLIGHT (kiểm tra cấu hình, EP304 checkpoint và thời gian):
  .\.venv311\Scripts\python.exe -u .\scripts\train_v11_congestion.py --source-episode 304 --dry-run

3. TRAIN ĐẾN 08:30 VIỆT NAM SÁNG TỚI (mốc 08:30 sắp đến; script tự dừng):
  .\.venv311\Scripts\python.exe -u .\scripts\train_v11_congestion.py --source-episode 304 --device cuda:0 --train-roles routing --max-minibatch 32

Nếu EP304 snapshot ở nơi khác, thêm:
  --source-checkpoint "C:\duong_dan\snapshot_ep_0304.pt" --source-episode 304

Để đặt CHÍNH XÁC mốc đích thay vì mặc định 08:30 sắp tới:
  --end-at "2026-11-01T08:30:00+07:00"
Ví dụ này phù hợp chạy từ đêm 31/10 sang sáng 01/11/2026;
chọn ngày khác nếu thực tế đã qua mốc đó.

4. RESUME AN TOÀN (sau khi dừng giữa chừng):
  .\.venv311\Scripts\python.exe -u .\scripts\train_v11_congestion.py --source-episode 304 --device cuda:0 --train-roles routing --max-minibatch 32 --resume

5. EVALUATE 5 SEED sau train:
  .\.venv311\Scripts\python.exe -u .\scripts\evaluate_v11_congestion.py --device cuda:0 --seeds 900001 900002 900003 900004 900005 --sim-seconds 1800

OUTPUT:
  mappo_runs/V11_CONGESTION_FROM_V8_EP304/
    experiment.json, state.json, metrics.jsonl
    rolling_slot_0.pt, rolling_slot_1.pt
    snapshot_ep_XXXX.pt, latest_v11.pt
    heldout_*.jsonl

Chú ý: checkpoint và dữ liệu V8 được ĐỌC, tuyệt đối không được GHI ĐÈ.
Lưu nguồn checkpoint khác tên trạng thái qua episode cần đặt đúng.
Tất cả tham số nguồn không thay đổi khi resume (để đảm bảo reproducibility).

Bài báo cơ sở:
- Chen et al. (2024), Traffic Flow Optimisation for Lifelong MAPF, AAAI.
  https://doi.org/10.1609/aaai.v38i18.30054
- PyTorch model warm-start:
  https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html

CHECK SOURCE EP304 TRONG POWERSHELL:
  Get-ChildItem .\mappo_runs -Recurse -File -Filter "*0304*.pt" | Select-Object FullName,Length
  Get-ChildItem .\mappo_runs -Recurse -File -Filter "state.json" | ForEach-Object { $f = Get-Content $_.FullName -Raw | ConvertFrom-Json; if ($f.last_episode -eq 304) { "$( $_.FullName ) -> $( $f.checkpoint )" } }

Lưu ý: trước khi chạy thật trên Windows, phải qua --dry-run và test offline;
code chưa được kiểm tra end-to-end với AnyLogic ở môi trường người dùng.
