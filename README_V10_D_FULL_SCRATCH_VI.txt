V10-D FULL SCRATCH — TẤT CẢ ACTOR/CRITIC KHỞI TẠO MỚI, KHÔNG NẠP V8 .pt
============================================================================

1) KHÁC V10-D CŨ
  V10-D cũ: Routing Actor và Routing Critic lấy từ pretrained V8 rồi đóng băng.
  V10-D FULL SCRATCH: Routing Residual Edge-GAT, Routing Critic, Dispatch Front
  Pure Transformer Actor+Critic, Dispatch Rear Pure Transformer Actor+Critic,
  và optimizer / ValueNorm đều khởi tạo mới. CẢ 3 ROLES ĐƯỢC TRAIN.

  Script chỉ đọc mappo_runs/V8_RESIDUAL_EDGEGAT_JOBDONE2_FROM205/run_config.json
  để tái sử dụng cấu hình môi trường, graph, PPO và đường dẫn AnyLogic ZIP.
  KHÔNG ĐỌC BẤT KỲ CHECKPOINT .pt NÀO CỦA V8.

2) CÀI ĐẶT
  Giải nén vào C:\MARL_AnyLogic\MAPPO_NEW\thesis, SAU khi đã cài
  bộ V10-D Pure Transformer trước đó. Chỉ có file MỚI:
    scripts/v10_full_scratch_common.py
    scripts/train_V10_full_scratch.py
    scripts/evaluate_V10_full_scratch.py
    tests/test_v10_full_scratch_preflight.py

3) PRE-FLIGHT
  cd C:\MARL_AnyLogic\MAPPO_NEW\thesis
  .\.venv311\Scripts\python.exe -m unittest discover -s tests -p "test_v10_full_scratch_preflight.py" -v
  .\.venv311\Scripts\python.exe -u .\scripts\train_V10_full_scratch.py --dry-run

4) SMOKE TEST FULL SCRATCH 2 EP; MẶC ĐỊNH MINIBATCH 64 DO EDGE-GAT CŨ TỪNG OOM
  .\.venv311\Scripts\python.exe -u .\scripts\train_V10_full_scratch.py `
    --device cuda:0 --hours 0.5 --episodes 2 `
    --layers 2 --heads 2 --hidden 32 --max-minibatch 64

  Muốn thử max minibatch 128, chỉ đổi --max-minibatch 128 ngay TỪ ĐẦU,
  KHÔNG tăng khi --resume. Nếu OOM, dùng output KHÁC hoặc xóa đúng run
  scratch chưa train; không đổi minibatch nửa chừng.

5) TRAIN TIẾP (TỔNG 50 EP)
  .\.venv311\Scripts\python.exe -u .\scripts\train_V10_full_scratch.py `
    --resume --device cuda:0 --hours 5 --episodes 50 `
    --layers 2 --heads 2 --hidden 32 --max-minibatch 64

6) EVALUATION SAME SEEDS V8 EP342
  .\.venv311\Scripts\python.exe -u .\scripts\evaluate_V10_full_scratch.py `
    --run .\mappo_runs\V10_D_FULL_SCRATCH_2LAYERS `
    --seeds 900001 900002 900003 900004 900005 `
    --sim-seconds 1800 --device cuda:0 --port 51881

7) LƯU TRỮ
   mappo_runs/V10_D_FULL_SCRATCH_2LAYERS/
     checkpoint_ep_0000.pt  (random weights, không phải episode có dữ liệu)
     state.json             (ban đầu last_episode=0)
     metrics.jsonl
     rolling_slot_0.pt / rolling_slot_1.pt
     snapshot_ep_0010.pt ...
     experiment.json
     run_config_blueprint.json

8) GHI CHÚ NGHIÊN CỨU
  - Dijkstra guide strength 1.5 được GIỮ LẠI, là heuristic chứ không phải
    pretrained policy. Reward JobDone +2, stop mask và operation_first mask.
  - EP1 là episode đầu tiên thu thập dữ liệu; checkpoint EP0 là trọng số ban đầu.
  - Full scratch 20 AGV là bài toán khó hơn nhiều so với warm-start Routing.
  - Để so sánh MLP vs Transformer công bằng, bổ sung MLP full-scratch với cùng
    seeds, horizon, PPO update budget và reward.
  - Model chỉ được kiểm tra offline/preflight, chưa chạy AnyLogic end-to-end.
