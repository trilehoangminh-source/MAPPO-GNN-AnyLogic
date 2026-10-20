# B3-GNN: Directed GNN Routing + Dijkstra + MAPPO / CTDE

**Bản bổ sung độc lập cho repository** `trilehoangminh-source/MAPPO-GNN-AnyLogic`, ngày 17/10/2026.

## 1. Nội dung thay đổi / không thay đổi

- **Routing Actor mới, GNN thực sự**: xây dựng đồ thị có hướng từ `env.agent_hive.nodes` và `env.agent_hive.paths` sau `env.reset()`. Mỗi layer tổng hợp embedding của **các node kế tiếp theo cạnh có hướng** (message passing), không chỉ dùng `pooled` hoặc đổi tên tham số.
- Node features động gồm toạ độ node, khoảng cách tương đối đến AGV/đích, tín hiệu AGV xuất hiện tại node. Tất cả dữ liệu động lấy từ `observation['agvs']` actor nhận được (không trộn `get_mappo_state()` của critic).
- Action head tạo **logit riêng** cho Stop (0) và từng cạnh ứng viên (1–4) dựa trên embedding node tiếp theo; hard action mask vẫn áp dụng cuối.
- **Dijkstra guidance** vẫn dùng `GuidedMAPPO`, `GuidedEventCollector` và `routing_guidance_v121.py` của B3 cũ, cộng một **logit prior mềm** và lưu nó vào observation để PPO dùng lại cùng một distribution.
- **Dispatching** tiếp tục cấu hình `MAPPO_DISPATCH_MASK_MODE=operation_first` và backbone `pooled` như B2/B3. Critic CTDE, GAE, PPO clipping, reward, logic alias/reward của collector giữ nguyên.
- Không sửa `algorithm.py`, `networks.py`, `behaviors_v121.py`, AnyLogic Java, các script cũ hoặc checkpoint. Mọi output B3-GNN dùng folder mới.

### Cấu trúc bổ sung

```text
thesis/
  mappo/
    gnn_routing_B3_v121.py              # GraphSpec, DirectedMP, GraphRoutingActor, nâng cấp GuidedMAPPO
scripts/
  check_B3_GNN_setup_v121.py             # Kiểm tra import + hook B2 (không chạy Java)
  train_mappo_long_B3_GNN_v121.py        # Train mới, checkpoint riêng
  evaluate_mappo_long_B3_GNN_v121.py     # Held-out greedy evaluation, không train
 tests/
  test_gnn_B3_v121.py                    # Tests mask, gradient, graph direction, PPO logprob parity
```

## 2. Cách cài trên Windows

1. **Sao lưu commit hiện tại** (nếu có thay đổi chưa commit, commit riêng). Giải nén `B3_GNN_patch.zip` vào **thư mục gốc** `C:\MARL_AnyLogic\MAPPO_NEW\thesis` (nơi có `scripts/` và thư mục con `thesis/`). File trong ZIP đều mang **tên mới**, không ghi đè module cũ.
2. Mở VS Code Terminal tại thư mục gốc; kích hoạt venv `.venv311` như bình thường.
3. Chạy lần lượt:

```powershell
python scripts/check_B3_GNN_setup_v121.py
python tests/test_gnn_B3_v121.py
```

**Điều kiện tiên quyết:** project phải đã có các file B2/B3 từ trước: `thesis/mappo/algorithm.py`, `networks.py`, `collector_stop_ab.py`, `routing_guidance_v121.py`, `scripts/train_mappo_long_B3_v121.py`, `thesis/envs/behaviors_v121.py` có hook `operation_first`. Nếu chỉ có source gốc Tobias, patch này không đủ để train.

## 3. Smoke 2 episodes: kiểm thử với mô phỏng AnyLogic thật

```powershell
python -u scripts/train_mappo_long_B3_GNN_v121.py --model envs/Matrix_critic_fresh.zip --port 51690 --episodes 2 --sim-seconds 720 --seed 43 --hidden 32 --graph-layers 2 --routing-guide-strength 1.5 --output mappo_runs/B3_GNN_smoke_seed43
```

- **Bắt buộc:** `envs/Matrix_critic_fresh.zip` phải tồn tại ở ổ đĩa cục bộ. Trong GitHub repository hiện tại, `.gitignore` có thể ẩn các ZIP này; project clone mới **không tự có AnyLogic model**.
- Trong terminal cần thấy `[B3-GNN] REAL graph actor: nodes=... edges=...`, tiếp theo `[B3-GNN EP ...]` với `guide>0`, `closed>0`, `mask_violations=0`.
- Nếu graph rỗng, không match được, hoặc graph khác nhau giữa các episode, script dừng rõ ràng; **không thay thế bằng pooled âm thầm**.
- Kết quả nằm trong thư mục `mappo_runs/B3_GNN_smoke_seed43/`; không ghi vào `B2_operation_first_50ep` hay B3 cũ.
- Nếu output folder đã tồn tại và có dữ liệu, **chọn tên output khác**, script từ chối ghi đè. Đây là fresh-only trainer, **không hỗ trợ resume**.

## 4. Sau khi smoke PASS: train 50 episodes

```powershell
python -u scripts/train_mappo_long_B3_GNN_v121.py --model envs/Matrix_critic_fresh.zip --port 51692 --episodes 50 --sim-seconds 720 --seed 43 --hidden 32 --graph-layers 2 --routing-guide-strength 1.5 --output mappo_runs/B3_GNN_seed43_50ep
```

Nếu chạy thành công, checkpoint cuối là:

```text
mappo_runs/B3_GNN_seed43_50ep/checkpoint_ep_0050.pt
```

**Không load B3-GNN checkpoint bằng script B3 pooled cũ.** Architecture và weights khác nhau. Evaluator GNN đọc `run_config.json`, kiểm tra version và dấu vân tay graph trước khi load.

## 5. Held-out evaluation

```powershell
python -u scripts/evaluate_mappo_long_B3_GNN_v121.py --checkpoint mappo_runs/B3_GNN_seed43_50ep/checkpoint_ep_0050.pt --seeds 1001 1002 1003 --sim-seconds 3600 --port 51693 --output mappo_runs/B3_GNN_seed43_50ep/evaluation_3600.jsonl
```

So sánh B2 và B3-GNN trên cùng seeds, cùng sim-seconds, `fleetsize` và AnyLogic ZIP. KPI: `jobDone`, `targetReached`, `wrongTarget`, `block`, `liveLock`, reward; không chỉ đếm `guided` hay `selected_guide`. Seed 43 chỉ là một training run: chưa kết luận cải tiến khoa học nếu thiếu lặp lại đa seed và kiểm định thống kê.

## 6. Nếu gặp lỗi

| Thông báo | Ý nghĩa / hành động |
|---|---|
| Missing B2 operation_first hook | B2 chưa được cài đúng; **không** sửa bừa `behaviors_v121.py` |
| LIVE HiveContext.nodes and paths | Graph chưa có sau reset; gửi log và kết quả kiểm tra model ZIP |
| graph changed during train | Node/cạnh AnyLogic thay đổi giữa episode; cần xử lý dynamic graph trước khi train lâu |
| No Dijkstra guidance | Kiểm tra mạng đường có hướng và toạ độ AGV/Target với `DirectedGraphGuide` |
| Missing model ZIP | Giữ bản model ZIP cũ trên PC; `.gitignore` không upload ZIP lên Github |
| Output directory nonempty | Đặt output mới; không xóa checkpoint B2/B3 cũ |
| OOM / CPU quá chậm | Giảm `--hidden 16`; chạy smoke; graph dense message passing có thể tốn CPU |

## 7. Trạng thái kiểm thử và giới hạn

**Đã kiểm tra ở môi trường tạo bản vá:** Python compile và **offline synthetic graph smoke**: forward/masked actions, directed GNN gradient >0, PPO initial logprob ratio = 1.0. **Chưa chạy AnyLogic Java hoặc chính xác toàn bộ test integration với source GitHub trên Windows** vì không có live AnyLogic model và không mount được checkout phiên bản MAPPO mới trong môi trường kiểm tra. Script `tests/test_gnn_B3_v121.py` cần được chạy ở máy ông, rồi smoke 2 episode trước khi kết luận tương thích thực tế.

Đây là **một thiết kế thử nghiệm**, không chứng minh thuật toán tối ưu hơn bản pooled hoặc Tobias gốc. GNN học biểu diễn mạng đường, nhưng chưa thay thế thuật toán tránh va chạm riêng hay giải bài toán phối hợp đa AGV toàn cục.
