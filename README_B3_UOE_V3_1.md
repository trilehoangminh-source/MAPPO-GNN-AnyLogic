# B3-UOE-V3.1 — Routing Decision Trace & Congestion Diagnostics

**Ngày 21/10/2026.** Tiếp nối B3-UOE-V2 đã chạy được trên máy Windows của ông. Đây là bản **CHẨN ĐOÁN/ĐÁNH GIÁ KHÔNG CAN THIỆP POLICY**, chưa đổi kiến trúc hoặc reward. **Đừng đánh đồng V3.1 với bản Traffic-Aware GAT đã huấn luyện.**

## 1. Đúng kiến trúc đã thống nhất

- **Routing:** Directed GAT hoặc MP-GNN hiện có của V2, với Dijkstra soft guidance và MAPPO/CTDE.
- **Dispatching:** giữ **B2 Operation-first, MLP/pooled, KHÔNG GNN**.
- Chỉ đọc checkpoint V2, **không train**, không ghi đè checkpoint, không đổi Java/AnyLogic, action mask, GAE, PPO, neural weights hay reward.
- Lấy ý tưởng về kiểm toán quá trình đánh giá từ **EPyMARL**, theo dõi task-vs-routing từ **TA-RWARE**; không copy code bên ngoài.

## 2. File mới (5 file)

- `thesis/mappo/routing_trace_uoe_v3.py`: đọc graph thật từ `GraphSpec`, ghi một dòng JSON cho từng lựa chọn Routing, với action thực do policy đưa ra, không thay đổi action.
- `scripts/evaluate_B3_UOE_v3_trace.py`: nạp checkpoint V2, chạy thử nghiệm đánh giá (evaluation) 3600 s/seed.
- `scripts/analyze_B3_UOE_v3_trace.py`: thống kê node thường xuất hiện trong quyết định routing, bất đồng Dijkstra, mật độ AGV gần node và các khoảng thời gian bộ đếm tắc nghẽn tăng.
- `scripts/verify_B3_UOE_v3_noninvasive.py`: so sánh KPI của replay V2 với V3 theo cùng seed/checkpoint.
- `tests/test_uoe_v3_trace_offline.py`: kiểm tra graph có hướng, tính trong suốt của trace, restore wrapper, công cụ phân tích.

## 3. Cài đặt không phá bản cũ

1. Chỉ giải nén **nội dung ZIP** vào thư mục: `C:\MARL_AnyLogic\MAPPO_NEW\thesis`.
2. Đứng trong project root và kích hoạt `.venv311` như trước.
3. Xác nhận các script V2 sẵn có: `scripts/evaluate_B3_GNN_UOE_v2.py`, `scripts/train_B3_GNN_UOE_v2.py`.
4. Chạy offline test:

```powershell
python tests/test_uoe_v3_trace_offline.py
```

Dòng mong đợi: `PASS V3.1 offline tests ...`.

## 4. Chạy một seed (nên bắt đầu bằng 1001)

Chỉ dùng **checkpoint V2 GAT đã có** và cấu hình cùng với lần đánh giá V2 trước:

```powershell
python -u scripts/evaluate_B3_UOE_v3_trace.py --checkpoint mappo_runs/B3_UOE_v2_GAT_smoke_seed43/checkpoint_ep_0002.pt --seeds 1001 --sim-seconds 3600 --port 51705 --output mappo_runs/B3_UOE_V3_1_trace_seed1001
```

Các file kết quả nằm trong thư mục output, không ảnh hưởng `mappo_runs/B3_UOE_v2_GAT_smoke_seed43/`:

- `evaluation_v3_diagnostic.jsonl`: các KPI gốc AnyLogic + audit mapping.
- `routing_trace_seed1001.jsonl`: từng quyết định Routing, node/target, action thực tế, các move hợp lệ, guide, khoảng cách, local-traffic proxy, cumulative statistics.
- `routing_trace_summary_seed1001.json`: tổng hợp các chỉ số chính.
- `diagnostic_run_config.json`: checkpoint hash, graph fingerprint, các seed, thời gian, version.

Sau đó tổng hợp:

```powershell
python scripts/analyze_B3_UOE_v3_trace.py --folder mappo_runs/B3_UOE_V3_1_trace_seed1001
```

Nếu file V2 trước còn, so sánh replay (rất nên làm):

```powershell
python scripts/verify_B3_UOE_v3_noninvasive.py --v2 mappo_runs/B3_UOE_v2_GAT_smoke_seed43/evaluation_3600_seed1001.jsonl --v3 mappo_runs/B3_UOE_V3_1_trace_seed1001/evaluation_v3_diagnostic.jsonl
```

Nếu KPI khác, **DỪNG** kiểm tra cùng checkpoint, cùng seed, sim-seconds, model ZIP, mô phỏng AnyLogic có tái lập hay không; không vội kết luận do trace.

## 5. Các trường trong trace

| Trường | Ý nghĩa |
|---|---|
| `time_s`, `physical_agv` | thời điểm mô phỏng, slot AGV đang quyết định |
| `current_node`, `target_node`, `next_node` | node gốc, đích, ứng viên đã chọn |
| `action`, `valid_actions` | action thực và legal mask (0=Stop, 1–4 các node kế theo thứ tự Tobias) |
| `guide_actions`, `selected_is_guided` | các action được Dijkstra ưu tiên, action có trùng không |
| `directed_remaining_before/after` | shortest remaining distance trên graph **có hướng** tới đích |
| `remaining_change` | remaining_after - remaining_before; âm là đang tới gần đích theo shortest remaining distance |
| `other_agvs_last_node_nearby` | số AGV *quan sát được ở last-node* trong bán kính 6m mặc định, **không phải đo khoảng cách thực tế liên tục** |
| `cumulative_stats` | KPI tích lũy ở thời điểm quan sát |
| `stats_increase_since_previous_routing_observation` | mức tăng bộ đếm từ lần quan sát Routing trước, **không phải ảnh hưởng nhân quả của action vừa chọn** |

Nếu biết tính bằng mét của mô hình khác, chỉnh radius với `--radius 6.0` phù hợp; phải giữ cùng radius giữa các thí nghiệm nếu đem so sánh.

**Không được diễn giải** “node có nhiều block-counter intervals” thành node gây block. Nếu bộ đếm tăng có thể là do AGV khác, Dispatching, và các sự kiện AnyLogic xen giữa hai lần đo.

## 6. Chạy ba seed sau khi test seed 1001

Dùng thư mục output mới và port riêng, ví dụ:

```powershell
python -u scripts/evaluate_B3_UOE_v3_trace.py --checkpoint mappo_runs/B3_UOE_v2_GAT_smoke_seed43/checkpoint_ep_0002.pt --seeds 1001 1002 1003 --sim-seconds 3600 --port 51706 --output mappo_runs/B3_UOE_V3_1_trace_3seeds
python scripts/analyze_B3_UOE_v3_trace.py --folder mappo_runs/B3_UOE_V3_1_trace_3seeds
```

**Chưa train 50 episodes** ở V3.1. Khi xác định AGV thường lệch hướng ở node nào, khi nào tắc và dữ liệu giao thông actor đang thấy, V3.2 có thể THỬ bổ sung traffic-aware features/reward cho Routing GAT (không động Dispatching B2), kèm ablation cùng seed để đánh giá hiệu quả.

## 7. Cảnh báo còn tồn tại

`[WARNING] observation.networkcontext = None` là cảnh báo Alpyne từ schema/model AnyLogic cũ. V3.1 không thay Java schema; báo cáo graph mapping V2 đã đạt 100% không đủ để xem warning này đã hết. Các số KPI trung bình đã có trên 3 seed: `jobDone=5.667`, `block=124`, `liveLock=93.667`, thời gian 3600 s, vẫn cần baseline matched.

## 8. Nguồn

- Steinbacher, Wegmann, Freitag (2025), *Production control with Reinforcement Learning for a matrix-structured production system*, §3.2.4 (Routing GAT vs Dispatching linear/attention), §3.3 (train riêng trước), §4 (block/livelock).
- https://github.com/uoe-agents/epymarl
- https://github.com/uoe-agents/task-assignment-robotic-warehouse
- https://github.com/trilehoangminh-source/MAPPO-GNN-AnyLogic

## 9. Đã kiểm tra trong môi trường tạo patch

- `python tests/test_uoe_v3_trace_offline.py`: PASS
- `python -m compileall -q thesis scripts tests`: PASS
- Không thể chạy tích hợp AnyLogic vì môi trường tạo patch không có phiên Java model đang thực thi trên máy ông. Cần ông chạy 1 seed thật để xác thực tương thích hoàn chỉnh.
