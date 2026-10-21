# B3-UOE V3.2 — GAT Influence Audit (read-only)

## Mục tiêu

V3.1 trên seed 1001 cho thấy Routing có 1457/1457 quyết định đa lựa chọn vẫn theo Dijkstra, không có action Stop hợp lệ, và proxy `other_agvs_last_node_nearby` luôn bằng 7. V3.2 kiểm tra **GAT raw logits vs GAT + Dijkstra logits** trước khi dự tính thay architecture/hyperparameters. V3.2 không huấn luyện thêm và không can thiệp action của AnyLogic.

Bản code này là công cụ chẩn đoán tự viết; chỉ tham khảo cách đo lường từ EPyMARL và cách chia routing/task từ TA-RWARE, không sao chép mã nguồn của họ.

## Cài đặt

Giải nén **thẳng vào project gốc** `C:\MARL_AnyLogic\MAPPO_NEW\thesis`, giữ nguyên B2, B3, V2, V3.1, checkpoint và mô hình `.zip` AnyLogic. Ba file mới nằm trong `scripts/`, một file trong `tests/` và một file trong `thesis/mappo/`.

Kiểm tra offline:

```powershell
python tests/test_uoe_v3_2_influence_offline.py
```

Chạy evaluation **một seed (1001) 3600 s**, dùng checkpoint V2 đã có:

```powershell
python -u scripts/evaluate_B3_UOE_v3_2_influence.py --checkpoint mappo_runs/B3_UOE_v2_GAT_smoke_seed43/checkpoint_ep_0002.pt --seeds 1001 --sim-seconds 3600 --port 51710 --output mappo_runs/B3_UOE_V3_2_influence_seed1001
```

Phân tích:

```powershell
python scripts/analyze_B3_UOE_v3_2_influence.py --folder mappo_runs/B3_UOE_V3_2_influence_seed1001
```

Nếu đã chạy V3.1 seed 1001 trên cùng checkpoint và thời lượng, so sánh KPI:

```powershell
python scripts/verify_B3_UOE_v3_noninvasive.py --v2 mappo_runs/B3_UOE_V3_1_trace_seed1001/evaluation_v3_diagnostic.jsonl --v3 mappo_runs/B3_UOE_V3_2_influence_seed1001/evaluation_v3_2_influence.jsonl
```

Nếu so sánh không khớp, dừng để kiểm tra nondeterminism / phiên bản code, không diễn giải khác biệt là thuật toán cải thiện.

## Các trường then chốt

- `has_guide`: số quyết định Dijkstra có hướng đề xuất.
- `raw_top1_on_guide`: nếu **GAT tự nó** đã chọn cùng hướng, prior có thể không phải tác nhân làm đổi top1.
- `guidance_changes_top1`: số lần cộng prior Dijkstra **làm thay đổi** top1 trong cùng observation.
- `raw_guide_probability_mean` và `guided_guide_probability_mean`: đánh giá prior đang đẩy xác suất về hướng guide nhiều cỡ nào.
- `stop_available`: kiểm tra mask. Theo collector V2, `forbid_when_move` làm Stop không hợp lệ khi còn hướng di chuyển; không được suy luận GAT "không chịu dừng".
- `nearby_1m`, `nearby_2m`, `nearby_6m`, `some_same_last_node`, `first_three_other_xy`: đo chẩn đoán LAST-NODE XY từ observation actor; **không phải khoảng cách vật lý liên tục giữa các AGV**. Nếu cả 7 đều gần, kiểm tra tiếp tính đúng đắn của observation, phép quy đổi đơn vị và cách đưa tọa độ Other AGVs trước khi xây Traffic-Aware GAT.

## Kiến trúc và an toàn

- Routing: directed GAT hoặc MP-GNN theo V2, Dijkstra prior như cũ.
- Dispatching: B2 Operation-first **pooled/MLP**, hoàn toàn không thêm GNN.
- PPO/CTDE, reward, hard mask: giữ nguyên.
- Logits được tính **thêm** trong evaluation với `no_grad`, không gọi optimizer. Bộ audit so khớp log-probability để phát hiện khác biệt.
- V3.2 tạo file riêng và từ chối ghi đè output cũ.
- Không biến event counter `block`/`liveLock` thành nhãn nhân quả cho action cuối.
- **Chưa kiểm thử integration thật với AnyLogic trên máy người dùng.** Offline tests chỉ xác minh logic chẩn đoán và tính không can thiệp trên mô hình stub.
