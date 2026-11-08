# V12 FULL NO DIJKSTRA — không ưu tiên đường ngắn nhất, không dự đoán flow bằng đường ngắn nhất

## Mục đích

Tiếp tục huấn luyện **tất cả** Routing, Dispatch Front, Dispatch Rear actor + critic từ **checkpoint V11 được commit mới nhất**, đến **EP1000 tổng cộng**. V11 dùng Dijkstra theo 2 cách: (1) gợi ý action +1.5 logit, (2) `next_hop` để dự đoán `predicted_flow` và kênh liên quan. Bản V12 mới loại **CẢ HAI** khỏi pipeline quyết định và tính toán feature.

**Đây là bản thay thế cho gói V12 trước**, chứ không phải bản sửa bên trong một run V12 đã train. Dùng **folder kết quả mới** để tránh trộn hai định nghĩa observation feature: `mappo_runs/V12_FULL_NO_DIJKSTRA_FROM_V11`.

- Không tạo `DirectedGraphGuide` hay `routing_guide_bias`. Thay `actor.dist` trở lại phân phối action-mask gốc của mạng; thay `model.act` trở lại `MAPPO.act` thông thường.
- Không có `predicted_flow` từ `next_hop`/Dijkstra, không tạo shortest-path table, không suy đường đi tương lai, không đọc tương lai mô phỏng.
- **Giữ nguyên 6 chiều traffic đầu vào, không đổi các lớp trainable** `traffic_score`, `traffic_value`, Routing Edge-GAT và các actor/critic/optimizer. Load STRICT tất cả trọng số từ V11. Loại **duy nhất buffer không học** `v11_traffic.next_hop` khi chuyển từ file V11.
- Kênh traffic trong V12 **quan sát trực tiếp**: `edge_occupancy`, `observed_edge_intent`, `upstream_waiting`, `downstream_load`, `pressure`, `observed_contraflow`. `observed_edge_intent` là AGV có waypoint `last -> next` được quan sát chứ không phải dự đoán đường đi đến target. Các kênh waiting/load chỉ là proxy, không chứng minh xe thực sự bị tắc.
- **Giữ tất cả trọng số V11**, bao gồm các weights cũ của flow feature nay đổi ý nghĩa. Do đó ban đầu policy có thể giảm hiệu suất (distribution shift); cần fine-tune thêm. Không reset Adam; các role train được (role không có dữ liệu trong một episode thì không có optimizer step tại episode đó).
- Chỉ can thiệp các hàm MAPPO Python. **Không tuyên bố xóa thuật toán điều hướng nội bộ của AnyLogic/Java** nếu mô hình AnyLogic còn dùng thuật toán riêng.

## Cài đặt

Dừng training và AnyLogic cũ (cổng 51880). Giải nén ZIP vào thư mục gốc:

`C:\MARL_AnyLogic\MAPPO_NEW\thesis`

Gói chứa duy nhất 6 file mới, không ghi đè V11 hay V12 trước:

```
scripts/train_v12_full_no_dijkstra.py
scripts/evaluate_v12_full_no_dijkstra.py
scripts/v12_clean_common.py
scripts/v12_clean_collector.py
thesis/mappo/congestion_edgegat_v12_clean.py
tests/test_v12_full_no_dijkstra_offline.py
```

Cần có sẵn thư mục dự án chạy được V11 (đặc biệt `scripts/v11_common.py`, `thesis/mappo/edge_gat_routing_v4.py`, `thesis/mappo/algorithm.py`, `mappo_runs/V11_CONGESTION_FROM_V8_EP304/`, source V8 EP304, model AnyLogic ZIP và `.venv311`).

## Bước 1 — Kiểm tra chưa đụng AnyLogic

```powershell
cd C:\MARL_AnyLogic\MAPPO_NEW\thesis
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p "test_v12_full_no_dijkstra_offline.py" -v
.\.venv311\Scripts\python.exe -u .\scripts\train_v12_full_no_dijkstra.py --max-episodes 1000 --dry-run
```

Kết quả DRY RUN hiển thị `source V11 EP###` lấy từ `state.json`. Nếu là EP529, rollout V12 đầu tiên sẽ là EP530 và train đến EP1000. Không giả định V11 vẫn ở EP529 nếu ông đã chạy thêm. Bản V12 mới **không** tự kế thừa checkpoint từ V12 cũ (vì định nghĩa traffic khác). Khi lần đầu bắt đầu, dùng **checkpoint V11 mới nhất**.

## Bước 2 — Train đến EP1000 (không có giờ dừng)

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v12_full_no_dijkstra.py --device cuda:0 --max-minibatch 64 --max-episodes 1000
```

- `--max-episodes 1000`: mốc **episode tuyệt đối**, không phải 1000 episode mới.
- Port train mặc định 51880, 20 AGV, PPO V7 stable, JobDone reward +2, operation-first mask, graph/mask/audit như V11.
- Thư mục `mappo_runs/V12_FULL_NO_DIJKSTRA_FROM_V11/` chứa source V11 checkpoint copy, `experiment.json`, `state.json`, rolling slots, snapshots mỗi 10 EP, `metrics.jsonl`.
- Cấu hình giảm warning `alpyne.data` và `alpyne.sim` đặt ERROR trong script. Nếu máy quá tải, dừng train an toàn, xem checkpoint đã commit và resume. Đừng chạy song song AnyLogic eval/train trên máy quá tải.
- **Không chạy script V12 cũ** `train_v12_no_dijkstra.py` nữa nếu muốn thí nghiệm no-shortest-path.

## Bước 3 — Resume đúng run V12 mới

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v12_full_no_dijkstra.py --device cuda:0 --max-minibatch 64 --max-episodes 1000 --resume
```

KHÔNG dùng `--resume` ngay lần đầu. `--resume` không cho thay đổi thông số cấu hình lõi so với manifest. Không xóa `state.json`, `experiment.json`, `rolling_slot_*.pt`.

## Bước 4 — Evaluate trên held-out seeds 900001..900005

Sau khi dừng train và đảm bảo cổng 51881 trống:

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\evaluate_v12_full_no_dijkstra.py --device cuda:0 --port 51881 --seeds 900001 900002 900003 900004 900005 --sim-seconds 1800
```

Đánh giá snapshot EP600 nếu tồn tại:

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\evaluate_v12_full_no_dijkstra.py --device cuda:0 --port 51881 --checkpoint ".\mappo_runs\V12_FULL_NO_DIJKSTRA_FROM_V11\snapshot_ep_0600.pt" --seeds 900001 900002 900003 900004 900005 --sim-seconds 1800
```

So sánh held-out công bằng với V8, V11 và V12 cũ bằng đúng model/fleet/seed/sim-seconds. Ưu tiên `JobDone ↑`, `Block ↓`, `LiveLock ↓`, `Block time ↓`, `WrongTarget ↓`, cùng thống kê độ biến thiên. Training EP không đủ chứng minh hơn.

## Lưu ý về số đo và giả thuyết

V11 có thể dùng Dijkstra giúp AGV không đi lạc. V12 hoàn toàn không có guidance có thể tạo nhiều WrongTarget lúc đầu; điều đó là một kết quả nghiên cứu, không nhất thiết là lỗi. Các đặc trưng traffic V12 tính từ trạng thái hiện tại, không dự đoán tất cả tuyến đường các AGV sẽ chọn; vì thế có thể mất một phần năng lực dự đoán so với V11. Đây là **thí nghiệm đo giá trị của việc loại bỏ nguồn heuristic** thay vì giả định chắc chắn tốt hơn.

## Giới hạn kiểm tra

Đã kiểm tra cú pháp Python, 6 bài test ngoại tuyến (bao gồm migration Adam/weights). **Chưa chạy end-to-end với AnyLogic trên Windows của ông**. Khi chạy thật, builder sẽ kiểm tra graph fingerprint, kiểm tra checkpoint và bắt buộc phân phối Routing không còn wrapper guidance.
