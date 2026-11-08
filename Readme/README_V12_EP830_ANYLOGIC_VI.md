# V12 EP830 — Xem animation 20 AGV trực tiếp trong AnyLogic 8.9

## Cần hiểu

- Checkpoint `snapshot_ep_0830.pt` **không** thể nhập trực tiếp vào ô `RL_pypeline_name` trong AnyLogic.
- Ô này là **tên module Python**; module mới `matrix_mappo_v12_20agv.py` sẽ nạp trọng số EP830 và suy luận theo chính sách đã học.
- Không train, không cập nhật weight/optimizer, không Dijkstra, không short-path predicted flow.
- Được xây dựng từ **connector V8 20AGV đang có trong GitHub của bạn**, giữ format giao tiếp Java `get_config`/`get_action` và action mask.
- Chưa kiểm thử end-to-end trên AnyLogic Windows; bắt đầu bằng kiểm tra các file và graph.

## 1. Giải nén

Giải nén ZIP vào `C:\MARL_AnyLogic\MAPPO_NEW\thesis` để nhận:

- `thesis\pypeline_connectors\matrix_mappo_v12_20agv.py` (file mới)
- `scripts\check_v12_ep830_animation.py` (file mới)

Không ghi đè V8/V11/V12 runner; các checkpoint không bị thay đổi.

## 2. Kiểm tra

```powershell
cd C:\MARL_AnyLogic\MAPPO_NEW\thesis
.\.venv311\Scripts\python.exe .\scripts\check_v12_ep830_animation.py
```

Trả về `OK: V12 3 role actor weights and traffic branches; no next_hop` mới tiếp tục.

## 3. Mở bản AnyLogic có animation

Tại `Main`:

- `RL_pypeline_path` = `"C:/MARL_AnyLogic/MAPPO_NEW/thesis"` (thư mục ROOT)
- `RL_pypeline_name` = `"matrix_mappo_v12_20agv"` (thay cho `"matrix_mappo_v8_20agv"`)
- `runmode` = `3`
- Bảo đảm Pypeline `py` chạy Python `C:/MARL_AnyLogic/MAPPO_NEW/thesis/.venv311/Scripts/python.exe`
- Chạy experiment **Simulation: Main**, KHÔNG chạy `RLExperiment: Main`.

Khi chạy, Python phải in: `[V12 EP830 ANIMATION] LOADED ... snapshot_ep_0830.pt`. Đồng thời thấy các log `[V12 EP830 ANIMATION] ... action=...`. Nếu chỉ in `V8 20AGV ANIMATION`, dừng: đang dùng connector V8.

## 4. Thời lượng và tình trạng mô hình

Đặt Simulation `Stop at specified time` tương đương 1.800 giây theo đơn vị thời gian mô hình, hoặc Stop=Never nếu muốn xem tiếp. Chỉ đổi Stop time không đổi checkpoint. Cần mô hình desktop có cùng graph (105 nodes/328 directed edges) và mapping với dữ liệu training; connector sẽ kiểm tra fingerprint.

## 5. Chuyển sang checkpoint khác

Mặc định dùng `snapshot_ep_0830.pt`. Muốn dùng checkpoint khác, sau khi đóng simulation, đặt biến môi trường hệ thống `V12_20AGV_DESKTOP_CHECKPOINT` cho tiến trình Python AnyLogic, hoặc sửa biến `V12_CHECKPOINT` trong module mới. Sau khi thay, khởi động lại AnyLogic/phiên Python để tránh dùng trọng số đã cache trong RAM.

**Chú ý:** EP30 nếu muốn theo nghĩa đen thì không phải cùng một checkpoint với EP830; V12 FULL No-Dijkstra bắt đầu từ EP530, do đó không có V12 EP30 trong thư mục run này.

## 6. Lỗi thường gặp

- `ModuleNotFoundError`: kiểm tra đúng thư mục `thesis/pypeline_connectors`, `RL_pypeline_path` và Python Pypeline.
- `Graph fingerprint mismatch`: model `.alp` desktop khác topology/model được dùng khi train. Dừng lại, không tắt check.
- `FileNotFoundError`: file EP830 hoặc checkpoint nguồn V8 từ manifest chưa có tại đường dẫn.
- Trường hợp Java hiển thị animation mà không có `[V12 EP830 ANIMATION] LOADED` là chưa thể khẳng định đang chạy EP830.
- Không chạy trainer và animation đồng thời nếu máy thiếu RAM/CPU; nếu đã có phiên run mode 3, không chạy evaluator Alpyne ở cổng khác để kiểm chứng cùng lúc.
