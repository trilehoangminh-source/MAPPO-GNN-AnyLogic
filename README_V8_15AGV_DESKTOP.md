# Bộ nạp MAPPO V8 — 15 AGV vào AnyLogic Desktop

Lệnh train V8 đã tạo `mappo_runs/V8_15AGV_fromV7_5h/best_mappo.pt` và `run_config.json`. Script mới **không chỉnh file `matrix_mappo_v7.py`**, nên mô phỏng V7 (8 xe) vẫn được giữ để dự phòng.

## 1. Giải nén

Giải nén ZIP vào `C:\MARL_AnyLogic\MAPPO_NEW\thesis` với đúng cấu trúc thư mục. File mới:
- `thesis/pypeline_connectors/matrix_mappo_v8_15agv.py`
- `scripts/check_v8_desktop_15agv.py`

Yêu cầu: giữ các file từ bộ V7 Pypeline và hotfix cũ, gồm `thesis/pypeline_connectors/v7_inference.py`; không đổi cấu trúc model AnyLogic.

## 2. Kiểm tra OFFLINE (chưa mở animation)

```powershell
python scripts/check_v8_desktop_15agv.py
```

Cần in `V8 DESKTOP CHECK: PASS`, `Fleetsize: 15`. Lệnh này xác minh checkpoint/metadata/khởi tạo cấu hình, **không** xác nhận chính sách chạy chính xác trong AnyLogic.

## 3. Cấu hình AnyLogic

Giữ `runmode = 3`. Đổi biến **RL_pypeline_name** từ `"matrix_mappo_v7"` thành:

```java
"matrix_mappo_v8_15agv"
```

Giữ nguyên `RL_pypeline_path` trỏ tới folder root Python project, `py` dùng `.venv311/Scripts/python.exe`, và các chỉnh sửa `init_pypeline()` trước đây.

Cẩn thận: trong screenshot cửa sổ mô phỏng cũ có tiêu đề `MiniMatrix`. Connector sẽ kiểm tra fingerprint live và từ chối nếu đồ thị khác `105 nodes / 328 arcs`; hãy dùng đúng mô hình trước đó đã chạy V7.

Giữ `fleetsize` là biến lấy từ `get_config('fleetsize')` trong `init_pypeline()`; **không cần** sửa từ 8 sang 15 bằng tay trong Java nếu hàm vẫn nạp thông tin như trước. Nếu có cơ chế khác ghi đè `fleetsize`, kiểm tra giá trị runtime sau khi init.

Nhấn Ctrl+S, chọn Simulation: Main, rồi Run. Kiểm tra dòng `[V8 DESKTOP] Loaded best_mappo.pt with 15 AGVs; graph fingerprint verified.` và các log `[V8 DESKTOP] caller=...`.

## 4. Nếu gặp lỗi

Không sửa trọng số `.pt`. Gửi **dòng lỗi đầu tiên**, dòng `get_config('fleetsize')` trả về, và `Main` có đang dùng bản đồ đúng 105/328 hay không. Chạy 15 AGV có thể tăng block/livelock so với 8 AGV. Hai seed evaluation 3201/3202 chỉ là kiểm tra bước đầu, chưa chứng minh model tối ưu.
