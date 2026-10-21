# Khôi phục V6 C/D khi AnyLogic/Alpyne timeout giữa evaluation

## Điều đã biết từ log

- Evaluator `evaluate_B3_edgeGAT_MAPPO_v4.py` mất kết nối tới `127.0.0.1:51831` trong `Alpyne.lock()`.
- V6 runner báo `C_eval_seed143.log`; file `evaluation_3600.jsonl` đã có 4/5 dòng.
- **Không xóa** `train_seed143`, checkpoint `.pt`, file kết quả 4 dòng hoặc log lỗi.
- `lock?timeout=180000` chỉ là tham số chờ trạng thái, **không** có nghĩa Requests đang cấu hình TCP connect timeout 180 giây (`connect timeout=None` trong traceback).

## Cài đặt

Giải nén patch vào thư mục gốc:

`C:\MARL_AnyLogic\MAPPO_NEW\thesis`

Các file chỉ được **thêm**:

- `scripts/recover_B3_V6_partial_eval.py`
- `tests/test_B3_V6_recovery_offline.py`
- `README_B3_V6_recovery.md`

## Bước 1: Kiểm tra cổng và các tiến trình Java (chỉ đọc)

```powershell
Get-NetTCPConnection -LocalPort 51831 -ErrorAction SilentlyContinue | Select-Object State, OwningProcess
Get-CimInstance Win32_Process -Filter "name = 'java.exe'" | Select-Object ProcessId, CommandLine
```

Không tự động `taskkill /IM java.exe /F`: có thể vô tình tắt các ứng dụng khác. Nếu AnyLogic còn bị treo, xác định đúng PID của mô phỏng V6 rồi xử lý thủ công. Bản recovery dùng cổng mới 51832.

## Bản sửa Windows UTF-8 (fix1)

Test offline dùng `Path.read_text(encoding="utf-8")` và các file JSON giả lập được tạo bằng UTF-8. Bản cũ thiếu `encoding` ở dòng đọc output, gây `UnicodeDecodeError: cp1252` trên Windows 3.11 khi đường dẫn có dấu tiếng Việt. Bản fix không thay đổi trọng số MAPPO, checkpoint, hoặc logic retry AnyLogic. Có test chuỗi tiếng Việt để kiểm tra đọc/ghi UTF-8.

Nếu đã từng giải nén bản recovery cũ, hãy cho phép **ghi đè đúng các file scripts/tests của recovery patch**, không tác động thư mục `mappo_runs`.

## Bước 2: Chạy test offline

```powershell
python tests/test_B3_V6_recovery_offline.py
```

## Bước 3: Kiểm tra 4/5 mà chưa chạy mô phỏng

```powershell
python scripts/recover_B3_V6_partial_eval.py --folder mappo_runs/B3_V6_CD_30ep_720s --condition C --train-seed 143 --dry-run
```

Kỳ vọng in `completed=4/5; missing=[2005]` nếu bốn dòng hiện tại là seed 2001–2004. Tool kiểm tra config train, checkpoint, 30 dòng metrics, thứ tự evaluation seeds và integrity của từng dòng; nếu không đúng sẽ dừng mà không thay đổi gì.

## Bước 4: Chạy **riêng seed còn thiếu**, giữ nguyên checkpoint

```powershell
python -u scripts/recover_B3_V6_partial_eval.py --folder mappo_runs/B3_V6_CD_30ep_720s --condition C --train-seed 143 --port 51832
```

Tool làm các việc sau:

1. Mở đúng checkpoint C/seed143 sau 30 episodes, chạy AnyLogic chỉ với seed chưa được đánh giá.
2. Ghi log retry trong `mappo_runs/B3_V6_CD_30ep_720s/recovery_logs/`.
3. Nếu evaluation lỗi nữa: **để nguyên file 4/5**, không ghép dữ liệu.
4. Nếu thành công: kiểm tra checkpoint/guide/version/terminal/seed; sao lưu file 4 dòng trước khi ghép; ghép vào file `evaluation_3600.jsonl` qua bước ghi file tạm rồi đổi tên.
5. Xác nhận 5/5 bằng đúng kiểm tra `valid_eval` của runner V6.

Nếu cổng 51832 đang bận thì dùng `--port 51833`. Không thay đổi Dijkstra strength, không train lại MAPPO.

## Bước 5: Tiếp tục toàn bộ thí nghiệm V6

Giữ **đúng tham số ban đầu** và thêm `--resume`. Có thể đổi cổng evaluation để tránh cổng 51831 cũ:

```powershell
python -u scripts/run_B3_V6_CD_replication.py `
  --train-seeds 43 143 243 `
  --eval-seeds 2001 2002 2003 2004 2005 `
  --train-episodes 30 `
  --train-sim-seconds 720 `
  --eval-sim-seconds 3600 `
  --output mappo_runs/B3_V6_CD_30ep_720s `
  --port-eval 51832 `
  --resume
```

Runner sẽ bỏ qua lượt đã hoàn thành và được kiểm tra hợp lệ. Nếu phát sinh một lượt lỗi mới, sử dụng lại recovery tool với `--condition` và `--train-seed` phù hợp. Tool chỉ hỗ trợ **evaluation có các dòng hợp lệ theo đúng thứ tự prefix**, không khôi phục training dở dang.

## Bước 6: Khi runner đã xong toàn bộ

```powershell
python scripts/analyze_B3_V6_CD_replication.py --folder mappo_runs/B3_V6_CD_30ep_720s
```

Mô hình kiến trúc và weights không bị đụng chạm: **Edge-GAT cho Routing, B2 pooled/MLP cho Dispatching, MAPPO/CTDE giữ nguyên**.
