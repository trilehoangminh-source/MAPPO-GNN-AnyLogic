# V8 — Console ít log, giữ nguyên Episode / checkpoint

**Cài đặt:** Giải nén ZIP vào thư mục gốc `thesis/`, để có:
- `scripts/run_v8_quiet.py`
- `scripts/_v8_quiet_evaluate.py`
- `tests/test_v8_quiet_console.py`

**Kiểm thử không chạy AnyLogic:**

```powershell
python tests/test_v8_quiet_console.py
```

**Tác dụng:** Chỉ ẩn đúng một cảnh báo của `alpyne.data` về `observation.networkcontext = None` / expected list. Giữ nguyên tất cả cảnh báo khác, lỗi CUDA OOM và traceback. Mỗi episode sẽ được **trainer V8 gốc** in sau khi checkpoint, metrics và state.json được commit. Final evaluation cũng được lọc thông qua wrapper chạy evaluator gốc. Không sửa file trainer, GNN, optimizer hay checkpoint.

**Lệnh để resume khi một trainer cũ đã dừng:**

```powershell
$env:MAPPO_MAX_MINIBATCH = "64"
python -u scripts/run_v8_quiet.py `
  --from-previous-run mappo_runs/V8_15AGV_fromV7_5h `
  --fleetsize 20 `
  --hours 5 `
  --train-sim-seconds 1800 `
  --max-additional-episodes 500 `
  --snapshot-every 10 `
  --eval-reserve-minutes 45 `
  --eval-seeds 3301 3302 `
  --port-train 51880 `
  --port-eval 51881 `
  --output mappo_runs/V8_20AGV_from15_5h `
  --device cuda:0 `
  --cpu-threads 2 `
  --resume
```

Điều kiện: đã cài bản sửa **GNN broadcasting/VRAM guard** và **resume metrics**, đã kiểm thử offline PASS. Đừng chạy 2 trainer dùng cùng checkpoint/output/port. File mới không thể thay đổi log của tiến trình Python đang chạy; nếu hiện đang train ổn, có thể cứ tiếp tục đến khi kết thúc. Nếu muốn chuyển ngay, chỉ dừng ở giữa các episode bằng Ctrl+C; episode đang chạy sẽ không được lưu.

Định dạng một dòng/episode từ chính trainer cũ (các giá trị chỉ ví dụ):

`[MAPPO-V8 EP 145] jobDone=... wrong=... block=... livelock=... reward=... KL=... elapsed=...min`

**Cảnh báo:** Ẩn warning không sửa nguyên nhân `networkcontext=None`. GPU OOM hoặc AnyLogic treo cần xử lý riêng. Nếu một episode rất lâu không kết thúc, sẽ chưa có dòng EP. Không tắt warning/error chung.
