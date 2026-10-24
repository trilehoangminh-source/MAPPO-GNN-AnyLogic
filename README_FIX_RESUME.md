# Sửa lỗi V8 resume khi chuyển 15 → 20 AGV

Lỗi cũ: `RuntimeError: Missing committed metrics for latest checkpoint` khi `state.json` vẫn ở Episode 144, trỏ vào rolling checkpoint của V8 15 AGV.

Nguyên nhân: `_mappo_v8_utils.py` dùng ngưỡng 30 thay vì đọc `first_completed_episode` trong `overnight_config.json`. Với run 20 AGV, Episode 144 là điểm chuyển giao; trước khi Episode 145 hoàn tất thì `metrics.jsonl` có thể chưa tồn tại.

## Cài đặt

1. Đóng phiên training Python/AnyLogic đang chạy.
2. Từ root repo, sao lưu file cũ:

   ```powershell
   Copy-Item scripts/_mappo_v8_utils.py scripts/_mappo_v8_utils.py.bak
   ```

3. Giải nén ZIP ở thư mục gốc của repo. Chỉ ghi đè `scripts/_mappo_v8_utils.py`, thêm file test.
4. Chạy `python tests/test_v8_transfer_resume_metrics.py` và kỳ vọng PASS.

## Resume lại đúng run cũ

```powershell
python -u scripts/train_B3_MAPPO_v8_overnight.py `
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
  --resume
```

**Không xóa** checkpoint, state, metadata hoặc các thư mục cũ. Từ trạng thái Episode 144, episode đầu tiên hoàn tất sẽ là 145. Đây chỉ là bản vá điều kiện resume, không chữa được treo máy do CPU/RAM/Java. `--hours 5` tính cho phiên mới.
