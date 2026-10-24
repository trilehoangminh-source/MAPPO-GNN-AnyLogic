# V8 CUDA OOM patch — 20 AGV

## Vấn đề
`torch.OutOfMemoryError` tại `GraphRoutingActor.logits()` trong PPO update. `pos_valid`, `tar_valid` có dạng `[B,1]`, còn `h[...]` dạng `[B,1,H]`. `torch.where` tự broadcast ra `[B,B,H]` => sử dụng VRAM tăng theo bình phương batch. Đây là lỗi logic kích thước tensor, không chỉ do 4 GB VRAM ít.

## Khắc phục
1. Đóng terminal cũ đã OOM. Dùng terminal mới, tại thư mục gốc `thesis`.
2. Giải nén ZIP vào project (chỉ có `scripts/fix_v8_gpu_oom.py`, `tests/test_v8_gpu_oom_offline.py`, README; không ghi đè trainer hiện tại).
3. Chạy `python scripts/fix_v8_gpu_oom.py --check` và `python scripts/fix_v8_gpu_oom.py --apply`. Script ghi `.bak` trước khi thay code, chỉ chấp nhận pattern đã biết; không đụng checkpoint.
4. Chạy `python tests/test_v8_gpu_oom_offline.py`.
5. Trong cùng terminal PowerShell định train, đặt `$env:MAPPO_MAX_MINIBATCH = "64"` để PPO chia nhiều minibatch nhỏ. Có thể thử `32` nếu lại OOM; điều này **thay đổi số optimizer updates** nên cần đánh giá lại learning curve.
6. Giữ nguyên các tham số và thư mục `--output`, dùng `--resume` để tiếp tục. Khi mới chuyển 15→20 mà chưa có episode 20 AGV hoàn tất, phải giữ bản `_mappo_v8_utils.py` đã được sửa từ bản vá resume-metrics trước đó.

## Cảnh báo
- Sửa broadcasting có thể làm hành vi các quan sát không match node được tính đúng hơn; không bảo đảm checkpoint cũ hành xử y hệt trong trường hợp có fallback.
- Hãy kiểm tra một Episode đầy đủ, GPU peak VRAM, `state.json` tăng episode rồi mới chạy qua đêm.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` không giải quyết lỗi tạo tensor `[B,B,H]` và `empty_cache()` cũng không khắc phục lỗi logic.
- Không sửa `.pt`, không xóa metrics và không chạy đồng thời nhiều Python/AnyLogic.
