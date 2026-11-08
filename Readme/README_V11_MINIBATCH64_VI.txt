V11: CHUYỂN MAX-MINIBATCH 32 -> 64 VÀ GIỮ NGUYÊN TIẾN ĐỘ
=======================================================

1) CHỈ ghi đè scripts/train_v11_congestion.py (không thay V8 hay các model).
2) Đảm bảo tiến trình train cũ đã dừng hoàn toàn và đã đóng AnyLogic.
3) Trong thư mục project chạy:
   python -m unittest discover -s tests -p "test_v11_minibatch_change.py" -v
4) Xem last_episode của run:
   (Get-Content .\mappo_runs\V11_CONGESTION_FROM_V8_EP304\state.json -Raw | ConvertFrom-Json).last_episode
5) Resume có chuyển giới hạn mini-batch:
   python -u .\scripts\train_v11_congestion.py --source-episode 304 --device cuda:0 --train-roles routing --max-minibatch 64 --resume --allow-minibatch-change --end-at "2026-11-01T08:30:00+07:00"

Chức năng sửa:
- Các tham số khác ngoài max-minibatch phải giữ NGUYÊN so với experiment.json.
- Đọc và kiểm tra state.json, checkpoint, danh sách metrics trước khi chuyển.
- Tự lưu backup experiment_before_minibatch_32_to_64_EPxxxx.json.
- Ghi minibatch_schedule trong experiment.json: 32 tới EP nào; 64 bắt đầu từ EP nào.
- metrics.jsonl của các episode mới thêm max_minibatch=64.
- Tiếp tục bằng checkpoint V11 cuối cùng đã được xác nhận, KHÔNG load lại V8 nếu đã hoàn thành EP305 trở đi.

LƯU Ý: Đổi giới hạn mini-batch làm thay đổi số bước PPO optimizer và quỹ đạo học.
Đây là chủ động đổi hyperparameter giữa run, không phải run đồng nhất hoàn toàn.
Nếu có CUDA out-of-memory thì dừng train và xin xử lý, không sửa lén manifest.
Mốc --end-at phải là thời điểm tương lai khi chạy lệnh.
