# Xác minh Dispatch action 21/22 trong AnyLogic

Giải nén ZIP vào `C:\MARL_AnyLogic\MAPPO_NEW\thesis`, không ghi đè các file hiện tại.

Tại PowerShell, kích hoạt `.venv311` rồi chạy:

```powershell
python -u scripts/inspect_dispatch_map_v121.py --port 51650
```

Script lấy `HiveContext.stations` trực tiếp từ `networkcontext` của mô hình AnyLogic, đối chiếu `MatrixPart.stat_node` để báo mapping thật của từng action Dispatching. Không chạy training, không cập nhật checkpoint và không gửi hành động điều phối AGV nào. Sẽ khởi động AnyLogic Java một lần; tắt sau khi in mapping. Nếu model zip đã thay đổi, có thể chỉ định `--model`.

Gửi output bảng `LIVE DISPATCH ACTION MAPPING` và `CHECKS` để tiếp tục kiểm tra việc Dispatching chọn puffer.
