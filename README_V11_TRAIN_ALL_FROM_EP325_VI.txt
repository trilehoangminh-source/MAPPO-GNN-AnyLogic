V11 — CHUYỂN TỪ ROUTING-ONLY SANG TRAIN CẢ 3 ROLE (EP326+)
=========================================================

LÝ DO CẦN BẢN VÁ
- Run V11_CONGESTION_FROM_V8_EP304 trước đây được ghi là train_roles=routing.
- Câu lệnh --train-roles all --resume thông thường sẽ bị chặn vì khác experiment.json.
- Bản vá cho phép chuyển một chiều routing -> all khi thêm --allow-role-change.
- Không ghi đè checkpoint V8 EP304 hoặc V10. Hai file rolling_slot_*.pt của V11 VẪN được dùng luân phiên nên cần sao lưu checkpoint V11 hiện tại.

CÀI ĐẶT
1. Tắt các tiến trình train V11; giữ bản sao checkpoint và experiment.json.
   Sao lưu snapshot checkpoint mới nhất trước khi chạy (PowerShell tại thư mục thesis):
     $run = ".\mappo_runs\V11_CONGESTION_FROM_V8_EP304"
     $s = Get-Content "$run\state.json" -Raw | ConvertFrom-Json
     Copy-Item $s.checkpoint "$run\backup_before_trainall_EP$($s.last_episode).pt" -ErrorAction Stop
     Copy-Item "$run\experiment.json" "$run\experiment_before_trainall_manual.json" -ErrorAction Stop
2. Giải nén ZIP vào gốc project C:\MARL_AnyLogic\MAPPO_NEW\thesis.
3. Chỉ ghi đè scripts/train_v11_congestion.py, thêm tests/test_v11_role_change.py.

KIỂM THỬ (KHÔNG CHẠY ANYLOGIC)
python -m unittest discover -s tests -p "test_v11_role_change.py" -v
python -m unittest discover -s tests -p "test_v11_minibatch_change.py" -v

TRAIN TIẾP ĐẾN 14H NGÀY 01/11/2026 (giờ Việt Nam)
python -u .\scripts\train_v11_congestion.py --source-episode 304 --device cuda:0 --train-roles all --max-minibatch 64 --resume --allow-role-change --end-at "2026-11-01T14:00:00+07:00"

NẾU THÀNH CÔNG
- In ra [V11 ROLE CHANGE] routing -> all starting EP326 nếu state.json hiện là EP325.
- train_roles='all': train Routing, Dispatch Front và Dispatch Rear (Actor + Critic cùng role) bằng PPO.
- Dispatch vẫn dùng MLP V8, chỉ cập nhật trọng số, KHÔNG chuyển sang Transformer.
- experiment.json ghi train_roles='all' và train_roles_schedule (routing EP305-325; all EP326+).
- Tạo backup experiment_before_roles_routing_to_all_EP0326.json.
- Từng dòng metrics.jsonl sẽ lưu train_roles của episode.

CHÚ Ý
- Nếu state.json hiện đã vượt EP325, script chuyển từ last_episode + 1, không ép EP326.
- Không chạy nhiều tiến trình dùng chung output/cổng 51880.
- Không dùng --allow-role-change sau khi chuyển xong; những lần resume sau chỉ cần --train-roles all --max-minibatch 64 --resume.
- Không cần --allow-minibatch-change nếu run đã chuyển lên 64.
- Việc chuyển train_roles giữa run thay đổi thiết kế thử nghiệm; phải ghi nhận trong báo cáo.
- Bản vá đã được kiểm thử offline, chưa chạy AnyLogic end-to-end trên PC của người dùng.
