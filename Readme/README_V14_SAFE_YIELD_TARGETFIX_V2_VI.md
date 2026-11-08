# V14 Safe Yield V2 — Sửa Target 6:8, train lại an toàn từ EP130

## Vì sao không dùng `--resume` của V1?

Ông đã sửa `thesis/envs/v14_safe_yield_stall.py` (từ `_point(ego,4)` sang `_point(ego,6)`). Runner V1 cố ý đối chiếu SHA-256 file reward với `experiment.json`; resume V1 sau khi sửa phải báo lỗi. Không bỏ khóa này và không tự sửa `experiment.json` của V1.

V2 là một **run mới** bắt đầu lại từ checkpoint EP130 đã dùng trong held-out 10 seed, không phải full scratch. Điều này bảo đảm V1 và V2 có chung mốc trọng số trước khi reward mới được áp dụng, tránh trộn giai đoạn học reward sai vào V2. Snapshot EP130 có SHA-256 cố định `d9060d5478cee62725d7bdd91c8391a1584870cce69a7b3665afe492cd0abb0c`.

## File trong ZIP

- `scripts/train_v14_safe_yield_stall_targetfix_v2.py` — runner V2.
- `scripts/evaluate_v14_safe_yield_stall_targetfix_v2.py` — held-out greedy V2, 10 seed.
- `tests/test_v14_safe_yield_targetfix_v2_offline.py` — 4 hồi quy quan trọng.

**Không có file reward** trong ZIP; script dùng lại file `thesis/envs/v14_safe_yield_stall.py` ông đã sửa trên máy. Vì vậy không ghi đè thay đổi của ông.

## Cài đặt

Giải nén ZIP tại gốc `C:\MARL_AnyLogic\MAPPO_NEW\thesis` (chỗ có `.venv311`, `scripts`, `tests`, `thesis`, `mappo_runs`). Chỉ thao tác khi không có phiên train khác sử dụng port 51880. Không xóa run V1.

## 1. Kiểm tra chỉnh sửa và checkpoint

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p 'test_v14_safe_yield_targetfix_v2_offline.py' -v

.\.venv311\Scripts\python.exe -u .\scripts\train_v14_safe_yield_stall_targetfix_v2.py --dry-run
```

Nếu báo `Target bug not fixed` hoặc source snapshot checksum sai, **dừng lại**. Không sửa/sao chép checkpoint thủ công, không sửa `experiment.json` để bỏ kiểm tra.

## 2. Tạo V2 nhưng chưa training

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_safe_yield_stall_targetfix_v2.py --init-only
```

Kiểm tra `mappo_runs\V14_GAT4_MLP_3600S_MB128_SAFE_YIELD_STALL_TARGETFIX_V2_FROM_EP130\state.json` có `last_episode=130` và checkpoint `.pt` thật sự tồn tại.

## 3. Chạy thử một episode (EP131)

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_safe_yield_stall_targetfix_v2.py --max-episodes 131 --resume
```

Tìm `[SAFE YIELD V2 EP131]`, xem `metrics.jsonl` trường `reward_shaping.target_changes_reset_dwell`, `stall_penalties_applied`, `stationary_at_target_exempt`; kiểm tra PPO audit, mask violation, JobDone/Block/LiveLock. Không mặc định `StallPenalty` phải >0 mỗi EP.

## 4. Nếu EP131 đạt, chạy đến EP150

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_safe_yield_stall_targetfix_v2.py --max-episodes 150 --resume
```

`--max-episodes` là mốc tuyệt đối. Ctrl+C một lần để ngắt an toàn.

## 5. Held-out chung 10 seed

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\evaluate_v14_safe_yield_stall_targetfix_v2.py --device cuda:0 --port 51881 --dry-run
.\.venv311\Scripts\python.exe -u .\scripts\evaluate_v14_safe_yield_stall_targetfix_v2.py --device cuda:0 --port 51881
```

Seed 900001–900010; 3.600s, 20 AGV; greedy, không PPO update. Có thể thêm `--resume` nếu đánh giá bị gián đoạn. Không chạy cùng port với trainer. Nếu chọn checkpoint EP150 hãy thêm `--checkpoint '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_SAFE_YIELD_STALL_TARGETFIX_V2_FROM_EP130\snapshot_ep_0150.pt'`.

## Bảo vệ

- Nguồn EP130 được đối chiếu hash; V1 và reward Stop/Swing gốc giữ nguyên.
- Trainer khóa cấu hình reward, Python implementation SHA, AnyLogic model SHA, kiến trúc/port/seed để resume không trộn thí nghiệm.
- `assert_target_fix()` chặn code vẫn dùng `_point(ego, 4)` hoặc chưa sửa `agvs.shape[1] < 8`.
- Không đưa `.pt`, `state.json`, `metrics.jsonl` lên GitHub.
- Sửa Target index là thay đổi quan trọng nhưng **không bảo đảm** Block/LiveLock giảm; cần held-out mới kết luận.

## Ghi chú nghiên cứu

Theo source Tobias, `agv[2:4]` = last node, `agv[4:6]` = next node, `agv[6:8]` = target node. Reward V2 vẫn dùng last-node làm proxy vị trí để theo dõi dwell; chưa tương đương vị trí liên tục của AGV, và vẫn không bắt được mọi livelock do đi vòng. Nếu cần mở rộng proxy, làm trong V3 riêng sau khi V2 đánh giá.
