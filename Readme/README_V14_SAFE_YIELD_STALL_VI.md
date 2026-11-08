# V14 — Safe Yield + Persistent Stall (thử nghiệm reward mới từ EP130)

**Dự án:** Minh Trí / MAPPO–GNN–AnyLogic, 20 AGV, GAT4, 2 Deep Candidate-Aware MLP Dispatch Actors, 3 Critics.  
**Ngày biên soạn:** 04/11/2026.  
**Quan trọng:** Bản vá này đã qua kiểm thử ngoại tuyến; **CHƯA chạy end-to-end trên AnyLogic máy Windows của bạn**. Không tuyên bố đây là kết quả tốt hơn V14 Reward trước.

## 1. Giả thuyết và tài liệu

Bản thí nghiệm V14 Stop/Swing trước dùng hình phạt thêm `−0.5` cho Stop được heuristic coi là không cần thiết. Kết quả **held-out 10 seed** của checkpoint EP130: JobDone trung bình **11,1**, Block **44,9**, LiveLock **161,2**, Block time **13.838 giây AGV cộng dồn**, Stop khoảng **15,1%**. EP110/120 có ít Block hơn nhưng thông lượng kém hơn. Mục tiêu V14 nhánh mới: giữ thông lượng, giảm Block/LiveLock/time.

- Steinbacher, Wegmann, Freitag (2025), *Production control with Reinforcement Learning for a matrix-structured production system*, phần 3.2.2 và 4.1.1, DOI: https://doi.org/10.1080/00207543.2024.2436126. Tác giả giải thích phạt dừng cao làm xe không chịu đứng lại, giới thiệu `reward_pass` và phạt thời gian dừng kéo dài thay cho phạt Stop đều.
- Song et al. (2024), *Cooperative Reward Shaping for Multi-Agent Pathfinding*: https://arxiv.org/abs/2407.10403. Đưa ra ý tưởng reward tính đến tương tác láng giềng; **không** đưa ra công thức chính xác dùng trong patch này.
- Phan et al. (2026), *Confidence-Based Curricula for Multi-Agent Path Finding via Reinforcement Learning*, https://link.springer.com/article/10.1007/s10458-026-09747-7. Thảo luận khác biệt chờ cần thiết/chờ vô ích là vấn đề khó. Đây là **proxy heuristic**, không chứng minh xe có nhường đường thực sự.

## 2. Những gì thay đổi

**Chỉ thay Python Routing reward ngay lúc môi trường lưu quan sát:**

```text
AnyLogic -> MultiAgent.convert_from_observation (reward Java gốc)
         -> Matrix._save_observation (lưu observation + reward)
         -> [NEW] điều chỉnh env.rewards[agent] bằng SafeYieldStallReward
         -> PettingZoo env.last() -> collector -> PPO
```

**Không sửa `thesis/envs/behaviors_v121.py`, `thesis/envs/matrix_v121.py`, Java/AnyLogic model, PPO collector, action mask, architecture, Actor/Critic, optimizer.** Lớp reward mới gắn wrapper chỉ trong tiến trình Python trainer/evaluator mới. Không can thiệp vào buffer sau khi transition hoàn tất. Java vẫn có phần thưởng gốc như `reward_pass=+0.2`, `reward_duration=−0.5`, `reward_invalid=−0.1` theo cấu hình nguồn (có thể kiểm tra trực tiếp ở `scripts/train_mappo_long_B3_v121.py`).

**Các hệ số mặc định của reward bổ sung** (giá trị nghiên cứu ban đầu, chưa tối ưu):

| Thành phần | Điều kiện | Giá trị |
|---|---|---:|
| Python Stop penalty mới | Bất kỳ Stop nào | **0** |
| Safe-yield bonus | Xe dừng và một xe **đã ở gần trước đó** quan sát được đi ra xa hơn. Chỉ 1 lần/đợt đứng tại cùng vị trí | **+0,35** |
| Grace period | Không di chuyển nhưng chưa quá 45 giây mô phỏng | **0** |
| Stall penalty | Không đổi vị trí khi chưa tới mục tiêu; chỉ phạt số giây mô phỏng tăng thêm **sau 45 giây** | **−0,10 mỗi 10 giây** |
| Giới hạn stall | Mỗi đợt đứng liên tục, tránh reward bị chi phối cực đoan | **tối đa −2,0** |
| Swing penalty | Phát hiện A → B → A trên quan sát node | **−1,0** |
| Đang ở mục tiêu hoặc Dispatch đổi mục tiêu | Miễn/reset đồng hồ tắc nghẽn | **không phạt** |

**Lưu ý chính xác hóa:** "Safe yield" là proxy dựa trên vị trí vật lý quan sát được, không phải xác nhận đã xảy ra nhường đường/collision-safe trong Java. "Stall" là proxy **thời gian đứng yên ngoài mục tiêu**, không phải truy xuất trực tiếp biến Block/LiveLock Java; có thể phạt một số tình huống chờ bắt buộc kéo dài. Chúng ta sẽ kiểm chứng tỷ lệ và dấu hiệu này qua log.

## 3. Những file ZIP bổ sung (KHÔNG ghi đè code cũ)

```text
scripts/train_v14_safe_yield_stall.py
scripts/evaluate_v14_safe_yield_stall.py
thesis/envs/v14_safe_yield_stall.py
tests/test_v14_safe_yield_stall_offline.py
tests/test_v14_safe_yield_preflight_offline.py
README_V14_SAFE_YIELD_STALL_VI.md
```

Các dependency V14 Stop-Unlock + Reward trước **phải còn tồn tại ở dự án gốc**, gồm `scripts/train_v14_stop_unlocked.py`, `scripts/v14_deep_mlp_common.py`, `scripts/v14_reward_collector.py`, `thesis/envs/v14_reward_shaping.py` và những phần kế thừa V12. ZIP **không** chứa checkpoint `.pt`, dữ liệu training, hay Java model.

## 4. Trình tự thao tác an toàn (PowerShell, VS Code)

### 4.1. Dừng trainer cũ và kiểm tra source snapshot (chỉ đọc)

Nếu còn trainer V14 chạy ở port 51880, nhấn **Ctrl+C một lần**, đợi log `committed EP...` và terminal trả prompt. Sau đó:

```powershell
cd C:\MARL_AnyLogic\MAPPO_NEW\thesis
$old = '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_WAIT_SWING_REWARD'
Test-Path "$old\snapshot_ep_0130.pt"
Get-FileHash "$old\snapshot_ep_0130.pt" -Algorithm SHA256
```

Phải thấy `True`, SHA-256 EP130 chính xác là:

`d9060d5478cee62725d7bdd91c8391a1584870cce69a7b3665afe492cd0abb0c`

Mã băm lấy từ **10-seed evaluation thật của EP130** người dùng đã gửi. Nếu khác, **DỪNG**, không dùng snapshot khác mà nói là EP130.

### 4.2. Giải nén ZIP vào gốc dự án

Giải nén vào `C:\MARL_AnyLogic\MAPPO_NEW\thesis` (cùng thư mục `scripts`, `thesis`, `tests`). Không thay thế file cũ. Đừng copy cả ký hiệu ``` từ ChatGPT vào PowerShell.

### 4.3. Chạy unit tests (không cần AnyLogic)

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p 'test_v14_safe_yield*_offline.py' -v
```

Dự kiến: **16 tests OK**. Đây chỉ là test code ngoại tuyến và kiểm tra preflight, không thay thế kiểm thử thực với AnyLogic.

### 4.4. Dry-run (không chạy AnyLogic, không ghi run)

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_safe_yield_stall.py --dry-run
```

Phải in `[SAFE YIELD DRY RUN OK]`. Nó bắt buộc: đúng nhánh nguồn reward cũ, snapshot EP130 đúng SHA, 20 AGV/3600s, code reward cũ khớp hash, model ZIP và blueprint không đổi, các cấu hình mạng/GPU/batch khớp.

### 4.5. Fork checkpoint nhưng CHƯA train

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_safe_yield_stall.py --init-only
```

Nó sẽ tạo thư mục **MỚI**:

`mappo_runs\V14_GAT4_MLP_3600S_MB128_SAFE_YIELD_STALL_FROM_EP130`

và file `rolling_slot_0.pt`, `experiment.json`, `state.json`, `metrics.jsonl`. Checkpoint EP130 copy nguyên byte và giữ cả actor/critic/Adam/ValueNorm; EP131 mới là episode đầu tiên có cập nhật. Không được dùng `--init-only` lần thứ hai trong cùng output.

### 4.6. Kiểm tra action không thay đổi ngay sau fork (rất nên làm)

Đánh giá checkpoint EP130 **chưa train với reward mới**, trước tiên trên 2 seed từng dùng cho baseline:

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\evaluate_v14_safe_yield_stall.py --device cuda:0 --port 51881 --seeds 900001 900002
```

Vì checkpoint và action mask vẫn nguyên, và Python reward không điều khiển mô hình AnyLogic, **KPI (không phải tổng Reward) dự kiến trùng với held-out EP130 cũ khi dùng greedy và môi trường deterministic cùng seed**:

| Seed | JobDone | Block | LiveLock | Stop ratio |
|---|---:|---:|---:|---:|
| 900001 | 8 | 28 | 99 | 42,09% |
| 900002 | 17 | 62 | 198 | 5,02% |

Nếu KPI khác lớn, dừng để tìm nguyên nhân (model version, model time, action policy hoặc nguồn ngẫu nhiên) trước khi so kết quả. Để đánh giá đủ 10 seed, chạy lại lệnh không có `--seeds ...` (sẽ tạo file tên khác vì số seed khác). **Không chạy trainer và evaluator đồng thời nếu AnyLogic/GPU trên máy thường bị treo.**

### 4.7. Train đúng 1 episode từ EP130 → EP131

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_safe_yield_stall.py --max-episodes 131 --resume
```

Trong log kỳ vọng: `[SAFE YIELD EP131] JobDone=... Block=... LiveLock=... YieldBonus=... StallPenalty=... Reward=...` và `committed EP 131`.

Kiểm tra metrics của run mới:

```powershell
$new = '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_SAFE_YIELD_STALL_FROM_EP130'
$m = Get-Content "$new\metrics.jsonl" -Tail 1 | ConvertFrom-Json
$m.environment.reward_shaping | Format-List
$m.ppo_audit | Format-List
$m.all_roles_audit | Format-List
```

Các trường `yield_proxy_bonuses_applied`, `stall_penalties_applied`, `total_stall_penalty`, `stationary_at_target_exempt`, `target_changes_reset_dwell` phải có ý nghĩa. Một trường có thể bằng 0 ở một episode; không phải lỗi. Kiểm tra PPO audit pass, `selected_mask_violations=0`, đủ Routing + Front/Rear update nếu role có trajectory.

### 4.8. Train tới EP160 trên nhánh mới

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_safe_yield_stall.py --max-episodes 160 --resume
```

Sẽ có snapshot EP140/150/160. **Không train thẳng tới 1100 ngay:** quan sát trung bình theo nhóm EP131–140, 141–150, 151–160, đặc biệt JobDone, Block, LiveLock, tổng `_blocksec_agv*`, số lần được thưởng Yield và thời gian đứng yên bị phạt.

### 4.9. Đánh giá 10 seed ở checkpoint mới

Dừng trainer an toàn. Ví dụ đánh giá snapshot EP160:

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\evaluate_v14_safe_yield_stall.py --device cuda:0 --port 51881 --checkpoint '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_SAFE_YIELD_STALL_FROM_EP130\snapshot_ep_0160.pt'
```

Nếu bị ngắt, **thêm** `--resume` với các flags như cũ. Evaluator ghi file held-out riêng trong run mới, summary CSV, không gọi `model.train()` và không sửa checkpoint. Có thể làm tương tự cho EP140 và EP150, cùng 10 seed mặc định từ 900001 tới 900010.

**Tiêu chí so baseline EP130:** ưu tiên `JobDone_mean >= 11.1` đồng thời `Block_mean < 44.9`, `LiveLock_mean < 161.2` và `block_time_s_mean < 13838`, đánh giá cùng điều kiện 20 AGV, 3600s, greedy, cùng 10 seed. Đây là ngưỡng nghiên cứu đề xuất, **không phải bảo đảm**.

## 5. Giới hạn và bảo vệ

- Bản vá **chỉ thay reward Python Routing**, không thêm mask cấm Stop. Stop gốc AnyLogic vẫn được phép như branch Stop-Unlock.
- Không train full scratch; đây là **fine-tune từ EP130**. Không công bố như một thí nghiệm 160 episode scratch.
- Không có bảo đảm rằng reward mới sẽ giảm LiveLock hay Block. Có thể cải thiện thông lượng nhưng khiến xe đứng đợi quá lâu hoặc cố tránh đường sai; chỉ kết luận sau 10-seed eval.
- Không thay code của run V14 Reward cũ; thay đổi tham số reward sau fork sẽ bị chặn lúc resume bằng manifest/hash.
- File source `snapshot_ep_0130.pt` được đọc nhưng **không bị sửa**. Các output chỉ ghi trong thư mục run mới.
- Nếu dry-run thất bại với `checkpoint hash`, `Source ... version`, hoặc `AnyLogic model ZIP changed`, không lách bằng cách xóa kiểm tra. Gửi log để khắc phục trên dữ liệu thực.

---

**Kết quả kiểm thử trong môi trường biên soạn:** `py_compile` thành công; 16 unit tests offline pass. **Chưa thực thi AnyLogic Windows, chưa có dữ liệu thực nghiệm cho reward mới.**
