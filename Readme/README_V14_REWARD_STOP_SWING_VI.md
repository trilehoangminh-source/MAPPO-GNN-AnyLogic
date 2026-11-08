# V14 — REWARD PHẠT STOP KHÔNG CẦN THIẾT + PHẠT A→B→A

**Dự án:** MAPPO–GNN–AnyLogic, 20 AGV, GAT4, Deep Candidate-Aware MLP, 3 Critics, 3600 s/episode, Stop UNLOCKED, không Dijkstra.

**Mục đích:** Tạo **thí nghiệm reward mới**, kế thừa toàn bộ trọng số Actor/Critic, optimizer Adam và ValueNorm từ **checkpoint V14 STOP-UNLOCK đã commit**; giữ nguyên nhánh backup GitHub và thư mục run V14 cũ.

> **Tình trạng kiểm thử:** Đã kiểm tra cú pháp, 9 unit tests reward, và preflight dry-run/resume bằng dữ liệu giả lập; **chưa chạy AnyLogic/Alpyne hoặc GPU thực tế** trên máy Windows của bạn. Chỉ xem là thử nghiệm, chưa phải hiệu năng đã được chứng minh.

## 1. Thay đổi thực tế của code

Thư mục giải nén chỉ thêm **các file mới**:

```
scripts/train_v14_reward_stop_swing.py
scripts/v14_reward_collector.py
thesis/envs/v14_reward_shaping.py
tests/test_v14_reward_stop_swing_offline.py
README_V14_REWARD_STOP_SWING_VI.md
```

**Không ghi đè** `behaviors_v121.py`, `matrix_v121.py`, `collector_stop_ab.py`, `train_v14_stop_unlocked.py` hoặc bất cứ file nguồn V14 hiện có nào. Runner mới gắn wrapper **tạm thời chỉ trong tiến trình train mới** quanh `MultiAgent.convert_from_observation()`/`convert_to_action()`; chính `reward` trả về môi trường đã được điều chỉnh *trước khi Collector thu thập trajectory*. KHÔNG gắn phạt bổ sung hậu kỳ vào transition PPO đã đóng. Mã cũ hoạt động như trước nếu vẫn gọi trainer cũ.

### Công thức

```
new_routing_reward = original_AnyLogic_reward
                   + original_existing_direction_reward
                   + wait_adjustment
                   + swing_adjustment
```

- `wait_adjustment = -0.5` **chỉ** khi action trước là `Stop=0` và tại thời điểm ra quyết định trước đó có ít nhất một hướng di chuyển hợp lệ *và tương đối thông thoáng*. Kiểm tra theo mask gốc, tọa độ node hướng đi (observation các cột 8–15), vị trí AGV khác (cột 2–3), vị trí waypoint quan sát được (cột 4–5). Nếu ở gần AGV khác, không có hướng hợp lệ, ứng viên bị chiếm/nhắm tới, thiếu dữ liệu, hoặc AGV đã ra khỏi hệ thì **không phạt thêm**.
- `swing_adjustment = -1.0` **khi phát hiện 3 vị trí Routing phân biệt liên tiếp** A→B→A của cùng AGV vật lý, tức quay lại vị trí nó vừa rời đi. Nếu AGV đứng yên ở B thì không tính swing. Đây là heuristic: trong tình huống buộc phải tránh chướng ngại, quay lại vẫn có thể hợp lý.
- Hai phạt này chỉ áp dụng cho **Routing**, không thêm phạt cho Dispatch Front/Rear. Mask `Stop` vẫn là `baseline` (Stop hợp lệ nếu AnyLogic cho phép); không chỉnh logits, PPO loss, Critic, reward Java hay số AGV.
- **Không bổ sung move penalty / goal bonus mới** để tránh chồng lắp quá mức lên reward AnyLogic đang có: `reward_duration=-0.5`, `reward_pass=+0.2`, `reward_reached_target=+10`, `reward_part_completed=+2` theo V14.
- Các ngưỡng `candidate_clearance = 0.025`, `local_risk_radius = 0.040`, `position_tolerance = 0.002` là **giả định ban đầu trên tọa độ chuẩn hóa**; không được xem như điều kiện an toàn thật của AnyLogic.

**Về nguồn:** Ma & Lian (2023), *Learning attention-based strategies to cooperate for multi-agent path finding*, đề xuất swing penalty khi AGV trở về vị trí trước đó (Table 1: wait −0.5, move −0.3, swing −1, goal +40). Phan và cộng sự, *Confidence-Based Curricula for Multi-Agent Path Finding via Reinforcement Learning* (2026), nêu sự đánh đổi của phạt chờ và vấn đề phân biệt necessary/unnecessary waits. **Bài Phan không cung cấp sẵn heuristic phạt riêng Stop không cần thiết đang dùng ở đây**; đó là phần **thiết kế thử nghiệm của dự án**. Chưa triển khai curriculum CACTUS (thay bán kính phân bố mục tiêu); chỉ lấy ý tưởng nhận diện thời gian chờ.

Nguồn:
- https://doi.org/10.52396/JUSTC-2022-0048
- https://doi.org/10.1007/s10458-026-09747-7

## 2. Dừng V14 trước khi giải nén

Trong PowerShell tại `C:\MARL_AnyLogic\MAPPO_NEW\thesis`:

```powershell
cd C:\MARL_AnyLogic\MAPPO_NEW\thesis
```

Nếu trainer còn chạy, nhấn Ctrl+C **một lần** rồi đợi terminal trả về prompt, có thông báo `Last committed EP ...`. Không giết Java bằng lệnh `taskkill /IM java.exe /F`.

Kiểm tra **checkpoint của STOP-UNLOCK**, không nhầm với V14 scratch:

```powershell
$source = '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK'
$s = Get-Content "$source\state.json" -Raw | ConvertFrom-Json
$m = Get-Content "$source\experiment.json" -Raw | ConvertFrom-Json
Write-Host "Committed EP: $($s.last_episode)"
Write-Host "Checkpoint exists: $(Test-Path -LiteralPath $s.checkpoint)"
Write-Host "Version: $($m.version) | Stop: $($m.stop_mask_mode)"
```

Cần `Checkpoint exists: True`, `Stop: baseline`, và `Version: V14_GAT4_CANDIDATE_MLP_STOP_UNLOCK_CONTINUE`. **Không** xóa `mappo_runs`, `.pt`, `state.json`, `metrics.jsonl`. Lưu ý GitHub chỉ backup mã nguồn, không backup checkpoint `.pt` trừ khi bạn lưu riêng. Runner mới sẽ sao chép nguyên byte checkpoint được commit sang run mới khi init.

## 3. Giải nén ZIP

Giải nén `V14_REWARD_STOP_SWING_PATCH.zip` **vào thư mục gốc `thesis`**, cùng cấp với `scripts`, `tests`, thư mục Python package `thesis` (tức `thesis\thesis\envs`). Không giải nén thêm một cấp `v14_reward_patch` vào bên trong. **ZIP không sửa file V14 hiện có**.

## 4. Kiểm thử OFFLINE (không chạy mô phỏng)

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p 'test_v14_reward_stop_swing_offline.py' -v
```

Kỳ vọng `Ran 9 tests ... OK`. Sau đó dry-run preflight chỉ đọc các file nguồn:

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_reward_stop_swing.py --fork-from '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK' --device cuda:0 --sim-seconds 3600 --max-minibatch 128 --dry-run
```

Phải thấy `[V14 REWARD DRY RUN]`, báo `source committed EP = ...`, mô tả `Stop remains UNLOCKED`, reward −0.5/−1.0. Nếu báo sai phiên bản, sai tham số/horizon/port/checkpoint hash, dừng để sửa tham số truyền vào cho khớp **đúng run cũ**, không sửa `experiment.json` bằng tay.

## 5. FORK từ checkpoint cuối đã commit — chưa train

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_reward_stop_swing.py --fork-from '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK' --device cuda:0 --sim-seconds 3600 --max-minibatch 128 --init-only
```

Lệnh trên **khởi động AnyLogic để kiểm tra graph/khởi tạo model/load checkpoint**, sau đó tạo thư mục mới:

```
mappo_runs\V14_GAT4_MLP_3600S_MB128_WAIT_SWING_REWARD\
   rolling_slot_0.pt hoặc rolling_slot_1.pt
   state.json
   experiment.json
   run_config_blueprint.json
   metrics.jsonl   (rỗng)
```

Log phải in `[V14 REWARD FORK] inherited EP... FULL checkpoint; no weights/Adam/ValueNorm reset` và `[V14 REWARD INIT ONLY]`.

## 6. Train đúng 1 EP để kiểm tra reward

```powershell
$new = '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_WAIT_SWING_REWARD'
$st = Get-Content "$new\state.json" -Raw | ConvertFrom-Json
$oneEP = [int]$st.last_episode + 1
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_reward_stop_swing.py --output "$new" --device cuda:0 --sim-seconds 3600 --max-minibatch 128 --max-episodes $oneEP --resume
```

Sau 1 EP, kiểm tra ngay **đúng số liệu của EP mới**:

```powershell
$last = Get-Content "$new\metrics.jsonl" -Tail 1 | ConvertFrom-Json
$last.environment.reward_shaping | Format-List
$last.environment.action_diagnostics | Format-List
$last.environment.statistics | Select-Object jobDone,block,liveLock,wrongTarget,targetReached
```

Các trường quan trọng:

- `apparently_unnecessary_stops_selected`: số Stop được phân loại là có thể tránh (heuristic).
- `stop_penalties_applied`: số Stop đã nhận −0.5 vào reward khi observation kế tiếp xuất hiện.
- `swing_penalties_applied`: số lần A→B→A nhận −1.0.
- `total_reward_adjustment`: tổng phạt cộng thêm (âm hoặc bằng 0).
- `actions_by_role.routing.0`, `routing_stop_with_move_available`, `routing_move_available_decisions`: so tần suất Stop ở các run trên **cùng điều kiện**.
- `ppo_audit`, `all_roles_audit`, `selected_mask_violations`: phải không có lỗi PPO/mask.

**Nếu `stop_penalties_applied=0` trong 1 EP:** có thể Stop không được chọn, được xác định là chờ cần thiết, hoặc hành động chờ cuối EP không có observation tiếp theo. Chưa thể kết luận code không hoạt động. Đọc `protected_wait_*` và test offline; đừng vội tăng mức phạt.

## 7. Sau khi 1 EP hợp lệ, resume đến EP1100

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_reward_stop_swing.py --output '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_WAIT_SWING_REWARD' --device cuda:0 --sim-seconds 3600 --max-minibatch 128 --max-episodes 1100 --resume
```

`--max-episodes 1100` là **mốc tuyệt đối**, không có nghĩa chạy thêm 1100 EP. Không thêm `--fork-from` với `--resume`.

Nếu muốn thử hệ số khác, tạo run **MỚI** với `--output` khác và tùy chọn `--stop-penalty`, `--swing-penalty` khi fork, không sửa hệ số giữa chừng một run đã học.

## 8. Đánh giá và rollback

Để kết luận reward mới có hiệu quả cần so sánh held-out cùng seed và cùng 3600s: `JobDone` (quan trọng nhất), `Block`, `LiveLock`, tổng `_blocksec_agv*`, `WrongTarget`, tỉ lệ Stop khi có thể đi, số lần swing. **Không so trực tiếp tổng reward giữa hai reward schemes**, vì mục tiêu tối ưu đã thay đổi. Hiện tại chưa cung cấp evaluator chuyên dụng cho reward-run này; cần xác minh evaluator phù hợp checkpoint V14 đã fork trước khi chạy đánh giá.

Muốn quay về reward cũ: **chạy lại trainer V14 STOP-UNLOCK cũ với `--resume` và `--output` cũ**, không nạp checkpoint của reward-run vào folder V14 cũ. Trong tiến trình trainer cũ, wrapper reward mới không được cài đặt nên reward trở về như trước. Thư mục GitHub backup code cũng hoàn toàn không bị đụng tới.

### Giới hạn đáng lưu ý

- `apparently_clear_candidate` không chứng minh đường đi an toàn hay Stop vô ích; phần proxy chỉ dùng action mask và tọa độ hiện tại/waypoint. Mục tiêu là phạt thận trọng, không khuyến khích AGV lao vào kẹt xe.
- Swing A→B→A có thể xảy ra hợp lý nếu đổi mục tiêu, đổi dispatch hoặc né xe khác; phạt là giả thuyết cần kiểm chứng, không phải quy tắc tuyệt đối.
- Cấu trúc bản mới không cố gán reward bổ sung cho một completed PPO transition bằng cách sửa buffer. Nó thay reward ngay tại nguồn `MultiAgent.convert_from_observation`, sau đó collector tiêu thụ reward như cũ. Nếu agent không được gọi lại trước khi episode kết thúc, không có reward giả tạo.
- Chưa thử chạy online trên AnyLogic Windows. Cần kiểm tra kết quả EP đầu trước khi train dài và giữ backup checkpoint gốc.
