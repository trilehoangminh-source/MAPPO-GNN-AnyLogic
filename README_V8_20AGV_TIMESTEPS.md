# MAPPO V8 — Tiếp tục train 15 -> 20 AGV và vẽ Training Timesteps

**Dùng trên repo `C:\MARL_AnyLogic\MAPPO_NEW\thesis` đã cài V8 15 AGV thành công.**

## 1. Thay đổi gì?

- `scripts/train_B3_MAPPO_v8_overnight.py`: chấp nhận `--from-previous-run` từ run V8 của **15 AGV** (không còn chỉ cho phép run 8 AGV). Nạp **rolling checkpoint đã commit** sau Episode 144, xác nhận topology/fleetsize/gamma/guide và huấn luyện **20 AGV**, không reset trọng số/optimizer.
- `--max-events` (tùy chọn) cho phép nâng trần số sự kiện nếu episode bị cắt trước khi đủ thời gian mô phỏng. Mặc định vẫn giữ từ cấu hình gốc; không tùy tiện sửa giữa phiên.
- `scripts/plot_mappo_training_timesteps.py`: vẽ 6 panel với trục X là **tổng `environment.closed_transitions` trong các episode đã ghi log**, KHÔNG phải episode, KHÔNG phải số giây mô phỏng và KHÔNG đơn giản nhân số episode với hằng số.
- Giữ nguyên mô hình Edge-GAT, Dispatch B2 MLP, Dijkstra 1.5, MAPPO PPO clipping/HUBER/KL. Không sửa checkpoint cũ và Java AnyLogic.

## 2. Cài đặt

Giải nén ZIP vào **root repo** `C:\MARL_AnyLogic\MAPPO_NEW\thesis`, ghi đè **chỉ** file `scripts/train_B3_MAPPO_v8_overnight.py` và `_mappo_v8_utils.py` (helper cùng bản trước), thêm 2 file `plot_mappo_training_timesteps.py` và test.

```powershell
python tests/test_V8_20AGV_timesteps_offline.py
```

Kỳ vọng: `V8 20 AGV + TRAINING TIMESTEPS OFFLINE: PASS`. Offline PASS không chứng minh AnyLogic 20 AGV chạy hay đã học tốt hơn.

## 3. Xác nhận Episode 144 còn nguyên

```powershell
$prev = Get-Content "mappo_runs/V8_15AGV_fromV7_5h/state.json" -Raw | ConvertFrom-Json
"Episode cuối: $($prev.last_episode)"
"Checkpoint tồn tại: $(Test-Path $prev.checkpoint)"
```

Phải là Episode 144 (hoặc episode cuối khác nếu đã tiếp tục thêm) và checkpoint `True`. Không sửa bất kỳ file nào trong thư mục V8 15 AGV.

## 4. Dry-run 15 -> 20

```powershell
python scripts/train_B3_MAPPO_v8_overnight.py `
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
  --dry-run
```

Xác nhận: `Transfer AGVs= 15 -> 20 max= 20`, `Initial episode: 144` (trường hợp hiện tại).

## 5. Train chính thức tối đa khoảng 5 giờ theo ngân sách clock

Đóng animation AnyLogic đang chạy và những lệnh training khác. Chạy **chính xác lệnh trên, nhưng bỏ `--dry-run`**, có thể thêm `-u` sau `python` để log ra liên tục:

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
  --output mappo_runs/V8_20AGV_from15_5h
```

Mô hình được nạp checkpoint Episode 144 từ 15 AGV, episode đầu tiên của đợt 20 AGV sẽ là 145. Nó lưu `state.json`, hai rolling checkpoints, metrics.jsonl, snapshots. Cuối đợt `final_selection` đánh giá cả nguồn 15 AGV **trên cấu hình 20 AGV** rồi so với giữa / cuối đợt 20 AGV, tất cả trên cùng 2 eval seeds. Nếu evaluation hoàn tất có `best_mappo.pt` và `best_info.json`. Không lấy reward episode cuối làm bằng chứng chọn tốt nhất.

**Thận trọng:** `--hours 5` được kiểm tra giữa các episode, quá trình evaluation hoặc một episode còn dở có thể làm chạy lâu hơn 5h. Việc huấn luyện thêm không bảo đảm tất cả KPI đều cải thiện.

### Resume nếu gián đoạn

Chạy lại với nguyên mọi tham số + `--resume` (giữ nguyên `--from-previous-run` và `--output`). Không trộn thư mục V8 15 và V8 20. Nếu script mất ngay trong episode, chỉ mất phần chưa commit.

### Nếu quá nhiều `max_event_steps` trước mốc 1800 giây

Kiểm tra `mappo_runs/V8_20AGV_from15_5h/metrics.jsonl` ở vài episode đầu (trường `environment.end_reason`). Nếu thấy liên tục bị cắt do giới hạn sự kiện hoặc `environment_terminal` không đạt mốc, cần phân tích kết thúc episode trước khi chỉnh `--max-events` ở **một run mới**, không được tự sửa rồi `--resume` giữa chừng.

## 6. Vẽ biểu đồ timestep

```powershell
python -m pip install matplotlib

# Có thể chạy ngay, từ data V8 15 AGV hiện có:
python scripts/plot_mappo_training_timesteps.py `
  --folders mappo_runs/V8_15AGV_fromV7_5h `
  --window 7 `
  --out figures/V8_15AGV/Figure_TrainingTimesteps_6panels

# Sau khi bắt đầu / kết thúc V8 20 AGV:
python scripts/plot_mappo_training_timesteps.py `
  --folders mappo_runs/V8_15AGV_fromV7_5h mappo_runs/V8_20AGV_from15_5h `
  --window 7 `
  --out figures/V8_15to20AGV/Figure_TrainingTimesteps_6panels
```

Đầu ra `.png`, `.pdf`, và `training_timesteps_data.csv` chứa episode, số transition của episode và số transition tích lũy.

**Về ý nghĩa trục X:** cộng `closed_transitions` trên các episode đang có log. Vì không có số transition trước episode đầu tiên của run V8 15, tổng này được tính **từ episode đầu được ghi trong V8 15**, không phải từ Episode 1/V7. Cách đếm này có thể khác với *Training Steps* do nhóm tác giả định nghĩa; muốn đối chiếu định lượng tuyệt đối hai đường X thì phải thống nhất đơn vị steps. Các đường ứng với 15 và 20 AGV nằm trong **hai điều kiện môi trường khác nhau**, không phải cùng một run đồng nhất.

## 7. Nạp desktop 20 AGV

**Chưa có** bản connector desktop 20 AGV được kiểm thử trong patch này. Script train tạo checkpoint V8 20 AGV, **chưa tự đổi** `RL_pypeline_name` hay checkpoint của bản demo 15 AGV. Chỉ nạp sau khi đã kiểm tra checkpoint, graph fingerprint và chạy thử riêng trên AnyLogic.
