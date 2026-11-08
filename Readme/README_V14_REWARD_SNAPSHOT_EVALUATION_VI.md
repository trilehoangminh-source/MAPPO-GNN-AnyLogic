# V14 Reward — Đánh giá SNAPSHOT EP110, EP120, EP130 (10 seed giống nhau)

**Mục đích:** so sánh checkpoint trong thí nghiệm `V14_GAT4_MLP_3600S_MB128_WAIT_SWING_REWARD` với **10 seed held-out**, **20 AGV**, **3600 giây mô phỏng/episode**. Không sửa reward, không huấn luyện tiếp và **không ghi đè** checkpoint, `state.json`, `metrics.jsonl`, hay các thư mục V14 cũ.

## 1. File trong ZIP

```
scripts/evaluate_v14_reward_snapshots.py
tests/test_v14_reward_snapshot_eval_offline.py
README_V14_REWARD_SNAPSHOT_EVALUATION_VI.md
```

Giải nén ZIP **vào thư mục gốc** `C:\MARL_AnyLogic\MAPPO_NEW\thesis` (chỗ có thư mục `scripts`, `tests`, `thesis`, `.venv311`). ZIP này **chỉ thêm ba file**, không ghi đè source V14 hiện hành.

Yêu cầu: bộ code Reward Stop/Swing phiên bản cũ đang chạy thành công, có `scripts/train_v14_reward_stop_swing.py`, `scripts/v14_reward_collector.py`, `thesis/envs/v14_reward_shaping.py`, cùng V14 base code. Các snapshot `.pt` **không nằm trong ZIP** mà ở trên máy Windows.

## 2. Cấu hình thí nghiệm cố định

- 3 checkpoint: `snapshot_ep_0110.pt`, `snapshot_ep_0120.pt`, `snapshot_ep_0130.pt`
- Cùng 10 seed: **900001 900002 900003 900004 900005 900006 900007 900008 900009 900010**
- 20 AGV, 3600 giây mỗi seed; cùng topology/config, cùng reward Stop/Swing đã lưu trong manifest.
- Action `greedy=True` qua `collector.episode(train=False)`: không phải lấy action ngẫu nhiên khi đánh giá; không cập nhật Actor, Critic, Adam hay ValueNorm.
- Giữ action mask `baseline` (Stop không bị khóa); reward patch gắn trong tiến trình evaluator giống như khi training.
- Đánh giá tuần tự: EP110×10 seed, EP120×10, EP130×10 (30 episodes). Không chạy song song với trainer khi GPU/Java đang áp lực.

**Lưu ý:** `900001..900010` cần khác seed train; runner gốc tăng seed theo episode từ `experiment.json`. Có thể kiểm tra khoảng seed đã huấn luyện trước khi chọn nếu từng dùng các seed này cho thí nghiệm khác. Cùng seed giữa checkpoint là thiết kế paired comparison; không có nghĩa mọi quyết định sau bước đầu giống nhau.

## 3. Tạm dừng training an toàn (nếu còn chạy)

Trong terminal training, nhấn **Ctrl+C một lần** và chờ dòng `[V14 REWARD DONE] committed EP ...`. Không kill thô Java và không sửa checkpoint đang được ghi. Mở terminal PowerShell mới:

```powershell
cd C:\MARL_AnyLogic\MAPPO_NEW\thesis
```

Đầu tiên kiểm tra có ba snapshot:

```powershell
$run = '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_WAIT_SWING_REWARD'
Test-Path "$run\snapshot_ep_0110.pt"
Test-Path "$run\snapshot_ep_0120.pt"
Test-Path "$run\snapshot_ep_0130.pt"
```

Cả ba cần trả về `True`. Nếu có `False`, kiểm tra `snapshot_every` và danh sách `state.json`; tuyệt đối không dùng tên file có vẻ giống hay lạm dụng `rolling_slot_0/1.pt` để giả làm snapshot EP cũ (rolling bị ghi đè sau mỗi EP).

## 4. Test offline và dry-run

**Chỉ copy các lệnh giữa dấu khung code; KHÔNG copy dòng có ba dấu backtick.**

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p 'test_v14_reward_snapshot_eval_offline.py' -v
```

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\evaluate_v14_reward_snapshots.py --device cuda:0 --dry-run
```

`--dry-run` kiểm tra SHA256 của bản reward patch, AnyLogic ZIP, run blueprint và cả ba `.pt`; in `[DRY RUN OK]` nếu mọi thứ khớp, không chạy mô phỏng/không tạo output.

Nếu báo `Reward implementation changed`, có thể file `v14_reward_shaping.py` hiện tại khác khi khởi tạo run. **Không xóa hoặc tự sửa manifest** để vượt lỗi; cần xem diff và dùng đúng phiên bản reward nguồn. Nếu báo `Missing exact immutable snapshot`, cần checkpoint gốc thật.

## 5. Chạy đánh giá đủ 30 episode

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\evaluate_v14_reward_snapshots.py --device cuda:0 --port 51881
```

Mỗi khi xong một lượt sẽ xuất hiện dòng tương tự (con số dưới đây chỉ mô tả nhãn log, không phải kết quả thực tế):

```
[EVAL EP110 seed=900001] JobDone=... Block=... LiveLock=... Stop%=... BlockTime=...s
```

Nếu điện/máy/AnyLogic bị ngắt sau vài seed, sau khi khôi phục môi trường, **chỉ** chạy tiếp:

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\evaluate_v14_reward_snapshots.py --device cuda:0 --port 51881 --resume
```

`--resume` đọc JSONL hiện có, xác nhận hash checkpoint/script/reward và bỏ qua seed đã hoàn tất. Nếu ngắt giữa **một dòng JSONL** khi mất điện, code sẽ dừng vì dòng hỏng: sao lưu file đánh giá và kiểm tra dòng cuối, không đụng training metrics.

## 6. Dữ liệu kết quả

Trong thư mục run V14 Reward sẽ có:

- `heldout_reward_ep110-120-130_10seeds_3600s.jsonl`: từng lượt đánh giá, kèm checkpoint SHA, seed, KPIs và chẩn đoán.
- `heldout_reward_ep110-120-130_10seeds_3600s_per_seed.csv`: bảng 30 dòng.
- `heldout_reward_ep110-120-130_10seeds_3600s_summary.csv`: trung bình và độ lệch chuẩn từng checkpoint.
- `heldout_reward_ep110-120-130_10seeds_3600s_paired.csv`: mức chênh EP120−EP110 và EP130−EP110 theo **các seed chung**, có khoảng tin cậy xấp xỉ khi đủ 10 cặp.

KPIs chính: `jobDone` (cao tốt), `block`, `liveLock`, `block_time_s`, `wrongTarget` (thấp tốt); theo dõi thêm `targetReached`, `stop_ratio_pct`, `stop_penalties`, `swing_penalties`. `block_time_s` là **tổng tất cả `_blocksec_agv*`**, có thể vượt 3600 do cộng thời gian của 20 xe. Reward chỉ so trong cùng chế độ reward shaping, và không được coi là KPI duy nhất.

`model_time_s` trong CSV là đồng hồ mô phỏng cuối lấy từ `run.get_state` nếu engine cho đọc; `last_decision_time_s` chỉ là thời điểm decision cuối, **không** phải thời điểm dừng của episode. Nếu không lấy được model_time, cần xác minh thời lượng thực từ engine/experiment trước khi tuyên bố cả 30 run đúng 3600s. Script chặn episode cắt ở `max_event_steps` hoặc không kết thúc, và chặn khi đồng hồ thực đọc được cho thấy dừng trước 3600s.

**Điểm rất quan trọng:** evaluator dùng 10 seed chưa thấy trong training và greedy policy, nên kết quả này mới phù hợp hơn để chọn snapshot. Trong môi trường stochastic, số lượng 10 seed vẫn có độ biến động; paired CI chỉ mô tả bất định mẫu nhỏ. Dữ liệu EP110/120/130 trong `metrics.jsonl` cũ **không thay thế** kết quả đánh giá này.

## 7. Cách chọn checkpoint tốt nhất

Đọc `summary.csv` kết hợp `paired.csv`: ưu tiên checkpoint có **JobDone cao hơn** mà **Block, LiveLock và tổng Block time không tăng đáng kể**. Nếu EP120 cho throughput cao nhưng Block time kéo dài, phải cân nhắc trade-off; tránh chọn checkpoint chỉ do một seed tốt hay raw Reward cao. Nếu chênh lệch nhỏ và khoảng tin cậy rộng, có thể cần thêm seed.

**Giới hạn kiểm thử:** đã kiểm tra `py_compile` và 7/7 unit tests offline của evaluator. Chưa thể chạy 30 lượt AnyLogic và chưa kiểm chứng snapshot thật vì `.pt` và model AnyLogic ZIP chỉ có trên máy ông.
