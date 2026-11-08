# V14 — GAT4 + Deep Candidate-Aware MLP Dispatch + 3 Centralized MLP Critics

**Ngày:** 2026-11-02 · **20 AGV · MAPPO · 3600 s/episode · Full scratch**

## 1. Điểm khác so với V12/V13

| Thành phần | V14 |
|---|---|
| Routing Actor | 4 lớp Congestion-Aware Residual Edge-GAT, 2 heads mỗi lớp (mặc định); 6 kênh traffic quan sát của V12 |
| Dispatch Front Actor | Deep Candidate-Aware MLP, chấm riêng từng station slot, action mask theo `operation_first` |
| Dispatch Rear Actor | Một mạng Deep Candidate-Aware MLP **khác trọng số** với Front |
| Routing + Front + Rear Critics | 3 mạng Centralized Deep MLP tách biệt; **value head** gồm 3 hidden layers 128–128–64, sau đó Linear 64→1 |
| Dijkstra routing guidance | **Không**. Không next-hop/static shortest path dự đoán trong traffic |
| Độ dài rollout | 3600 giây thời gian mô phỏng, max_events thêm giới hạn bảo vệ |
| Khởi tạo | **Không tải .pt cũ**. EP0 là checkpoint random; EP1 là lần train đầu tiên |
| Trạng thái/đầu ra | Riêng ở `mappo_runs/V14_GAT4_CANDIDATE_MLP_FULL_SCRATCH_3600S` |

**Quan trọng**: `--gat-layers 4` thật sự được truyền vào `v8.train_args` và builder kiểm tra đủ 4 lớp Residual Edge-GAT; không chỉ thay metadata. V14 **không** dùng Transformer cho Dispatch hoặc Critic.

### Dispatch đầu vào / đầu ra

- `agvs [B,20,F]`, `stat [B,27,F]`, `action_mask [B,27]` được kế thừa đúng pipeline Matrix/PettingZoo hiện tại. F lấy trực tiếp từ model gốc.
- `ego_encoder`: 3 Linear F→128→128→64; `other_encoder`: 3 Linear (ego + other)→128→128→64; `station_encoder`: 3 Linear (ego + station + station-ID embedding)→128→128→64.
- Có **masked mean/max** để tổng hợp trạng thái hệ thống; giữ nguyên embedding trạm `j`. `station_identity` dùng 16 chiều và trạm được đánh chỉ số ổn định theo 27 action slots.
- Shared scorer cho mỗi trạm: `[h_ego, h_other_mean,max, h_stat_mean,max, h_station_j, h_ego*h_station_j]` → **128→128→64→1**, tạo 27 logits.
- Action mask gốc **không đổi**; với trạm không hợp lệ, logit bị che bằng `masked_logits`. Vẫn giữ `forbid_when_move` cho Stop Routing.

### Centralized Critic

- Mỗi role có Critic riêng: nhận `get_mappo_state()` gồm **AGV vật lý không bị shuffle**, trạm và `ego` index.
- AGV encoder và Station encoder dùng 3 Linear: F→128→128→64. Global state có masked mean/max; value head **320→128→128→64→1**.
- Khởi tạo mới optimizer Adam và RunningValueNorm cho cả ba Critic; CTDE vẫn giữ nguyên.

## 2. Giải nén vào đúng thư mục dự án

Dự án local: `C:\MARL_AnyLogic\MAPPO_NEW\thesis`.

Giải nén ZIP **vào gốc dự án**, sao cho tồn tại:

```
C:\MARL_AnyLogic\MAPPO_NEW\thesis\
  scripts\train_v14_deep_mlp_scratch.py
  scripts\evaluate_v14_deep_mlp.py
  scripts\v14_deep_mlp_common.py
  scripts\v14_deep_mlp_collector.py
  thesis\mappo\v14_deep_candidate_mlp.py
  tests\test_v14_deep_mlp_offline.py
```

ZIP cũng đi kèm ba file V12 dùng làm dependency (`congestion_edgegat_v12_clean.py`, `v12_clean_audit.py`, `v12_clean_collector.py`) với nội dung không thay đổi. **Không có checkpoint cũ**, không có AnyLogic model ZIP, không có source V8/V11 khác; phải chạy trên dự án MAPPO đã dùng để train V12.

Hãy sao lưu thư mục dự án trước khi giải nén. File nào trùng trong bộ V12 phải cùng phiên bản hiện tại; kiểm tra diff nếu từng sửa thủ công.

## 3. Kiểm tra môi trường

Tắt các Python / Alpyne engine đang train qua port `51880` trước khi chạy. AnyLogic PLE GUI dùng để xem animation là chuyện **khác** với engine headless dùng training.

PowerShell (VS Code):

```powershell
cd C:\MARL_AnyLogic\MAPPO_NEW\thesis

# Python hỗ trợ torch + CUDA
.\.venv311\Scripts\python.exe -c "import torch; print(torch.__version__, 'CUDA=', torch.cuda.is_available())"

# Kiểm tra blueprint V8 và mô hình AnyLogic ZIP gốc
Test-Path .\mappo_runs\V8_RESIDUAL_EDGEGAT_JOBDONE2_FROM205\run_config.json

# 12 test offline không khởi động AnyLogic
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p "test_v14_deep_mlp_offline.py" -v

# Đọc và xác nhận thông số; không ghi run
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_deep_mlp_scratch.py --device cuda:0 --sim-seconds 3600 --dry-run
```

Nếu báo `ModuleNotFoundError` cho module V8/V11 (`train_B3_MAPPO_v8_overnight`, `thesis.mappo.*`) thì bản MAPPO đang thiếu các file nguồn kế thừa; không dùng nguyên `thesis.zip` của Tobias làm thay thế.

## 4. Tạo EP0 trước để xác minh

**Khuyên thực hiện EP0 trước khi train dài.** Đây là khởi tạo mạng ngẫu nhiên hoàn toàn; cần một lần khởi động AnyLogic/Alpyne để dựng graph rồi lưu trọng số.

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_deep_mlp_scratch.py `
  --device cuda:0 --sim-seconds 3600 --gat-layers 4 --gat-heads 2 `
  --dispatch-width 128 --embedding 64 --max-minibatch 16 --init-only
```

Kỳ vọng có log: `[V14 EP0] randomly initialized checkpoint saved...` và `[V14 INIT ONLY] ...`. File mới `checkpoint_ep_0000.pt` được tạo.

**EP0 phải có đủ:**
- `routing.actor.layers` đúng **4 GAT attention layers** (runner tự assert)
- `dispatch_front.actor`, `dispatch_rear.actor` dùng candidate-aware MLP
- `routing.critic`, `dispatch_front.critic`, `dispatch_rear.critic` là 3-layer deep MLP
- tất cả tham số trainable, không Dijkstra và không `next_hop`

## 5. Test EP1 rồi train dài

Vì EP0 đã tạo ở bước 4, chạy **resume** thay vì khởi tạo lại thư mục:

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_deep_mlp_scratch.py `
  --device cuda:0 --sim-seconds 3600 --gat-layers 4 --gat-heads 2 `
  --dispatch-width 128 --embedding 64 --max-minibatch 16 `
  --max-episodes 1 --resume
```

Quan sát có `[V14 EP1] ...` và trong `metrics.jsonl` có `policy_updates`, `ppo_audit`, `all_roles_audit`, `graph_mapping`, `dispatch_choice_diagnostics`.

Khi EP1 chạy xong, tiếp tục:

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_deep_mlp_scratch.py `
  --device cuda:0 --sim-seconds 3600 --gat-layers 4 --gat-heads 2 `
  --dispatch-width 128 --embedding 64 --max-minibatch 16 `
  --max-episodes 1100 --resume
```

**Nếu muốn bỏ qua bước EP0 và chạy từ đầu ngay**, chỉ chạy lệnh trên nhưng **bỏ `--resume` và `--init-only`** trong một thư mục V14 chưa tồn tại.

- `--max-episodes` là **episode cuối cùng tuyệt đối**, không phải số episode tăng thêm. EP0 khởi tạo, EP1 mới cập nhật PPO.
- `--max-minibatch 16` là lựa chọn an toàn hơn cho GPU 4GB với **GAT4**. Nếu máy vẫn OOM, cần thiết kế run mới với tham số khác; runner khóa hyperparameters để tránh resume tạo thí nghiệm không nhất quán.
- Nếu bị `max_event_steps`: episode không đi hết được điều kiện kết thúc, runner **chặn PPO update** để tránh âm thầm train trên episode bị cắt. Hãy tăng `--max-events` trong **một run mới**, ví dụ `--max-events 15000` (số này là giới hạn event, không phải giây).

## 6. Evaluate 5 held-out seeds, 3600s, không training

Không chạy đồng thời với train trên cùng cổng. Dùng cổng `51881`:

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\evaluate_v14_deep_mlp.py `
  --device cuda:0 --port 51881 --sim-seconds 3600 `
  --seeds 900001 900002 900003 900004 900005
```

Mặc định lấy checkpoint đang được `state.json` cam kết. Hoặc thêm `--checkpoint` trỏ đến snapshot `.pt` bên trong đúng thư mục V14.

Kết quả ở `mappo_runs\V14_GAT4_CANDIDATE_MLP_FULL_SCRATCH_3600S\heldout_*.jsonl`.

KPI: JobDone, Block, LiveLock, WrongTarget, Block time, reward. Dùng **cùng 3600s và cùng seeds** để đối chiếu với V12 nếu muốn kết luận V14 tốt hơn. So sánh trực tiếp với held-out V12 1800s trước đây sẽ thiếu công bằng.

### Chỉ số riêng để kiểm tra giả thuyết Dispatch

`environment.dispatch_choice_diagnostics.dispatch_front.multi_choice_rate` và `dispatch_rear.multi_choice_rate`: tỷ lệ lần ra quyết định có >=2 action hợp lệ. Nếu đa phần mask chỉ cho phép đúng một trạm, tăng số lớp Dispatch **không thể cải thiện lựa chọn ở chính những lần đó**. Hãy xem cả tỷ lệ này, không chỉ Reward.

## 7. Resume an toàn

`--resume` chỉ nhận checkpoint đã được xác minh bởi `state.json` và nằm trong run V14. Không nạp EP830 V12 vào GAT4/MLP V14; số lớp và tên tham số khác nhau. Các trường seed/sim_seconds/architecture/device/port/batch được khóa.

Nếu muốn đổi hẳn cấu hình (ví dụ GAT 2 lớp, widths khác, minibatch khác), tạo **run mới** và đừng ghi đè.

## 8. Kiểm thử đạt được và giới hạn

- CPU offline **12/12 unit tests**: logits/gradient, masked actions, station-sensitive scoring, pooling permutation, critic values, fresh optimizer+ValueNorm, state_dict, EP/resume và dry-run.
- `py_compile` thành công cho mọi file Python mới.
- **Chưa chạy end-to-end AnyLogic/Alpyne 3600s ở máy Windows của ông**, nên chưa thể tuyên bố training EP1 hoặc JobDone đã cải thiện.
- V14 ZIP **chưa có connector desktop Animation**. Khi train xong muốn xem bằng AnyLogic PLE (`runmode=3`), cần connector V14 riêng; connector V12 EP830 hiện tại không tương thích checkpoint này.

### Tài liệu tham khảo

- Yu et al. (2022), *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS. https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract.html
- Steinbacher/Wegmann et al., *Production control with Reinforcement Learning for a matrix-structured production system*; Tobias original GitHub https://github.com/tobihoernchen/thesis
- Các file source hiện tại: `thesis/mappo/congestion_edgegat_v12_clean.py`, `thesis/mappo/edge_gat_routing_v4.py`, `thesis/mappo/networks.py`, `thesis/mappo/algorithm.py`.
