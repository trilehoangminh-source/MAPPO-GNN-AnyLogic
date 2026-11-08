# V15 — Tobias-style GATv2 + LinFE Dispatch + Centralized MAPPO

## 1. Mục tiêu và phạm vi

Phiên bản V15 **train mới từ Episode 0**, mỗi episode **3.600 giây mô phỏng**. Không đọc bất kỳ checkpoint V14 nào và không ghi đè thư mục `mappo_runs` của V14.

V15 dành để thử giả thuyết: kiến trúc Actor đơn giản, gần kiến trúc tác giả Tobias, kết hợp cùng MAPPO ổn định từ V8/V14, có tạo chính sách tốt hơn bản V14 phức tạp không? **Chưa có kết quả AnyLogic thực tế**, không tuyên bố đã cải thiện Block/LiveLock.

### Thiết kế mạng

- Routing Actor: **2 lớp GATv2, mỗi lớp 4 heads, embedding 32**, GraphNorm sau mỗi GAT; Fourier features 4; node/goal embedder 2 Linear (64,32); action head **160→128→32→5** (Stop, 4 hướng). Giữ điều kiện action mask gốc; **không có Dijkstra logit bias**.
- Dispatch Front / Dispatch Rear: hai policy **khác trọng số, cùng kiến trúc** LinFE: ego `input→64→64→64→32`; other AGV `input→128→128→64→32`; stations `input→128→128→64→32`. Pooling mean/max theo lối Tobias, ghép 5 vector × 32, action head **160→128→32→N_station**. Trong cấu hình V14 của ông có 27 slot trạm/buffer. Giữ `operation_first` action mask.
- Ba centralized critics: **MLP riêng theo role**, global state (AGV vật lý không shuffle + station + physical ego ID), encoder 32 chiều, value head **160→128→128→32→1**. *Đây là thiết kế của V15 phù hợp CTDE, không phải value head cục bộ nguyên xi của Tobias.*
- MAPPO: **giữ nguyên file `algorithm.py`, `mappo_stability_v7.py`, `collector_stop_ab.py`**, GAE, PPO clipping, ValueNorm, KL guard, gradient clipping, minibatch guard và optimizer. Không sửa thuật toán PPO để làm V15.
- Không dùng Residual GAT, congestion channels, Candidate-Aware MLP, Transformer, Stop/Swing/Safe-Yield/Stall shaping, `GuidedMAPPO` hay `GuidedEventCollector`.

**Giới hạn độ tương đương:** file `v15_tobias_arch.py` là *bản triển khai lại bằng PyTorch thuần* từ **mẫu kiến trúc GATv2/LinFE trong source Tobias**, không phải import trực tiếp `torch_geometric.GATv2Conv`, `torch_scatter`, RLlib. Kiến trúc và luồng đầu vào được bám sát, nhưng trọng số/chi tiết phép toán có thể khác; **không dùng checkpoint Tobias và không gọi là sao chép số học 1:1**. Nếu cần chứng minh parity, phải so sánh output của cả hai implementation bằng test chuyên biệt.

## 2. Reward gốc như notebook Tobias

Đã xem `exploration/MiniMatrix/training_disp_rout.ipynb` trong `thesis.zip`. Các hệ số ở notebook huấn luyện kết hợp:

| Thuộc tính AnyLogic | Giá trị |
|---|---:|
| `reward_reached_target` | +10 |
| `reward_wrong_target` | -1 |
| `reward_removed_for_block` | -20 |
| `reward_target_distance` | 0 |
| `reward_invalid` | -0.1 |
| `reward_duration` | -0.5 |
| `reward_pass` | +0.2 |
| `reward_accepted_in_station` | +0.5 |
| `reward_declined_in_station` | -0.25 |
| `reward_geo_operation` | +0.25 |
| `reward_reduce` | -0.005 |
| `block_timeout` | 60s |
| `direction_reward` (Python) | -0.1 |

**Quan trọng:** `reward_part_completed` trong notebook tác giả bị **comment**, do đó V15 **không ghi đè** biến này. Giá trị mặc định của AnyLogic nếu có sẽ quyết định. Không dùng +2 như V14.

File chứa cấu hình: `scripts/v15_tobias_build.py`.

## 3. Giai đoạn huấn luyện (mỗi phase có EP0 riêng)

| Phase | Fleet | Routing thực thi | Dispatch thực thi | PPO cập nhật |
|---|---:|---|---|---|
| `routing8` | 8 | GATv2 | Random station hợp lệ | Chỉ Routing Actor + Critic |
| `dispatch8` | 8 | **Conventional Routing gốc Tobias** (`CollisionFreeRouting`, đặt trước khung thời gian trên cung) | 2 LinFE MLP | Chỉ Dispatch Front/Rear Actor + Critics |
| `joint8` | 8 | GATv2 | 2 LinFE MLP | Cả ba role |
| `joint12` | 12 | GATv2 | 2 LinFE MLP | Cả ba role |
| `joint16` | 16 | GATv2 | 2 LinFE MLP | Cả ba role |
| `joint20` | 20 | GATv2 | 2 LinFE MLP | Cả ba role |

**Phân biệt rõ theo yêu cầu đã chốt:** Routing Actor GATv2 trong `routing8` và các phase `joint*` **hoàn toàn không sử dụng Dijkstra guidance/logit bias**. Riêng khi huấn luyện `dispatch8`, mô hình AnyLogic được khởi tạo với `pseudo_routing=True` để chính `Matrix` gọi **`CollisionFreeRouting` cùng `CollisionFreeRoutingHive` của Tobias**. Bộ định tuyến này tìm tuyến trên đồ thị có xét **cửa sổ thời gian còn trống** và **đặt trước các cạnh xung đột**. Đây là thuật toán truyền thống, không phải mạng GAT và không tham gia tối ưu PPO. Nó có tìm đường theo nhãn/khoảng cách thời gian; yêu cầu "không Dijkstra" của ông áp dụng cho **GAT learned-routing**, không phải cơ chế Conventional Routing phục vụ Dispatch-only.

**Không còn** `masked_geometric_routing()`/bộ định tuyến Euclid một bước của V15 trước. Khi `dispatch8`, toàn bộ các Routing decisions được Auto-sim xử lý, **không xuất hiện trong collector PPO**; chỉ hai Dispatch MLP và hai Dispatch Critics được cập nhật. Nếu Routing lọt vào collector ở phase này, chương trình dừng với lỗi chứ không âm thầm dùng heuristic khác.

Tui lấy chính `CollisionFreeRouting`/`CollisionFreeRoutingHive` từ `thesis/envs/behaviors_v121.py` có sẵn trong repo (kế thừa source Tobias). `BaseAlpyneZoo._collect()` chạy quyết định Routing nội bộ khi `is_for_learning()==False`. Không cần patch Java, MAPPO hay Collector.

**Không có CACTUS/Confidence tự động trong ZIP này:** tăng 8→12→16→20 theo lộ trình đã thống nhất và chỉ do người nghiên cứu quyết định sau khi xem KPI. Khi có baseline V15, mới thử bản confidence-gated riêng để kiểm chứng tác động curriculum.

## 4. Cài đặt (Windows PowerShell, VS Code)

1. Dừng training V14 sau khi nó commit episode và **sao lưu nhánh/source hiện tại**. ZIP V15-V2 **thêm/cập nhật các file V15**, không ghi đè file MAPPO/V14 gốc. Nếu đã cài V15-V1, bản này sẽ thay các file V15 cùng tên.
2. Giải nén ZIP vào gốc repo: `C:\MARL_AnyLogic\MAPPO_NEW\thesis` sao cho các file nằm ở đúng đường dẫn `scripts/` và `thesis/mappo/`.
3. Mở terminal tại gốc repo, kích hoạt môi trường `.venv311`. Code dùng PyTorch mới có `scatter_reduce_` (khuyến nghị PyTorch >=2.0).

### Bước A — offline tests

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p 'test_v15_tobias_staged_offline.py' -v
```

### Bước B — kiểm tra cấu hình chưa mở AnyLogic

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v15_tobias_staged.py --phase routing8 --dry-run
```

Nếu không tự tìm thấy `run_config_blueprint.json` V14, thêm:

```powershell
--blueprint '.\mappo_runs\TEN_RUN_V14\run_config_blueprint.json'
```

Nếu đường dẫn `.zip` của model AnyLogic đã thay đổi, thêm:

```powershell
--model '.\envs\TEN_FILE_MODEL.zip'
```

Runner **chỉ đọc** blueprint môi trường, topology và tham số PPO; V15 cố định embedding=32 theo notebook tác giả, bất kể trường `hidden` kiến trúc cũ trong blueprint; không lấy trọng số V14. `--dry-run` chỉ kiểm tra cấu hình/tệp, **không thay thế kiểm thử AnyLogic thật**.

### Bước C — Khởi tạo EP0 và train thử EP1 (Routing)

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v15_tobias_staged.py --phase routing8 --init-only
.\.venv311\Scripts\python.exe -u .\scripts\train_v15_tobias_staged.py --phase routing8 --max-episodes 1 --resume
```

Nếu EP1 thành công, chọn điểm dừng tạm thời (ví dụ 20 EP; **đây là số thử nghiệm, không phải ngưỡng từ bài Tobias**):

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v15_tobias_staged.py --phase routing8 --max-episodes 20 --resume
```

### Bước D — Chuyển sang Dispatch-only, vẫn 8 AGV

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v15_tobias_staged.py --phase dispatch8 --max-episodes 1
```

Sau đó tăng `--max-episodes` và thêm `--resume` để tiếp tục phase `dispatch8`. Các phase phía sau tương tự, kế thừa nguyên actor, critic, optimizer và ValueNorm từ checkpoint đã commit của phase trước.

### Bước E — Joint training, tăng số AGV

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v15_tobias_staged.py --phase joint8 --max-episodes 1
.\.venv311\Scripts\python.exe -u .\scripts\train_v15_tobias_staged.py --phase joint12 --max-episodes 1
.\.venv311\Scripts\python.exe -u .\scripts\train_v15_tobias_staged.py --phase joint16 --max-episodes 1
.\.venv311\Scripts\python.exe -u .\scripts\train_v15_tobias_staged.py --phase joint20 --max-episodes 1
```

**Lưu ý:** các lệnh 1 EP bên trên chỉ để **smoke test**; không phải lịch huấn luyện nên dùng để công bố kết quả. Phải train từng giai đoạn đủ lâu, đánh giá rồi mới chuyển sang phase tiếp theo. Ở mỗi phase, tăng `--max-episodes N --resume`.

## 5. Resume và checkpoint

- Mỗi phase có thư mục riêng: `mappo_runs/V15_TOBIAS_GATV2_LINFE_MAPPO_NATIVE_ROUTING_3600S/<phase>/`.
- Lưu `initial_ep_0000.pt`, `rolling_slot_0.pt`, `rolling_slot_1.pt`, `state.json`, `experiment.json`, `metrics.jsonl`, và snapshots mỗi 10 EP.
- **Chỉ ghi nhận episode đã commit** trong `state.json`. `--resume` kiểm tra SHA-256 checkpoint, số episode metrics và hash của bộ code V15/MAPPO gốc trước khi nạp.
- Khi `Ctrl+C`, runner dừng an toàn tại mốc commit trước đó; nên không thay code/blueprint/mô hình trong khi một run đang tiến hành.
- Không train chung cùng port với một AnyLogic session khác. Nếu đổi `--port`, không làm thay đổi mô hình hoặc checkpoint.

## 6. Đánh giá 10 seed, 3.600s

Khi hoàn thành phase `joint20`, chạy:

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\evaluate_v15_tobias_staged.py --phase joint20 --dry-run
.\.venv311\Scripts\python.exe -u .\scripts\evaluate_v15_tobias_staged.py --phase joint20 --seeds 10 --first-seed 900001 --port 51881
```

Dừng giữa chừng thì thêm `--resume`. File JSONL có số liệu từng seed và không làm cập nhật PPO. Đọc tổng hợp:

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\summarize_v15_heldout.py '.\mappo_runs\V15_TOBIAS_GATV2_LINFE_MAPPO_NATIVE_ROUTING_3600S\joint20\heldout_joint20_10seeds_3600s.jsonl'
```

Muốn so sánh khoa học với V14 (EP130 Stop/Swing) và Tobias (Figure 8), phải xác nhận các bên **cùng điều kiện sản xuất, logic Block/LiveLock, 20 AGV, 3600s và giao thức greedy/seed**. Tobias và MAPPO khác framework/thuật toán nên **không thể khẳng định hơn kém chỉ qua một episode**.

## 7. Tài liệu đối chiếu

- Tobias repo: https://github.com/tobihoernchen/thesis
- Source tác giả: `thesis/policies/ma_gnn_routing.py` (`GNNFeatureExtractor`, `GNNRoutingNet`), `thesis/policies/ma_lin_net.py` (`LinFE`, `MALinPolicy`)
- Notebook tác giả: `exploration/MiniMatrix/training_disp_rout.ipynb`
- Bài báo: https://doi.org/10.1080/00207543.2024.2436126
- MAPPO source trong repo của ông: `thesis/mappo/algorithm.py`, `thesis/mappo/mappo_stability_v7.py`, `thesis/mappo/collector_stop_ab.py`

## 8. Phạm vi kiểm thử trước khi giao

Đã chạy 11/11 unit tests offline với PyTorch, kiểm tra output shape, gradient GAT, action mask, reward config và lọc role; đã chạy giả lập CLI `--dry-run` với blueprint mẫu và compile toàn bộ file Python. **Chưa thể chạy AnyLogic 3.600s trong môi trường phát triển này** và chưa kiểm tra sự tương thích với mô hình ZIP thực tế trên máy ông. Trước khi train dài, phải hoàn thành EP1 của `routing8`, kiểm tra `end_reason`, `selected_mask_violations`, role updates và số liệu JobDone/Block/LiveLock.

## 9. V15 sửa đúng Conventional Routing (bản này)

Đây là **V15 revised/V2**, không phải ZIP V15 cũ. Thư mục run mặc định đã đổi tên để tránh trộn checkpoint do trước đó `dispatch8` dùng bộ định tuyến hình học. Khi đổi từ bản cũ sang bản này, không dùng `--resume` vào run V15 cũ. **Phải bắt đầu lại một run V15-V2 từ EP0** nếu trước đó ông đã init/train bản cũ.

Quy trình kiểm tra trước khi train:

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p 'test_v15_tobias_staged_offline.py' -v
.\.venv311\Scripts\python.exe -u .\scripts\train_v15_tobias_staged.py --phase routing8 --dry-run
.\.venv311\Scripts\python.exe -u .\scripts\train_v15_tobias_staged.py --phase routing8 --init-only
.\.venv311\Scripts\python.exe -u .\scripts\train_v15_tobias_staged.py --phase routing8 --max-episodes 1 --resume
```

Sau khi Routing pretrain đã lưu ít nhất một episode, `dispatch8` sẽ kế thừa checkpoint và chỉ khởi tạo **native Conventional Routing** trong mô phỏng; có thể xem phần `experiment.json` với `dispatch_pretraining_router`.
