# TỔNG KẾT PHIÊN CHAT — TỪ V15 ĐẾN V16-B TRAFFIC-AWARE EDGE-GATv2

> **Dự án:** MAPPO–GNN–AnyLogic điều phối AGV trong hệ thống sản xuất dạng ma trận  
> **Thời điểm tổng kết:** 19/11/2026  
> **Repository:** https://github.com/trilehoangminh-source/MAPPO-GNN-AnyLogic  
> **Nhánh nền V15:** `refactor/v15-minimal`  
> **Nhánh phát triển V16-B được thấy trên VS Code:** `feature/v16b-traffic-edge-gatv2`  
> **Tình trạng cuối phiên:** V16-B **đã init EP0, chạy và cập nhật thành công Joint16 EP1 trên máy người dùng**; chưa có kết quả đánh giá Held-out của V16-B.

---

## 1. Bối cảnh: hệ thống và mục tiêu nghiên cứu

Người dùng đang phát triển hệ thống **Multi-Agent Proximal Policy Optimization (MAPPO)** kết nối với **AnyLogic**, kế thừa cách mô hình hóa và các thành phần gốc của Tobias. Hệ thống có ba vai trò học:

| Vai trò | Mạng/thiết kế trong V15 | Chức năng |
|---|---|---|
| **Routing** | GATv2, 2 lớp, 4 attention heads, embedding 32 | Chọn `Stop / Up / Down / Left / Right` cho AGV |
| **Dispatch Front** | LinFE/MLP | Chọn trạm hoặc đích phục vụ một phía luồng công việc |
| **Dispatch Rear** | LinFE/MLP | Chọn trạm hoặc đích phục vụ phía còn lại |
| **Critic cho từng vai trò** | Centralized critic (CTDE) | Ước lượng giá trị trạng thái phục vụ PPO |

- Mô phỏng mỗi episode: **3.600 giây**.
- Curriculum V15 đã bàn: `routing8 → dispatch8 → joint8 → joint12 → joint16 → joint20`.
- `dispatch8` sử dụng bộ định tuyến truyền thống `CollisionFreeRouting`/`CollisionFreeRoutingHive` có cơ chế đặt chỗ theo khoảng thời gian; các phase `joint*` học Routing bằng GATv2.
- Hình thức huấn luyện V15 đang sử dụng: **Full-Rollout Gradient Accumulation, microbatch 512 (GA512)**.
- **Quyết định quan trọng do người dùng xác nhận:** Routing V15 **cố tình KHÔNG dùng action mask** để bám theo phương án thí nghiệm gốc của Tobias. Vì vậy, **không được tự ý bật action mask trong V16-B**, cũng không gọi việc bỏ mask là lỗi.

### Công việc AnyLogic ngoài lề đầu phiên

Người dùng từng hỏi cách **xoay camera trong cửa sổ 3D của AnyLogic**. Trợ lý đã hướng dẫn thao tác điều hướng góc nhìn và đề cập cấu hình `3D Window`/`Navigation type`. Đây là câu hỏi thao tác phần mềm, tách biệt khỏi cải tiến thuật toán Routing.

---

## 2. Vấn đề của V15 Joint16: huấn luyện dài hơn nhưng hiệu suất không tốt lên

Người dùng gửi hai nhóm đồ thị:

1. **Training Curves:** `Targets Reached`, `Number of Blocks`, `Episode Reward`, `Products Completed (JobDone)`.
2. **Training Diagnostics:** `LiveLock`, `WrongTarget`, `Invalid Actions Taken`, PPO Entropy, PPO Approx KL, PPO Value Loss.

Những đồ thị này cho thấy ở giai đoạn muộn của Joint16:

- **JobDone trung bình trượt có xu hướng giảm**, khoảng ~14–15 ở một số đoạn giữa quá trình xuống ~10–12 gần 3 triệu cumulative events.
- **Reward và Targets Reached** cũng suy giảm ở các đoạn cuối.
- **Invalid Actions Taken** có xu hướng tăng, trong khi Routing Entropy tương đối cao và tăng ở giai đoạn cuối.
- **Blocks** đôi khi giảm nhưng điều đó **không nhất thiết là tín hiệu tốt**: nếu xe vận chuyển ít hơn thì số lần bị chặn cũng có thể giảm.
- **Approx KL nhỏ** và Value Loss không cho thấy hiện tượng nổ gradient rõ ràng, nhưng **PPO ổn định về mặt số học không đồng nghĩa policy tốt hơn**.

Các giá trị đọc trực tiếp từ hình là **ước lượng**, không phải thống kê Held-out trên checkpoint cố định.

### 2.1. Phân tích metrics Joint16

Trong phiên chat, log `metrics(20261116-153419).jsonl` được dùng để rà **532 episode**, tổng gần **2.994.938 cumulative events**. Trợ lý đưa ra ba checkpoint ứng viên/đối chứng để kiểm thử: **EP190, EP200 và EP530**.

Một số trung bình **cửa sổ 50 episode huấn luyện** đã nêu trong chat:

| Thước đo | EP190 | EP200 | EP530 |
|---|---:|---:|---:|
| JobDone trung bình | 14,24 | 14,30 | 11,04 |
| Reward trung bình | +1.883 | +1.632 | −303 |
| Invalid Actions trung bình | 889 | 962 | 1.740 |
| Blocks trung bình | 39,3 | 40,2 | 28,1 |
| Tỷ lệ EP có JobDone ≤ 10 | 14% | 14% | 52% |

**Diễn giải đúng:** EP190/EP200 là **ứng viên từ training log**, chứ **không tự động là checkpoint tốt nhất khi evaluation**. Không chọn theo một EP đạt đỉnh JobDone 20–22; cần so sánh **cùng seed, cùng thời gian 3.600 giây**, ưu tiên JobDone trung bình, phương sai và tỷ lệ episode thất bại.

### 2.2. Held-out evaluation đã được bàn

Người dùng nhắc đúng rằng code dự án **đã có cơ chế Held-out** nên **không cần viết script đánh giá mới cho V15**. Các file V15 có sẵn:

- `scripts/evaluate_v15_tobias_staged.py`
- `scripts/summarize_v15_heldout.py`

Lưu ý kỹ thuật: evaluator V15 đọc checkpoint do `joint16/state.json` chỉ tới — mặc định thường là checkpoint mới nhất — chứ không tự chọn EP190. Khi muốn kiểm tra EP190/EP200/EP530 phải bảo đảm kết quả thực sự gắn với đúng checkpoint (ví dụ bản sao metadata ở một output root riêng). Không chỉnh `state.json` của run chính tùy tiện.

Người dùng thông báo **kết quả evaluate V15 rất kém**, dẫn đến quyết định nghiên cứu cải tiến Routing. **Không có bảng kết quả Held-out cụ thể được đưa vào đoạn chat này**, nên không khẳng định các con số Held-out ngoài những thống kê training nêu ở trên.

---

## 3. Hai hướng nghiên cứu đã cân nhắc để cải thiện Routing

### Hướng A — Bổ sung thông tin giao thông vào các cạnh của GATv2

Ý tưởng: Routing V15 đã có node features mô tả vị trí/AGV/trạm/mục tiêu, nhưng attention chưa có **đặc trưng cạnh động** phản ánh tình trạng giao thông. Nâng cấp sang **Traffic-Aware Edge-GATv2** để mạng học lựa chọn hướng trong bối cảnh giao thông đông AGV.

Ưu điểm:

- Giữ được **MAPPO–GNN–AnyLogic** hiện tại.
- Giữ nguyên Routing có **5 hành động** và không mask.
- Có thể chạy thí nghiệm so sánh V15/V16 một cách kiểm soát.
- Dễ tách ảnh hưởng của edge features hơn so với thay toàn bộ Routing bằng bộ lập kế hoạch mới.

Hạn chế: đây vẫn là **policy học được**, **không bảo đảm** tránh va chạm/deadlock/livelock; hiệu quả phải đo bằng Held-out.

### Hướng B — MARL kết hợp MAPF (Multi-Agent Path Finding)

Ý tưởng: để MARL/MAPPO học **mức ưu tiên, trọng số, hoặc quyết định phối hợp cấp cao**, còn MAPF planner giải bài toán tìm đường có xét xung đột giữa nhiều AGV. Các khả năng từng bàn: học từ MAPF teacher (imitation), thêm safety shield, hoặc cho MAPPO điều khiển MAPF planner.

Ưu điểm: đưa các ràng buộc không xung đột vào quá trình lập đường thay vì để policy tự học tất cả từ reward.

Hạn chế: thay đổi lớn action space, thời điểm ra quyết định, pipeline thu thập trajectory và cơ chế log-probability/PPO; MAPF tránh collision cũng **không đồng nghĩa** tự giải hết deadlock/livelock.

### Một số tài liệu đã được nhắc tới khi thảo luận

> Các tài liệu sau là **danh sách tham khảo xuất hiện trong phiên chat**; chưa phải danh mục đã tái thẩm định toàn bộ nội dung, dữ liệu và khả năng tái lập.

- **PRIMAL: Pathfinding via Reinforcement and Imitation Multi-Agent Learning** — RL + imitation learning cho MAPF: https://arxiv.org/abs/1809.03531
- **PRIMAL2** — Lifelong MAPF trong môi trường đông tác tử: https://arxiv.org/abs/2010.08184
- **Distributed Heuristic Multi-Agent Path Finding with Communication (DHC)** — heuristic/communication/Deep RL; tra cứu chính xác theo tên bài báo.
- **Learning-guided Prioritized Planning for Lifelong Multi-Agent Path Finding in Warehouse Automation** — ý tưởng học độ ưu tiên và cho MAPF planner lập đường: https://arxiv.org/abs/2603.23838
- **Decentralized Graph Attention Multi-Agent Reinforcement Learning for Adaptive Urban Traffic Routing** — nguồn tham khảo về đồ thị có thông tin giao thông: https://www.nature.com/articles/s41598-026-56204-2
- **Graph Transformer with Reinforcement Learning for Vehicle Routing Problem** — nguồn tham khảo về attention có thông tin cạnh: https://doi.org/10.1002/tee.23771

**Về Tobias:** phiên chat từng thảo luận hướng cải thiện biểu diễn đồ thị/GAT và điều tiết giao thông như nghiên cứu tiếp theo. Cần phân biệt **đề xuất của tác giả** với **thiết kế cụ thể `[L,F,O,D]` do nhóm mình đề xuất**; không nên ghi rằng tác giả Tobias đã đích danh chọn bốn edge features ấy.

### Quyết định cuối cùng

**Làm V16-B Traffic-Aware Edge-GATv2 trước; giữ MARL + MAPF là hướng nghiên cứu mở rộng nếu V16-B chưa hiệu quả.**

---

## 4. Tìm hiểu source GitHub và xác minh observation phục vụ V16-B

Repository được kiểm tra trong phiên:

- https://github.com/trilehoangminh-source/MAPPO-GNN-AnyLogic/tree/refactor/v15-minimal
- Các module liên quan: `thesis/envs/behaviors_v121.py`, `thesis/mappo/v15_tobias_arch.py`, `thesis/mappo/gnn_routing_B3_v121.py`, `scripts/v15_tobias_build.py`, `scripts/v15_staged_collector.py`, `thesis/mappo/collector_stop_ab.py`.

**Kết quả đọc code quan trọng:** trong `TrainingBehavior._state_dict()` của `behaviors_v121.py`, các trường AGV có ý nghĩa:

| Vị trí dữ liệu | Ý nghĩa |
|---|---|
| `agvs[..., 0]` | `in_system` |
| `agvs[..., 1]` | `moving` |
| `agvs[..., 2:4]` | `last_node` |
| `agvs[..., 4:6]` | `next_node` |
| `agvs[..., 6:8]` | `target_node` |
| `agvs[..., 8:16]` | `nodes_in_reach` |
| `agvs[..., 16:18]` | `distance_next`, `distance_last` |

`HiveContext.nodes` và `HiveContext.paths` cung cấp hình học và đồ thị có hướng. `GraphSpec.from_hive()` dùng những dữ liệu này để tạo đồ thị cho Routing Actor.

**Kết luận thực thi:** có thể suy ra **AGV đang được mô tả là di chuyển trên cạnh từ `last_node` đến `next_node`** từ observation, đủ để xây dựng proxy giao thông động cho GATv2 **mà không phải sửa Java AnyLogic** ở phiên bản đầu. Nhưng chưa có các số đo vật lý chính xác như chiều dài hàng đợi thực, thời gian chờ tích lũy, hoặc reservation time-window cho Routing GAT.

---

## 5. Đặc trưng cạnh chính thức chốt cho V16-B

Mỗi cạnh có hướng từ `src` đến `dst` được biểu diễn bởi vector bốn chiều:

\[
\mathbf e_{src\rightarrow dst}(t)
= [L_{src,dst}, F_{src,dst}(t), O_{src,dst}(t), D_{dst}(t)]
\]

| Thứ tự | Thuộc tính | Cách tính trong V16-B | Chuẩn hóa |
|---|---|---|---|
| 0 | **Length** `L` | Khoảng cách Euclidean giữa hai node theo tọa độ GraphSpec; là khoảng cách hình học chuẩn hóa, không phải thời gian chạy vật lý | Chia độ dài cạnh lớn nhất |
| 1 | **Same-direction flow** `F` | Số AGV **khác ego** đang hoạt động và `moving=1`, với `last_node=src`, `next_node=dst` | Chia 20 |
| 2 | **Opposite-direction flow** `O` | Số AGV **khác ego** đang hoạt động và `moving=1`, với `last_node=dst`, `next_node=src` | Chia 20 |
| 3 | **Downstream pressure** `D` | Số AGV **khác ego** đang hoạt động có `next_node=dst` | Chia 20 |

Các quy ước:

- Bỏ **ego** khỏi bộ đếm xe khác để tránh tự xem xe mình là xe gây ùn tắc.
- Bỏ qua các row padding/AGV ngoài hệ thống.
- Cạnh **self-loop** dùng cho message passing nhận vector cạnh 0.
- `D` là **proxy áp lực node đích**, **không phải queue length đo vật lý**.
- `O` là số xe **quan sát thấy** đang đi chiều ngược; không tự suy diễn xe có thể đi trên cạnh ngược nếu topo không hỗ trợ.
- Đặc trưng được tính lại từ **actor-visible observation tại thời điểm ra quyết định**, không dùng dữ liệu tương lai, không lấy global state của Critic để cấp thêm cho Actor.

### Công thức Edge-GATv2

Thay vì chỉ dựa trên hai node, điểm attention bao gồm thông tin cạnh:

\[
s_{ij}^{(k)}(t)
= \mathbf a_k^{\mathsf T}\operatorname{LeakyReLU}
\left(W_{s,k}h_i + W_{t,k}h_j + W_{e,k}\mathbf e_{ij}(t)\right),
\qquad
\alpha_{ij}^{(k)} = \operatorname{softmax}_{j}(s_{ij}^{(k)})
\]

Những điểm giữ nguyên để làm đối chứng với V15:

- **2 lớp GATv2, 4 heads, embedding 32**.
- **Node features 22 chiều** và head ra **5 hành động**.
- **`uses_action_mask = False`** (quyết định có chủ đích của người dùng).
- Hai Dispatch Actor LinFE/MLP và ba Centralized Critic.
- Reward, cấu hình AnyLogic và logic PPO/GA512; **không thêm MAPF, Dijkstra hay reward shaping mới**.

### Warm-start từ V15

- Chuyển những tham số Actor/Critic tương thích từ checkpoint V15.
- **`W_e`/`edge_lin.weight` khởi tạo 0**: trước khi được học, phần cạnh bổ sung chưa tác động tới attention; hữu ích khi chuyển từ V15 sang V16-B.
- Giữ các trạng thái optimizer/ValueNorm tương thích của Dispatch và Critic.
- **Reset Adam của Routing Actor** vì mạng Routing có thêm tham số cạnh mới.
- Đây là **khởi tạo từ pretrained weights**, **không phải resume hoàn toàn** trạng thái tối ưu V15.

---

## 6. Gói V16-B đã được bàn giao

Tên file ZIP đã bàn giao trong chat: `V16B_Traffic_Aware_Edge_GATv2_Overlay.zip`.

Gói dạng **overlay**, giải nén nội dung vào thư mục gốc dự án Windows:

```text
C:\MARL_AnyLogic\MAPPO_NEW\thesis\
```

Trong source hiện tại, việc có **`thesis\thesis\mappo\`** là đúng vì thư mục ngoài cùng là project root còn thư mục trong là Python package `thesis`.

7 file mới trong ZIP:

```text
README_V16B_TRAFFIC_EDGE_GATV2_VI.md
RELEASE_V16B_TEST_REPORT.txt
scripts/evaluate_v16b_traffic.py
scripts/train_v16b_fullrollout.py
scripts/v16b.ps1
tests/test_v16b_traffic_edge.py
thesis/mappo/v16b_traffic_edge.py
```

**Nguyên tắc:** overlay V16-B không ghi đè file nguồn V15; kết quả V16-B được lưu riêng ở:

```text
mappo_runs\V16B_TRAFFIC_EDGE_GATV2_GA512\joint16\
```

V16-B **vẫn tái sử dụng** một số module có tên `v15_*`, ví dụ `train_v15_tobias_staged.py`, `train_v15_fullrollout.py`, `v15_tobias_build.py`, `v15_tobias_arch.py`, `v15_staged_collector.py` và `v15_fullrollout_accum.py`. **Không xóa những file này.**

---

## 7. Sao lưu GitHub và dọn các file chẩn đoán cũ

Người dùng xác nhận đã **push/backup V15 lên GitHub trước khi triển khai V16-B** và đang làm việc trên nhánh phát triển `feature/v16b-traffic-edge-gatv2`.

Để làm gọn `scripts/`, trợ lý đã xác định bốn file chẩn đoán/kiểm thử cũ không thuộc chuỗi chạy chính của V16-B:

```text
scripts/diagnose_v15_ep100_dispatch.py
scripts/evaluate_v15_ep100_buffer.py
scripts/smoke_v15_fullrollout_offline.py
scripts/test_v15_routing_no_mask.py
```

Người dùng đã chạy `git rm` và `git commit -m "Remove obsolete V15 diagnostic scripts for V16B"`; ảnh terminal xác nhận commit:

```text
[feature/v16b-traffic-edge-gatv2 a87caa8] Remove obsolete V15 diagnostic scripts for V16B
4 files changed, 1197 deletions(-)
```

**Lưu ý:** thao tác này là dọn file ở nhánh V16-B. Chưa có bằng chứng trong chat rằng toàn bộ thay đổi V16-B đã được `git add`, `git commit` và `git push` sau đó. Các file có dấu `U`/`??` trong VS Code cần kiểm tra riêng bằng `git status` trước khi push.

Các script vẽ đồ thị V15 vẫn nên giữ vì có thể dùng so sánh V15/V16.

---

## 8. Quá trình khởi tạo và kiểm thử V16-B trên máy Windows

### 8.1. Kiểm thử Python

Ban đầu người dùng gặp:

```text
No module named pytest
```

Nguyên nhân: `.venv311` chưa cài `pytest`. Sau khi cài:

```powershell
.\.venv311\Scripts\python.exe -m pip install pytest
.\.venv311\Scripts\python.exe -m pytest -q tests\test_v16b_traffic_edge.py
```

Ảnh terminal của người dùng xác nhận:

```text
6 passed in 9.46s
```

**Ý nghĩa:** unit test qua, **không đồng nghĩa** V16-B đã được chứng minh tốt hơn V15 về JobDone.

### 8.2. Lỗi PowerShell execution policy

Khi chạy `scripts/v16b.ps1`, Windows báo file chưa được ký số/chính sách bảo mật chặn thực thi (`PSSecurityException`, `UnauthorizedAccess`). Đã hướng dẫn cho phép **chỉ trong phiên PowerShell hiện tại**:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Sau đó người dùng chạy được các thao tác `dryrun`, `init` và `train`.

### 8.3. Dry-run và khởi tạo EP0

Các lệnh:

```powershell
.\scripts\v16b.ps1 -Action dryrun
.\scripts\v16b.ps1 -Action init
```

Ảnh terminal xác nhận **nguồn warm-start thực tế**:

```text
mappo_runs\V15_FULLROLLOUT_GA512\joint12\snapshot_ep_0850.pt
```

**Phải nhớ:** V16-B **khởi tạo từ V15 Joint12 EP850** để học tiếp tại **Joint16 EP0 (16 AGV)**, **KHÔNG phải Joint16 EP850**.

Log khởi tạo có thông điệp tương đương:

```text
[V16B WARMSTART] V15 actors/critics transferred;
Routing actor Adam reset (new edge params);
Dispatch and critics Adam/ValueNorm preserved.
[V15 PHASE INITIALIZED] joint16 EP0 ...
[V15 INIT ONLY] No training conducted
```

Dòng log có chữ `[V15 ...]` là do V16-B **tái sử dụng runner/logger V15**, **không phải** đang ghi đè dữ liệu V15.

### 8.4. Joint16 EP1 — đã chạy tích hợp thành công

Người dùng chạy:

```powershell
.\scripts\v16b.ps1 -Action train -MaxEpisodes 1
```

Ảnh terminal gửi tại cuối phiên xác nhận:

| Chỉ số EP1 | Giá trị |
|---|---:|
| Fleet | **16 AGV** |
| JobDone | **18** |
| Blocks | **55** |
| LiveLock | **188** |
| Delivered Reward | **+2.708,57** |
| Routing samples | **5.327** |
| Dispatch Front samples | **223** |
| Dispatch Rear samples | **208** |
| Thời gian chạy thực | **177,1 giây** |

Terminal báo ba vai trò đã cập nhật:

```text
Updated=['dispatch_front', 'dispatch_rear', 'routing']
[V15 PHASE DONE] joint16 committed EP 1
```

**Kết luận đúng ở mốc EP1:** V16-B đã đi qua một lượt **AnyLogic → thu thập observation → Edge-GATv2/MAPPO → update → lưu checkpoint** mà không xuất hiện lỗi rõ ràng trong log. **JobDone = 18 của duy nhất EP1 không đủ kết luận V16-B tốt hơn V15**; tại thời điểm bắt đầu, `edge_lin` bằng 0 nên edge features còn cần quá trình học.

---

## 9. Các lệnh cần dùng tiếp — từ tình trạng cuối phiên

Mở PowerShell tại:

```text
C:\MARL_AnyLogic\MAPPO_NEW\thesis
```

### 9.1. Train tiếp tới tổng EP50 (không phải chạy thêm 50 EP)

```powershell
.\scripts\v16b.ps1 -Action train -MaxEpisodes 50
```

Vì EP1 đã xong, runner tiếp tục từ **EP2 đến EP50**. Có thể tăng tiếp:

```powershell
.\scripts\v16b.ps1 -Action train -MaxEpisodes 100
```

`-MaxEpisodes` là **tổng số episode mục tiêu của phase**, không phải số episode cộng thêm. **Không chạy `-Action init` lại** sau khi đã có EP0/EP1.

### 9.2. Evaluation Held-out V16-B bằng chức năng có sẵn

```powershell
.\scripts\v16b.ps1 -Action eval -Seeds 10 -Port 51921
```

Tổng hợp kết quả:

```powershell
.\.venv311\Scripts\python.exe .\scripts\summarize_v15_heldout.py `
  .\mappo_runs\V16B_TRAFFIC_EDGE_GATV2_GA512\joint16\heldout_joint16_10seeds_3600s.jsonl
```

- Held-out **greedy**, 10 seed, 3.600 giây/episode, không cập nhật PPO.
- Evaluator gắn với **checkpoint mới nhất từ `state.json`** và không trộn checkpoint khác nhau trong cùng file kết quả; phải giữ quy tắc này khi so sánh EP50, EP100.
- So sánh V15/V16-B **trên cùng seed, cùng fleet 16 và cùng mô hình môi trường**.

### 9.3. Theo dõi các chỉ số

Tối thiểu cần quan sát:

- `JobDone` trung bình và tỷ lệ episode JobDone thấp.
- `Episode Reward`, `Targets Reached`, `WrongTarget`.
- `Invalid Actions`, `Blocks`, `LiveLock`.
- Routing Entropy, Approx KL, Value Loss.
- Giá trị/độ ảnh hưởng của trọng số edge attention sau nhiều update, tránh kết luận rằng mạng đã sử dụng traffic features chỉ vì code đã cấp `edge_attr`.

---

## 10. Các quyết định đã chốt, các điểm còn mở

| Vấn đề | Trạng thái |
|---|---|
| Giữ Routing không action mask như Tobias | **Đã chốt** |
| Chưa chuyển sang MARL+MAPF, thử Edge-GATv2 trước | **Đã chốt** |
| Edge features V16-B = `[L, F, O, D]` | **Đã triển khai** |
| Không sửa Java AnyLogic ở bản đầu | **Đã triển khai** |
| Duy trì 3 role MAPPO, PPO GA512, reward V15 | **Đã triển khai** |
| Khởi tạo từ V15 Joint12 EP850 | **Đã xác nhận từ log** |
| Offline unit tests | **6/6 pass** |
| Windows/AnyLogic tích hợp EP1 | **Đã chạy thành công** |
| Huấn luyện EP50/EP100 | **Chưa có kết quả trong chat** |
| Held-out V16-B so với V15 | **Chưa có kết quả trong chat** |
| Kết luận V16-B cải thiện JobDone ổn định | **Chưa thể kết luận** |
| MARL + MAPF nếu V16-B chưa đủ hiệu quả | **Hướng nghiên cứu dự phòng** |

### Một số rủi ro cần ghi nhớ

1. **Traffic proxy không phải cảm biến vật lý:** `D` là số xe có `next_node=dst`, không phải queue length đo trực tiếp; `F/O` phụ thuộc chất lượng tín hiệu `moving/last_node/next_node` của observation.
2. **Không mask có chủ đích:** V16-B vẫn có thể chọn hướng không hợp lệ; cần đo Invalid Actions, không tự coi chúng sẽ biến mất.
3. **Khởi tạo cạnh bằng 0:** EP1 chưa chứng minh mạng đã tận dụng traffic-aware attention.
4. **So sánh checkpoint công bằng:** dùng Held-out cùng seed, tránh chọn checkpoint chỉ theo những episode train may mắn.
5. **Giữ V15 làm baseline:** không xóa `mappo_runs\V15_FULLROLLOUT_GA512` vì có checkpoint/metadata cần để tái lập.
6. **Khi tiếp tục nghiên cứu:** có thể thử ablation chỉ `Length` (V16-A) và đầy đủ `L,F,O,D` (V16-B), sau đó mới xem xét thêm lịch sử `Waiting Time` hoặc chuyển sang MAPF.

---

## 11. Các đường dẫn cần nhớ

**GitHub:**

- Repository: https://github.com/trilehoangminh-source/MAPPO-GNN-AnyLogic
- Nhánh nền đã đối chiếu: https://github.com/trilehoangminh-source/MAPPO-GNN-AnyLogic/tree/refactor/v15-minimal

**Source trên Windows:**

```text
C:\MARL_AnyLogic\MAPPO_NEW\thesis
  ├─ scripts\v16b.ps1
  ├─ scripts\train_v16b_fullrollout.py
  ├─ scripts\evaluate_v16b_traffic.py
  ├─ tests\test_v16b_traffic_edge.py
  ├─ thesis\mappo\v16b_traffic_edge.py
  └─ mappo_runs\
      ├─ V15_FULLROLLOUT_GA512\
      │   └─ joint12\snapshot_ep_0850.pt
      └─ V16B_TRAFFIC_EDGE_GATV2_GA512\
          └─ joint16\
              ├─ experiment.json
              ├─ state.json
              ├─ metrics.jsonl
              └─ ...checkpoint...
```

**Các file ngoài repository được dùng trong phiên:** `thesis.zip` (source Tobias), log `metrics(20261116-153419).jsonl`, các hình Training Curves/Diagnostics và ảnh terminal VS Code.

---

## 12. Kết luận ngắn gọn để tiếp tục ở phiên chat kế tiếp

> **Hiện trạng:** V15 Joint16 huấn luyện dài nhưng JobDone/Reward không cải thiện ổn định; checkpoint EP190/EP200 từng được cân nhắc nhưng kết quả Held-out V15 bị người dùng nhận xét là kém. Nhóm quyết định thử **V16-B Traffic-Aware Edge-GATv2 + MAPPO**, giữ Routing **không action mask** như V15/Tobias. V16-B bổ sung 4 edge features từ actor observation `[length, same-direction flow, opposite-direction flow, downstream pressure]`, không sửa AnyLogic Java, không MAPF và không đổi reward/PPO. Đã push backup V15, giải nén overlay V16-B, test **6 passed**, dryrun/init thành công từ **V15 Joint12 EP850**, rồi train **Joint16 EP1** thành công (**JobDone 18, Reward +2708.57, Blocks 55, LiveLock 188; cả 3 role updated**). **Bước tiếp theo:** tiếp tục EP2→EP50 bằng `.\scripts\v16b.ps1 -Action train -MaxEpisodes 50`, phân tích training và Held-out cùng seed so V15, kiểm tra edge weights có học và có giúp Routing thật hay không.

