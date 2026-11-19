# V16-B — Traffic-Aware Edge-GATv2 + MAPPO (đối chứng V15)

## A. BẢO VỆ V15 TRƯỚC KHI GIẢI NÉN

**Chưa giải nén gói V16-B nếu chưa `git push` thành công bản code V15 hiện tại.**

Mở PowerShell tại `C:\MARL_AnyLogic\MAPPO_NEW\thesis`:

```powershell
# Xác nhận đang ở git repo và đúng branch
 git status --short
 git branch --show-current
 git remote -v

# Nếu hiện tại đúng branch refactor/v15-minimal:
 git add -u -- scripts thesis/mappo thesis/envs
# Nếu `git status --short` liệt kê source V15 .py MỚI chưa tracked,
# thêm TỪNG FILE cần backup bằng: git add -- scripts/ten_file.py
# Nếu có source V15 ở ngoài ba thư mục này, thêm đúng file đó.
 git diff --cached --name-only
 git diff --cached --stat
# Kiểm tra kỹ KHÔNG stage checkpoint, AnyLogic .zip, dữ liệu riêng tư.
 git commit -m "Backup V15 before V16B traffic edge GATv2"
 git push origin refactor/v15-minimal
 git rev-parse HEAD
 git status --short
```

Nếu `git commit` nói "nothing to commit" thì vẫn kiểm tra `git status`/`git push`/đối chiếu HEAD trước khi tiếp tục. Nếu đang ở branch khác, **không chạy push mù**: chuyển sang nhánh chứa V15 và kiểm tra thay đổi cục bộ trước. Không upload `mappo_runs`, `.venv311`, AnyLogic ZIP, checkpoint `.pt` lên GitHub.

Sau khi V15 đã được push thành công, có thể tạo nhánh mới để phát triển V16-B (tùy chọn):

```powershell
 git switch -c feature/v16b-traffic-edge-gatv2
```

### B. Giải nén đúng chỗ

ZIP có các đường dẫn tương đối `scripts/`, `thesis/mappo/`, `tests/`, `README...`. **Giải nén nội dung ZIP vào thư mục dự án `C:\MARL_AnyLogic\MAPPO_NEW\thesis`**, tức là tạo `thesis\scripts\train_v16b_fullrollout.py` và `thesis\thesis\mappo\v16b_traffic_edge.py` (thư mục `thesis\thesis\mappo` là bình thường với cấu trúc source hiện tại).

ZIP **không ghi đè file V15** và **không bao gồm** `.git`, model AnyLogic, checkpoint, hoặc run logs.

### C. Test trước khi chạy AnyLogic

```powershell
.\.venv311\Scripts\python.exe -m pytest -q tests\test_v16b_traffic_edge.py
```

Các test gồm: các trọng số edge mới bằng 0 tạo logits như V15; hướng của edge và luồng ngược; padding; gradient edge attention; không có action mask; chuyển checkpoint V15 sang V16-B. Nếu máy chưa cài pytest, có thể cài `python -m pip install pytest`, không ảnh hưởng cấu hình mạng.

### D. Train Joint16 từ snapshot tốt V15

Lệnh chạy nhanh nhất, **tại thư mục gốc của dự án**:

```powershell
# Kiểm tra config/nguồn, không tạo file và không khởi động AnyLogic
.\scripts\v16b.ps1 -Action dryrun

# Mặc định khởi tạo Joint16 V16-B từ V15 Joint12 EP850
.\scripts\v16b.ps1 -Action init

# Train thử EP1 rồi tới EP500
.\scripts\v16b.ps1 -Action train -MaxEpisodes 1
.\scripts\v16b.ps1 -Action train -MaxEpisodes 500
```

Nếu PowerShell chặn `.ps1`, có thể chạy trong cùng phiên lệnh:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Muốn warm-start từ V15 Joint16 EP190 thay vì Joint12 EP850, **trước khi init**, dùng:

```powershell
.\scripts\v16b.ps1 -Action dryrun -WarmstartPhase joint16 -WarmstartEp 190
.\scripts\v16b.ps1 -Action init -WarmstartPhase joint16 -WarmstartEp 190
```

**Chỉ chọn một nguồn** cho mỗi run, không dùng cả hai. Chọn Joint12 EP850 nếu ưu tiên tiếp cận hệ thống 16 AGV từ chính sách 12 AGV đã ổn định tương đối; chọn Joint16 EP190 nếu muốn tiếp nối chính sách đã thích nghi phần nào với 16 AGV. Đây là giả thuyết thực nghiệm, không khẳng định checkpoint nào tốt hơn.

Dùng `-Port 51890` mặc định cho training; khi chạy held-out trong cùng máy cần dùng port khác (ví dụ 51921) và tránh quá tải GPU/AnyLogic.

**Chạy tiếp sau restart**: PowerShell mới, thư mục gốc dự án:

```powershell
.\scripts\v16b.ps1 -Action train -MaxEpisodes 500
```

`--max-episodes` là **tổng EP của phase**, không phải số EP cần chạy thêm. V16-B có run root RIÊNG:

`mappo_runs\V16B_TRAFFIC_EDGE_GATV2_GA512\joint16\`

Các file V15 và checkpoint không bị sửa. Runner sao chép optimizer, ValueNorm, critic. Riêng Routing actor Adam reset khi chuyển từ V15 vì có `edge_lin.weight` mới; không được quảng cáo là resume toàn phần optimizer.

### E. Đánh giá Held-out bằng code có sẵn

```powershell
# Giữ 10 seeds, greedy, 3600s, không cập nhật PPO
.\scripts\v16b.ps1 -Action eval -Seeds 10 -Port 51921

# Đọc kết quả bằng summarizer V15 có sẵn
.\.venv311\Scripts\python.exe .\scripts\summarize_v15_heldout.py `
  .\mappo_runs\V16B_TRAFFIC_EDGE_GATV2_GA512\joint16\heldout_joint16_10seeds_3600s.jsonl
```

Chú ý: evaluator V15/V16-B đọc **checkpoint mới nhất** từ `state.json` và từ chối trộn các checkpoint khác nhau trong cùng file heldout. Muốn đánh giá nhiều snapshot thì cần một bản sao metadata ở một thư mục riêng; không chỉnh `state.json` trong run chính. Đổi `-Seeds 10` thành 20 nếu muốn tập đánh giá rộng hơn; tên file đầu ra thay đổi theo số seeds.

### F. Thiết kế V16-B đã chốt

Giữ nguyên: GATv2 2 lớp x 4 heads, embedding 32; node feature 22 chiều; five Routing actions; Routing **không action mask**; LinFE Dispatch; 3 centralized critics; V15 reward; Full-Rollout GA512. V16-B chỉ thêm đúng 4 `edge_attr` vào attention, **không dùng MAPF**, không Dijkstra, không giả lập dữ liệu chờ/queue/reservation không quan sát được.

`e_{src->dst}` từ `obs['agvs']` tại mỗi quyết định:

| k | Thuộc tính | Định nghĩa | Chuẩn hóa |
|---|---|---|---|
| 0 | Độ dài cạnh | Euclidean từ GraphSpec tọa độ chuẩn hóa; không phải thời gian vật lý | chia max độ dài cạnh |
| 1 | Same-direction flow | số AGV khác có `in_system`=1, `moving`=1, `last_node=src`, `next_node=dst` | /20 |
| 2 | Opposite-direction flow | như trên nhưng `last_node=dst`, `next_node=src` | /20 |
| 3 | Downstream next-node pressure | số AGV khác đang `in_system`=1 và có `next_node=dst` | /20 |

**Quan trọng:** chỉ số (3) là *proxy áp lực đích kế tiếp*, KHÔNG phải độ dài hàng đợi vật lý. Chỉ số (2) vẫn có thể đếm xe ngược chiều ngay cả khi edge kia không có chiều ngược trong topo, bởi đang phản ánh chuyển động quan sát được; nếu AnyLogic đảm bảo không thể có chiều ngược thì giá trị sẽ bằng 0. Các cạnh self-loop chỉ làm message passing và được gán vector 0; dummy padded rows không tính lưu lượng. Ego row 0 không tính vào traffic counts để tránh tự coi mình là xe gây tắc.

Dữ liệu đúng từ Tobias `MultiAgent._state_dict()`:

- `agvs[...,0]` `in_system`
- `agvs[...,1]` `moving`
- `agvs[...,2:4]` `last_node`
- `agvs[...,4:6]` `next_node`
- `agvs[...,6:8]` `target_node`

Edge attributes được tính **trong actor forward** từ observation (không sử dụng global state/AnyLogic tương lai) → PPO rollout và optimizer recompute cùng một hàm. Thêm `W_e edge_attr` vào LeakyReLU attention, không thay đổi nội dung message values. Edge weight mới được khởi tạo bằng 0 để V16-B có thể bắt đầu với hành vi như V15 khi tải trọng số cũ.

### G. Cần kiểm tra khi chạy thật

1. Test offline thành công; dry-run báo `phase=joint16 fleet=16`, source checkpoint đúng.
2. Init EP0 in `...\V16B_...\joint16\state.json`; log có `[V16B WARMSTART]...`.
3. EP1 báo 3 role được cập nhật, không lỗi `logprob mismatch`.
4. Quyết định không mask ở Routing giữ nguyên (đường đi sai vẫn có thể xảy ra).
5. Held-out 10–30 seed phải **so sánh cùng seed** với V15 cùng fleet và môi trường; JobDone, Invalid, Blocks, LiveLock, tỷ lệ episode thất bại.

V16-B là **nguyên mẫu nghiên cứu có kiểm thử unit**, chưa thể khẳng định hiệu suất vì AnyLogic và file checkpoint thực tế cần chạy trên máy Windows của ông.
