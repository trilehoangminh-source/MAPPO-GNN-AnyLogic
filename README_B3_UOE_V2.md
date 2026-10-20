# B3-GNN-UOE V2 — Routing GNN / GAT + Diagnostics + MAPPO

Ngày: 20/10/2026. Bản **bổ sung** cho project `MAPPO-GNN-AnyLogic`, dựa trên B3-GNN v1 đã smoke thành công trên máy ông. **Không thay file cũ**, không thay mô hình AnyLogic, không thay optimizer/critic của Dispatching.

## 1. Những gì lấy cảm hứng từ nguồn

- **Bài báo Steinbacher, Wegmann & Freitag, 2025, §3.2.4, Figure 4, §4.1.2:** Routing dùng Graph Attention Network, Dispatching dùng linear/attention (không GNN); huấn luyện tách nhiệm vụ trước khi phối hợp. Bản V2 triển khai directed GAT *tùy chọn* nhưng vẫn là MAPPO (không khẳng định tái hiện GAT/DQN nguyên tác).
- **EPyMARL, github.com/uoe-agents/epymarl:** log các metric rõ ràng; tách train vs held-out evaluation, dùng seeds độc lập, kiểm tra policy trên cùng môi trường; thêm pre-update log-prob parity audit (triển khai riêng, không sao chép code EPyMARL).
- **TA-RWARE, github.com/uoe-agents/task-assignment-robotic-warehouse:** phân biệt task/target vs pathfinding; giám sát mapping và mục tiêu trước khi thay quy tắc tránh va chạm. Bản này CHƯA thay logic tránh va chạm hay task-dispatching của AnyLogic.

## 2. Tính tương thích và các file

Bản V2 cần project trên máy **đã có** `scripts/train_mappo_long_B3_GNN_v121.py`, `thesis/mappo/gnn_routing_B3_v121.py`, `thesis/mappo/routing_guidance_v121.py`, và source B2 Operation-first đầy đủ.

- `thesis/mappo/gnn_attention_uoe_v2.py`: `DirectedGraphAttention`, chuyển Routing Actor sang directed GAT tùy chọn. Luôn rebuild routing optimizer.
- `thesis/mappo/gnn_audit_uoe_v2.py`: đếm exact-node mapping với **đúng tolerance 1e-3** của B3 V1; ước lượng tọa độ lệch; kiểm tra action mask và log-prob ratio của PPO **trước update**.
- `scripts/train_B3_GNN_UOE_v2.py`: train bản V2 vào output riêng; chọn `--routing-encoder gat` hoặc `mp`, ghi thêm `gnn_graph_mapping`, `ppo_preupdate_audit` trong `metrics.jsonl`.
- `scripts/evaluate_B3_GNN_UOE_v2.py`: held-out eval chỉ đọc checkpoint V2, không train.
- `scripts/compare_marl_runs_uoe_v2.py`: hiển thị trung bình/SD của KPI từ log, ghi rõ train ≠ đánh giá.
- `tests/test_uoe_v2_offline.py`: test độc lập không cần Java.

**Quan trọng:** V2 GAT tạo checkpoint có state_dict khác V1 message-passing. Dùng evaluator **V2**, không dùng evaluator V1 hoặc cố load checkpoint V1 vào GAT.

## 3. Cài đặt

1. Đang đứng trong thư mục gốc: `C:\MARL_AnyLogic\MAPPO_NEW\thesis` (có `scripts/`, `tests/`, thư mục con `thesis/`).
2. Giải nén **nội dung** `B3_UOE_v2_patch.zip` trực tiếp vào thư mục gốc, giữ đúng cấu trúc. Không giải nén thành một folder con `B3_UOE_v2_patch` bên trong project.
3. Kích hoạt `.venv311` như đã dùng.
4. Kiểm tra:

```powershell
python tests/test_gnn_B3_v121.py
python tests/test_uoe_v2_offline.py
```

Nếu không PASS hai bước này, không train.

## 4. Smoke 2 episodes — B3-GAT (thí nghiệm mới)

```powershell
python -u scripts/train_B3_GNN_UOE_v2.py --model envs/Matrix_critic_fresh.zip --port 51695 --episodes 2 --sim-seconds 720 --seed 43 --routing-encoder gat --gat-heads 2 --hidden 32 --graph-layers 2 --output mappo_runs/B3_UOE_v2_GAT_smoke_seed43
```

Nếu muốn kiểm chứng chỉ phần diagnostics mà **không thay kiến trúc**, so sánh smoke với `--routing-encoder mp --output mappo_runs/B3_UOE_v2_MP_smoke_seed43 --port 51696`.

Lưu ý tên output mới mỗi lần. Script cố tình từ chối ghi đè.

**Những dòng cần chụp và gửi:**

```text
[B3-UOE-V2] Routing: {'encoder': 'directed_gat', ...} | Dispatching NON-GNN: ['dispatch_front','dispatch_rear']
[B3-UOE-V2 EP 1/2] ... ego_map=... target_map=... candidate_map=... PPO_logp_err=...
[B3-UOE-V2 EP 2/2] ...
```

Trong `mappo_runs/.../metrics.jsonl`, đọc thêm:

- `environment.gnn_graph_mapping.ego_map_rate`
- `environment.gnn_graph_mapping.target_map_rate`
- `environment.gnn_graph_mapping.candidate_map_rate`
- `environment.gnn_graph_mapping.nearest_node_distance`
- `environment.ppo_preupdate_audit.max_logprob_error`
- `environment.statistics.jobDone`, `wrongTarget`, `targetReached`, `block`, `liveLock`.

Mapping thấp **không tự động nghĩa GNN vô dụng**: có thể AGV đang ở trên cạnh, candidate node nằm khác tọa độ chuẩn, hoặc nguồn observation không chính xác. Chưa thay `1e-3` trước khi xác minh. Có thể tùy chọn `--min-candidate-map-rate 0.95` để dừng nếu rate thấp, **không bật mặc định**.

## 5. Held-out evaluation trên model 50-episode (CHỈ sau khi smoke đạt)

```powershell
python -u scripts/train_B3_GNN_UOE_v2.py --model envs/Matrix_critic_fresh.zip --port 51697 --episodes 50 --sim-seconds 720 --seed 43 --routing-encoder gat --output mappo_runs/B3_UOE_v2_GAT_seed43_50ep
python -u scripts/evaluate_B3_GNN_UOE_v2.py --checkpoint mappo_runs/B3_UOE_v2_GAT_seed43_50ep/checkpoint_ep_0050.pt --port 51698 --seeds 1001 1002 1003 --sim-seconds 3600
python scripts/compare_marl_runs_uoe_v2.py mappo_runs/B3_UOE_v2_GAT_seed43_50ep/evaluation_uoe_v2.jsonl
```

Không thể khẳng định hiệu quả hơn B2/B3 chỉ từ 2 episodes; cần cùng test seeds, thời lượng, fleet size và AnyLogic ZIP, đồng thời so sánh ablation `mp` vs `gat` vs `pooled`.

## 6. Những gì CHƯA giải quyết

- `observation.networkcontext=None` là warning từ Alpyne: V2 chỉ quan sát mapping dữ liệu đang được adapter đưa vào Routing, **chưa sửa schema Java**.
- `wrongTarget` có thể do Routing, Dispatching, dữ liệu đích, hoặc thiết kế reward. Không tự động xem GAT sẽ sửa `wrongTarget`.
- Không chuyển TA-RWARE A* hoặc collision checker sang AnyLogic; đó là thay mô hình động lực học và có thể làm sai benchmark gốc.
- Chưa chạy được AnyLogic Java trong môi trường tạo patch; **phải smoke trên máy Windows ông**, sau đó gửi lại log.

## 7. Nguồn

- Steinbacher et al., *Production control with Reinforcement Learning for a matrix-structured production system*, IJPR 2025, DOI 10.1080/00207543.2024.2436126.
- https://github.com/uoe-agents/epymarl
- https://github.com/uoe-agents/task-assignment-robotic-warehouse
- https://github.com/uoe-agents/robotic-warehouse

Mọi source V2 viết mới dựa trên ý tưởng học thuật, không copy module upstream; phải đọc và tuân thủ giấy phép nguồn nếu về sau trực tiếp nhập module từ các repository khác.
