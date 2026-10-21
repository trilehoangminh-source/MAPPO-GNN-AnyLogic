# B3-V4 — Edge-aware GAT + Dijkstra + MAPPO (Routing only)

## Mục tiêu

Đây là **bản thử nghiệm riêng**, phát triển từ code B3-GNN-V1 và B3-UOE-V2 đã cài ở thư mục `thesis` của bạn. Chỉ thay cấu trúc **Routing Actor**. **Dispatching B2 Operation-first** tiếp tục dùng actor pooled/MLP. PPO/CTDE Critic, AnyLogic, cơ chế `Action Mask` và `GuidedEventCollector` hiện tại được giữ nguyên.

**Không có cam kết rằng V4 sẽ giảm block hay tăng jobDone.** Phải đo bằng AnyLogic trên nhiều seed cùng 3.600 giây, so với B3-UOE-V2.

## Nghiên cứu GitHub tham khảo

- [Lei-Kun — DRL-and-graph-neural-network-for-routing-problems](https://github.com/Lei-Kun/DRL-and-graph-neural-network-for-routing-problems): paper của Lei et al., *Solve routing problems with a residual edge-graph attention neural network* (Neurocomputing 508, 2022). Ý tưởng dùng **edge features** trong attention, residual connections và PPO.
- [aalp75 — neural-execution-graph-algorithms](https://github.com/aalp75/neural-execution-graph-algorithms): tổ chức kiểm thử và học thuật toán đồ thị như Bellman–Ford. Đây là tham khảo tư duy kết hợp heuristic tìm đường với policy học, **chưa** chép implementation từ repo.
- [wouterkool — attention-learn-to-route](https://github.com/wouterkool/attention-learn-to-route): đánh giá bằng greedy rollout, validation riêng; đây là hệ TSP/VRP và REINFORCE, **không** phải API AnyLogic.

Tất cả code trong patch là implementation mới viết độc lập; không yêu cầu `torch-geometric`, không sao chép source upstream.

## Các thay đổi kỹ thuật

1. `ResidualEdgeGAT` tính attention theo **cạnh có hướng** (u→v), xử lý node sink bằng self-loop, thêm thông tin cạnh trong **attention score lẫn message**.
2. Mỗi cạnh có 3 feature: `(normalized_length, normalized_dx, normalized_dy)`. Đây là **đơn vị tọa độ graph đã chuẩn hóa**, không được hiểu trực tiếp là mét. Thông tin mật độ AGV theo node vẫn được lấy từ `GraphRoutingActor` V1 cũ; patch này **chưa thêm dự đoán va chạm trên cạnh**.
3. Kết nối residual + layer normalization; lớp `edge_score`/`edge_value` có gradient và được thêm vào Routing optimizer.
4. Dijkstra vẫn là **soft prior trên logits** nhưng dùng schedule giảm dần: mặc định `--routing-guide-strength 0.5` → `--guide-final 0.2` trong 10 episodes. Strength được lưu thành `routing_guide_bias` của mỗi observation, bảo toàn PPO rollout/update parity.
5. Không thay Dijkstra thành mạng neural, không sửa `reward_wrong_target`, không cho Stop là action hợp lệ nếu bản B2 đã mask Stop. Mọi thay đổi phải được đối chiếu thí nghiệm.

## Cài đặt

Giải nén **nội dung** file ZIP vào thư mục gốc repo:

`C:\MARL_AnyLogic\MAPPO_NEW\thesis`

Các file được bổ sung:

- `thesis/mappo/edge_gat_routing_v4.py`
- `scripts/train_B3_edgeGAT_MAPPO_v4.py`
- `scripts/evaluate_B3_edgeGAT_MAPPO_v4.py`
- `tests/test_edgeGAT_MAPPO_v4_offline.py`

**Không ghi đè các file MAPPO, B2 hoặc checkpoint cũ.** Cần cài sẵn patch B3-GNN V1 và B3-UOE-V2 vì V4 tái sử dụng các mô-đun `gnn_routing_B3_v121`, `routing_guidance_v121` và `gnn_audit_uoe_v2`.

## Bước 1 — Offline test (không chạy AnyLogic)

```powershell
python tests/test_edgeGAT_MAPPO_v4_offline.py
```

Kỳ vọng:

`PASS: directed edge features, gradient, residual attention, sinks, action mask, PPO pre-update parity`

## Bước 2 — AnyLogic smoke 2 episodes, 720 giây

```powershell
python -u scripts/train_B3_edgeGAT_MAPPO_v4.py `
  --model envs/Matrix_critic_fresh.zip `
  --port 51730 `
  --episodes 2 `
  --sim-seconds 720 `
  --seed 43 `
  --edge-heads 2 `
  --hidden 32 `
  --graph-layers 2 `
  --routing-guide-strength 0.5 `
  --guide-final 0.2 `
  --guide-decay-episodes 10 `
  --output mappo_runs/B3_EDGE_GAT_V4_smoke_seed43
```

Kỳ vọng mỗi episode có `candidate_map`, `PPO_logp_err`, `jobDone`, `block`, `liveLock`. Trước khi train PPO, code xác minh cùng graph topology, tỷ lệ legal candidate mapping ≥99% và rollout log-probability parity. Nếu fail: **không tiếp tục train dài**.

**Quan trọng:** Khi dùng checkpoint từ epoch thứ 2, guide vẫn đang ở gần giá trị start, chưa ở mức cuối. Đừng đánh giá dựa trên mô tả “annealed” nếu chỉ chạy 2 episodes.

## Bước 3 — Đánh giá held-out 3.600 giây (sau smoke)

```powershell
python -u scripts/evaluate_B3_edgeGAT_MAPPO_v4.py `
  --checkpoint mappo_runs/B3_EDGE_GAT_V4_smoke_seed43/checkpoint_ep_0002.pt `
  --seeds 1001 1002 1003 `
  --sim-seconds 3600 `
  --port 51731 `
  --output mappo_runs/B3_EDGE_GAT_V4_eval_3600.jsonl
```

Đây là **diagnostic** vì checkpoint 2 episodes còn non. Eval tự lấy guidance strength tương ứng với episode checkpoint; có thể chủ động làm ablation `--guide-strength 0.0` nhưng **kết quả khi ấy là thay policy lúc inference**, không phải chứng minh đã huấn luyện không guide.

```powershell
python scripts/compare_marl_runs_uoe_v2.py `
  mappo_runs/B3_EDGE_GAT_V4_eval_3600.jsonl
```

## Bước 4 — Sau smoke, huấn luyện có ý nghĩa

Tạo **output mới**, ví dụ 50 episodes, 3.600 giây mỗi episode (có thể chạy khá lâu):

```powershell
python -u scripts/train_B3_edgeGAT_MAPPO_v4.py `
  --model envs/Matrix_critic_fresh.zip `
  --port 51732 `
  --episodes 50 `
  --sim-seconds 3600 `
  --seed 43 `
  --edge-heads 2 `
  --routing-guide-strength 0.5 `
  --guide-final 0.2 `
  --guide-decay-episodes 30 `
  --output mappo_runs/B3_EDGE_GAT_V4_50ep_seed43
```

Sau đó đánh giá checkpoint cuối bằng script evaluation trên **cùng seeds** 1001/1002/1003 như V2; tốt nhất lặp huấn luyện với nhiều seed. KPI tối thiểu: `jobDone`, `targetReached`, `wrongTarget`, `block`, `liveLock`, `delivered_reward`.

## Các nguyên tắc tránh lỗi

- Dijkstra prior và mask phải giữ **nguyên giá trị đã dùng khi rollout** trong PPO; không scale bias lại trong optimizer.
- `guide_start` và `guide_final` là bias logits, không phải xác suất.
- Giá trị thấp có thể khiến policy non trẻ chọn đường kém hơn; giá trị cao từng được đo là lấn át GAT. Không thay đổi nhiều yếu tố cùng lúc.
- Giữ `Stop` masked nếu không sửa môi trường; `stops=0` trong V2 không phải do GAT học.
- `networkcontext=None` vẫn cần xem ở Alpyne nếu cảnh báo còn lặp. Patch không tự tuyên bố sửa nó.
- Model save/load checkpoint của V4 phải được tái dựng bằng **EdgeGAT V4 architecture**, không nạp trực tiếp vào V2 GAT.
- Không đưa `mappo_runs` và checkpoints lên GitHub Public nếu `.gitignore` đang bỏ qua.
