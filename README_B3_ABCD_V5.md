# B3-V5: Thí nghiệm A/B/C/D (GAT/Edge-GAT × Dijkstra 1.5/0.5)

## 1. Mục tiêu và phạm vi

Bộ này **không thay đổi thuật toán MAPPO**, cũng **không đưa GNN vào Dispatching**. Nó chạy một thiết kế thực nghiệm 2×2 để xác định mức cải thiện trước đó có thể đến từ **Residual Edge-GAT**, từ việc **giảm Dijkstra prior**, hay từ **tương tác của hai yếu tố**.

| Nhánh | Routing Actor | Dijkstra bonus cộng vào logits |
|---|---|---:|
| A | Directed GAT (V2) | 1.5 |
| B | Directed GAT (V2) | 0.5 |
| C | Residual Edge-GAT (V4) | 1.5 |
| D | Residual Edge-GAT (V4) | 0.5 |

Cả bốn nhánh:
- train **riêng từ đầu**, dùng **cùng training seeds**, cùng hyperparameters PPO, fleet size, action mask, thời lượng mỗi episode, cùng model AnyLogic;
- dùng **guidance strength cố định khi train và khi đánh giá**. Không lấy checkpoint đã train với prior 1.5 để evaluate tạm với prior 0.5 rồi gọi đó là thí nghiệm B;
- giữ **Dispatching B2 Operation-first, pooled/MLP, MAPPO-CTDE**; thay duy nhất encoder Routing và Dijkstra logit strength;
- dùng evaluation seeds giống nhau, tách biệt với mọi environment seeds trong training;
- lưu mỗi model, config, metrics và eval trong thư mục riêng. Không resume một checkpoint sang architecture khác.

**Mã được xây từ các script V2/V4 trên máy bạn:** repo GitHub chính có V2 nhưng V4 vừa tạo có thể vẫn chưa được push. Không phụ thuộc Internet khi chạy.

## 2. Nguồn gốc ý tưởng

- `uoe-agents/epymarl`: cấu hình thí nghiệm, nhiều seed, giữ log/checkpoint và so sánh công bằng. https://github.com/uoe-agents/epymarl
- `Lei-Kun/DRL-and-graph-neural-network-for-routing-problems`: Residual Edge-GAT khai thác thông tin cạnh và node trong bài toán định tuyến. https://github.com/Lei-Kun/DRL-and-graph-neural-network-for-routing-problems
- Steinbacher, Wegmann & Freitag (2025), *Production control with Reinforcement Learning for a matrix-structured production system*: dùng graph-attention cho routing, còn dispatching là mạng không GNN; có vấn đề mất ổn định khi train chung.

Đây là **bộ điều phối và phân tích mới viết** dựa trên các thành phần sẵn có. Chưa phải kết quả thực nghiệm; chưa chạy AnyLogic trên máy của bạn.

## 3. Cài đặt

Giải nén nội dung ZIP **vào thư mục gốc**:

`C:\MARL_AnyLogic\MAPPO_NEW\thesis`

Các file **mới**, không ghi đè code cũ:

```
scripts/ablation_abcd_v5_common.py
scripts/run_B3_ablation_ABCD_v5.py
scripts/analyze_B3_ablation_ABCD_v5.py
tests/test_B3_ablation_ABCD_v5_offline.py
README_B3_ABCD_V5.md
```

Phải có sẵn:

```
scripts/train_B3_GNN_UOE_v2.py
scripts/evaluate_B3_GNN_UOE_v2.py
scripts/train_B3_edgeGAT_MAPPO_v4.py
scripts/evaluate_B3_edgeGAT_MAPPO_v4.py
envs/Matrix_critic_fresh.zip
```

Và các package `thesis/mappo` của V1/V2/V4 như trước, cùng Python `.venv311` hoạt động. Chỉ cần một tiến trình AnyLogic/Alpyne dùng port tương ứng: runner chạy **tuần tự**, không chạy 4 mô phỏng cùng lúc.

## 4. Bước 1 — Kiểm tra offline

```powershell
python tests/test_B3_ablation_ABCD_v5_offline.py
```

Kỳ vọng `Ran 3 tests ... OK`.

## 5. Bước 2 — Xem chính xác 8 lệnh trước khi chạy

```powershell
python scripts/run_B3_ablation_ABCD_v5.py --dry-run
```

Sẽ hiện **4 lệnh train + 4 lệnh evaluate**, chưa chạy AnyLogic hay ghi file.

## 6. Bước 3 — Chạy pilot công bằng

```powershell
python -u scripts/run_B3_ablation_ABCD_v5.py `
  --train-seeds 43 `
  --eval-seeds 1001 1002 1003 `
  --train-episodes 2 `
  --train-sim-seconds 720 `
  --eval-sim-seconds 3600 `
  --output mappo_runs/B3_ABCD_V5_pilot
```

**Tổng khối lượng:** 4 model × 2 training episodes 720 giây mô phỏng; 4 model × 3 held-out evaluations 3.600 giây mô phỏng. Có thể tốn thời gian chạy thật dù tổng thời gian mô phỏng được cố định. Đây chỉ là **pilot**, không kết luận ý nghĩa thống kê.

Nếu bị ngắt sau một số nhánh **đã hoàn tất**, chạy lại đúng lệnh và thêm `--resume`. Tool chỉ bỏ qua những lượt có cả checkpoint/config/metrics hợp lệ; thư mục đang chạy dở sẽ dừng để tránh ghi đè, không âm thầm tiếp tục bằng checkpoint không đủ thông tin.

Nếu Java/AnyLogic đang giữ cổng `51780` hay `51781`, đổi qua hai cổng khác:

```powershell
--port-train 51800 --port-eval 51801
```

**Không dùng `--mode eval` nếu chưa train cả 4 nhánh.**

## 7. Bước 4 — Tổng hợp và đọc hiệu ứng

```powershell
python scripts/analyze_B3_ablation_ABCD_v5.py `
  --folder mappo_runs/B3_ABCD_V5_pilot
```

Kết quả sẽ nằm tại:

```
mappo_runs/B3_ABCD_V5_pilot/
  experiment_manifest.json
  A_gat_guide1p5/train_seed43/...
  B_gat_guide0p5/train_seed43/...
  C_edge_gat_guide1p5/train_seed43/...
  D_edge_gat_guide0p5/train_seed43/...
  all_evaluations.csv
  condition_summary.csv
  factorial_contrasts.csv
  ABCD_REPORT.md
```

Ba hiệu ứng trong `factorial_contrasts.csv`:

- **Edge-GAT – GAT** = `(C + D − A − B)/2`
- **Guide 0.5 – Guide 1.5** = `(B + D − A − C)/2`
- **Interaction** = `(D − C) − (B − A)`

Các công thức tính trên **chỉ số ghép cặp cùng training seed và evaluation seed**, rồi tổng hợp theo từng training seed. Không nhầm 3 evaluation seeds là 3 independent training runs.

Chỉ số chính: `jobDone` (cao tốt), `block`/`liveLock` (thấp tốt). Cần xét đồng thời `targetReached`, `wrongTarget`, `delivered_reward`.

## 8. Giai đoạn nghiên cứu đầy đủ (sau khi pilot chạy ổn)

Cần training **ít nhất 3 seed độc lập**, mô phỏng train dài hơn, trong khi **không thay bất kỳ yếu tố nào khác**:

```powershell
python -u scripts/run_B3_ablation_ABCD_v5.py `
  --train-seeds 43 143 243 `
  --eval-seeds 1001 1002 1003 `
  --train-episodes 30 `
  --train-sim-seconds 3600 `
  --eval-sim-seconds 3600 `
  --output mappo_runs/B3_ABCD_V5_full
```

**Không dùng cùng output folder pilot:** vì khác ngân sách training. Full run = 4×3×30 = **360 training episodes** (mỗi episode 3.600 giây mô phỏng) và **36 evaluation episodes**; có thể rất lâu. Có thể chạy `--mode train` trước rồi `--mode eval`, giữ nguyên các tùy chọn. Khi hoàn tất mới chạy analyzer.

## 9. Kiểm soát chất lượng và giới hạn

- Bộ test offline đã được kiểm tra: cấu hình đủ A/B/C/D, training/eval không trùng seed, guide cố định trong V4, phép tính hiệu ứng chính và tương tác trên dữ liệu giả lập, từ chối file kết quả thiếu seed.
- Chưa chạy AnyLogic thực tế trong môi trường tạo patch. **Không khẳng định cả bốn nhánh sẽ chạy xong cho đến khi có log từ máy bạn**.
- V4 **chỉ giữ guide strength cố định** bằng việc truyền `--guide-final` bằng `--routing-guide-strength`, không dùng schedule giảm dần như V4 thử trước đó. Như vậy factor guidance được cô lập.
- Graph fingerprint phải nhất quán; action mask Stop như hiện tại; đoạn này **không sửa reward và không thêm GNN cho Dispatching**.
- Nếu một nhánh fail thì runner dừng luôn, không tự trộn kết quả thiếu vào báo cáo.
- Nếu training chỉ 2 episodes, việc nhánh thắng có thể do random initialization hay policy chưa học; phải làm thí nghiệm lặp mới có nhận định đáng tin cậy.
- Các trường `*.zip` và `mappo_runs/` đang được `.gitignore` bỏ qua trên máy bạn: giữ file model và checkpoint local, không tự push chúng lên GitHub.
