# B3-V6 — Huấn luyện dài C/D và đánh giá trên seed mới

## 0. Bối cảnh và điều KHÔNG thay đổi

Từ thí nghiệm A/B/C/D V5, C và D là hai cấu hình Edge-GAT có kết quả tốt:
- **C:** Routing = Residual Edge-GAT, Dijkstra logit prior **1.5**, cố định.
- **D:** Routing = Residual Edge-GAT, Dijkstra logit prior **0.5**, cố định.

Trong cả hai nhánh:
- **Routing** dùng Residual Edge-GAT + MAPPO + Dijkstra guidance.
- **Dispatching** giữ **B2 Operation-first + pooled/MLP**, KHÔNG dùng GNN.
- Giữ PPO/GAE/centralized critic, action mask, model AnyLogic, đồ thị, cùng tham số và ngân sách train.
- **Train lại từ đầu cho mỗi cặp (C/D, training seed)**. Không resume từ checkpoint V5 (tránh không công bằng khi đổi số episodes).
- Cùng training seeds (43, 143, 243), đánh giá trên 2001–2005; không dùng seed từng xuất hiện trong training cho evaluation.

Nguồn tham khảo phương pháp:
- EPyMARL: https://github.com/uoe-agents/epymarl — quản lý cấu hình, seeds và logs.
- MAPPO: https://github.com/marlbenchmark/on-policy — giữ đồng nhất optimizer, clip, PPO epochs.
- Lei et al.: https://github.com/Lei-Kun/DRL-and-graph-neural-network-for-routing-problems — edge-aware attention cho Routing.
- Bài báo Steinbacher, Wegmann & Freitag (2025), mục 3.2.4 và 3.3: phân biệt routing và dispatching.

**Quan trọng:** Đây là bộ điều phối thí nghiệm/đo KPI, không phải nâng cấp thuật toán mới. Chưa chạy AnyLogic ở môi trường tạo patch này; không có kết quả train V6 nào được tuyên bố trước.

## 1. Cài đặt

Giải nén `B3_V6_CD_Replication_patch.zip` trực tiếp vào:

`C:\MARL_AnyLogic\MAPPO_NEW\thesis`

Tệp mới (không ghi đè code MAPPO cũ):

```
scripts/v6_cd_common.py
scripts/run_B3_V6_CD_replication.py
scripts/analyze_B3_V6_CD_replication.py
tests/test_B3_V6_CD_offline.py
README_B3_V6_CD.md
```

Tệp bắt buộc đã có từ V4:

```
scripts/train_B3_edgeGAT_MAPPO_v4.py
scripts/evaluate_B3_edgeGAT_MAPPO_v4.py
envs/Matrix_critic_fresh.zip
```

Cần dùng môi trường `.venv311` đã chạy V5; chạy từ thư mục gốc `thesis` trong PowerShell.

## 2. Test và xem trước (chưa chạy AnyLogic)

```powershell
python tests/test_B3_V6_CD_offline.py
python scripts/run_B3_V6_CD_replication.py --dry-run
```

Test kỳ vọng 4 tests / OK. `--dry-run` in **6 lệnh train + 6 lệnh eval** và không tạo file.

## 3. Chạy V6 (khuyến nghị trước tiên 30 episodes × 720 s)

```powershell
python -u scripts/run_B3_V6_CD_replication.py `
  --train-seeds 43 143 243 `
  --eval-seeds 2001 2002 2003 2004 2005 `
  --train-episodes 30 `
  --train-sim-seconds 720 `
  --eval-sim-seconds 3600 `
  --output mappo_runs/B3_V6_CD_30ep_720s
```

**Khối lượng**: 2 cấu hình × 3 training seeds × 30 episodes = **180 training episodes**; 2 × 3 × 5 = **30 evaluation episodes**, mỗi evaluation 3600 giây mô phỏng. Đây là thời gian mô phỏng, **không** phải thời gian chạy đồng hồ trên máy. Runner chạy tuần tự từng model để tránh đụng cổng Java.

Lựa chọn chạy theo 2 giai đoạn:

```powershell
# Train trước, chưa evaluation
python -u scripts/run_B3_V6_CD_replication.py --mode train --output mappo_runs/B3_V6_CD_30ep_720s

# Sau khi train xong cả 6 lượt mới evaluate (giữ nguyên thông số)
python -u scripts/run_B3_V6_CD_replication.py --mode eval --output mappo_runs/B3_V6_CD_30ep_720s
```

Nếu chương trình đã hoàn tất một vài lượt nhưng bị dừng trước khi hoàn thành toàn bộ, chạy lại **cùng thông số** và thêm `--resume`:

```powershell
python -u scripts/run_B3_V6_CD_replication.py --resume --output mappo_runs/B3_V6_CD_30ep_720s
```

**`--resume` chỉ SKIP lượt đã hoàn tất và xác thực được. Không nối tiếp huấn luyện từ checkpoint đang dở.** Nếu lượt train bị ngắt ở episode 19/30, tool sẽ từ chối tự ghi đè. Giữ lại thư mục để kiểm tra rồi quyết định chạy lại lượt đó vào output khác hoặc chuyển kết quả dở đi trước.

Mỗi lượt có `run_config.json`, `metrics.jsonl`, `checkpoint_ep_XXXX.pt`, `latest_mappo.pt` và `evaluation_3600.jsonl`, cùng log stdout trong thư mục gốc V6. Không sửa dữ liệu gốc B2, V2, V4, V5.

## 4. Tổng hợp sau khi tất cả train/eval thành công

```powershell
python scripts/analyze_B3_V6_CD_replication.py --folder mappo_runs/B3_V6_CD_30ep_720s
```

Chương trình kiểm tra **đủ toàn bộ các cặp train/eval seed**, mô hình/graph fingerprint và các tham số ngoài strength đồng nhất; thiếu một lượt sẽ **báo lỗi**, không tính trung bình thiếu dữ liệu.

Kết quả:

```
mappo_runs/B3_V6_CD_30ep_720s/
  experiment_manifest.json
  C_edge_gat_guide1p5/train_seed43/...   # và 143, 243
  D_edge_gat_guide0p5/train_seed43/...   # và 143, 243
  V6_all_evaluations.csv
  V6_per_training_seed.csv
  V6_C_vs_D_summary.csv
  V6_CD_REPORT.md
  logs/...
```

V6 báo **D - C ghép cặp theo cùng (train seed, eval seed)**, chênh lệch jobDone/block/livelock/reward và SD theo **ba training seeds độc lập**. Không coi 15 evaluations/cấu hình là 15 training models độc lập. 3 training seeds chưa đủ để khẳng định ý nghĩa thống kê đáng tin cậy.

Chỉ số quan trọng: `jobDone` lớn (ưu tiên chính), `block`, `liveLock`, `wrongTarget` nhỏ, `targetReached`, reward lớn. Nếu dùng seeds 2001–2005 để **chọn** C hoặc D, cần để dành test seeds mới cho lần kết luận cuối.

## 5. Nếu muốn train dài hơn

Sau khi đánh giá 30ep, có thể thiết kế một thí nghiệm **mới từ đầu** (không ghi đè và không nối sai ngân sách), ví dụ 50 episodes với 720s:

```powershell
python -u scripts/run_B3_V6_CD_replication.py `
  --train-episodes 50 --train-sim-seconds 720 `
  --output mappo_runs/B3_V6_CD_50ep_720s
```

Nếu muốn cả mỗi training episode dài 3.600 giây, dùng `--train-sim-seconds 3600` cùng output mới. Tuy nhiên chi phí chạy sẽ tăng nhiều; nên làm sau khi xem kết quả 30ep.

## 6. Kiểm soát và giới hạn

- C/D cùng Edge-GAT, chỉ Dijkstra prior strength khác nhau.
- Không thay reward, kiến trúc Dispatching, mô hình AnyLogic hay thuật toán MAPPO.
- Giữ ngẫu nhiên train của hai nhánh cùng seed, nhưng do mô phỏng và chính sách khác, trajectory vẫn có thể khác; vì vậy đây là thiết kế ghép cặp theo seed, không phải kiểm soát mọi biến ngẫu nhiên.
- `--dry-run` không gọi Java; tests offline dùng JSON giả, **không chứng minh chương trình chạy AnyLogic thật**.
- Nếu thấy warning `networkcontext=None`, đừng mặc nhiên coi là đã hỏng graph; xem graph mapping audit và các warning riêng.
- `.gitignore` trước đây bỏ qua `mappo_runs/` và `*.pt`, nên sẽ không tự đẩy checkpoint lên GitHub.
