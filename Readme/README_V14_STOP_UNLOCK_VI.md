# V14 STOP-UNLOCK — Mở lại action Stop, tiếp tục từ checkpoint đã học

## Thay đổi duy nhất

- **Routing `stop_mask_mode='baseline'`**, tức KHÔNG áp đặt `mask[0]=0` khi có hướng đi hợp lệ. Giữ nguyên action mask gốc từ AnyLogic (các hướng đi không khả thi vẫn bị cấm).
- Mạng GAT4, 2 Dispatch Deep Candidate-Aware MLP, 3 Critics, reward, 20 AGV, 3600s, 128 mẫu mỗi minibatch, tính log-prob PPO, no-Dijkstra: **không đổi**.
- **Warm start** từ checkpoint *committed* cuối cùng của V14 gốc. Kế thừa **actor weights, critic weights, Adam optimizer states, value normalizers** thông qua `model.load(checkpoint)`; sao chép `.pt` gốc nguyên byte sang run mới. KHÔNG phải full scratch.
- Bản nguồn giữ nguyên. Nhánh mới có `experiment.json` ghi `fork_episode`, `source_checkpoint`, hash checkpoint, `stop_mask_mode=baseline`.

## 0. Dừng V14 an toàn

Trong Terminal VS Code đang training, bấm **Ctrl+C một lần**. Chờ terminal in `[V14 INTERRUPTED] Last committed EP ...` và trả prompt. Không `Stop-Process`/End Task nếu Ctrl+C hoạt động bình thường.

```powershell
cd C:\MARL_AnyLogic\MAPPO_NEW\thesis
$old = '.\mappo_runs\V14_GAT4_MLP_3600S_MB128'
$s = Get-Content "$old\state.json" -Raw | ConvertFrom-Json
Write-Host "Committed episode = $($s.last_episode)"
Write-Host "Checkpoint = $($s.checkpoint)"
Test-Path $s.checkpoint
```

**PHẢI thấy `True`.** Nếu thư mục gốc khác, thay `$old` cho đúng `--output` ông đã dùng. Không xóa/checkpoint cũ.

## 1. Cài patch

Giải nén ZIP vào `C:\MARL_AnyLogic\MAPPO_NEW\thesis` (cùng chỗ có `scripts`, `thesis`, `.venv311`). ZIP chỉ thêm:
- `scripts/train_v14_stop_unlocked.py`
- `tests/test_v14_stop_unlocked_offline.py`
- README này.

V14 trainer cũ không bị thay đổi.

## 2. Kiểm tra offline và nguồn checkpoint (KHÔNG train)

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p 'test_v14_stop_unlocked_offline.py' -v

.\.venv311\Scripts\python.exe -u .\scripts\train_v14_stop_unlocked.py --fork-from '.\mappo_runs\V14_GAT4_MLP_3600S_MB128' --output '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK' --device cuda:0 --sim-seconds 3600 --max-minibatch 128 --max-episodes 1100 --dry-run
```

Nếu `--dry-run` báo `Fork args differ from original V14 setup`, chạy đúng các tham số của run cũ (chẳng hạn `--port`, `--snapshot-every`, `--gat-heads`). Đừng sửa `experiment.json`.

## 3. Tạo nhánh mới, kế thừa checkpoint, chưa train

**Copy nguyên một dòng PowerShell**, không thêm dấu backtick xuống dòng:

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_stop_unlocked.py --fork-from '.\mappo_runs\V14_GAT4_MLP_3600S_MB128' --output '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK' --device cuda:0 --sim-seconds 3600 --max-minibatch 128 --max-episodes 1100 --init-only
```

Kỳ vọng:

```text
[V14 STOP UNLOCK FORK] inherited EPxxx checkpoint; Stop allowed when moves exist
[V14 STOP UNLOCK FORK ONLY] checkpoint inherited, no PPO rollout
```

`xxx` là episode cuối ghi trong `state.json` V14 gốc. Trong thư mục nhánh mới sẽ có `rolling_slot_0.pt` hoặc `rolling_slot_1.pt` tương ứng `xxx % 2`, cùng `state.json`, `experiment.json`, `run_config_blueprint.json`, và `metrics.jsonl` rỗng (chỉ ghi các episode sau fork).

## 4. Test một episode với mask mới

```powershell
.\.venv311\Scripts\python.exe -u .\scripts\train_v14_stop_unlocked.py --output '.\mappo_runs\V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK' --device cuda:0 --sim-seconds 3600 --max-minibatch 128 --max-episodes 1100 --resume
```

Lệnh này sẽ tiếp tục đến EP1100; **để thử đúng một episode**, chạy thêm `--max-episodes X`, với X = `last_episode` trước khi fork + 1 (ví dụ nguồn EP92 thì `--max-episodes 93`). Sau khi một episode hoàn tất và được commit, chạy lệnh bên trên để train tiếp EP1100.

**KHÔNG thêm `--fork-from` khi dùng `--resume`**. Dùng đúng `--output` nhánh mới.

## 5. Xác minh `Stop` đã mở

Mở `mappo_runs/V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK/metrics.jsonl` sau EP đầu của nhánh; kiểm tra:

- `environment.action_diagnostics.stop_mask_mode == "baseline"`.
- `environment.action_diagnostics.routing_stop_masked_decisions == 0`.
- `environment.action_diagnostics.routing_stop_with_move_available` là **số lần policy tự chọn Stop khi vẫn có hướng đi**; có thể bằng 0 ở một episode, không đồng nghĩa Stop bị khóa.
- `ppo_audit`, `all_roles_audit`: pass; `selected_mask_violations == 0`.
- Theo dõi `jobDone`, `block`, `liveLock`, `_blocksec_agv*`, `targetReached`. Không mặc định Stop sẽ giúp: policy ban đầu chưa được train hành động này ở những state có thể di chuyển.

## Cơ sở kỹ thuật

Trong `thesis/mappo/collector_stop_ab.py` của repo, `_prepare()` chỉ đặt `mask[0]=0` khi `stop_mask_mode == 'forbid_when_move'`. Chế độ `baseline` không làm việc này. `_prepare` trả mask lưu vào transition, `model.act()` dùng chính mask đó. PPO update nhận transition với mask gốc đã ghi, giúp rollout và optimization nhất quán.

Khác biệt giữa **thay đổi mask lúc inference** và **train tiếp theo mask mới** quan trọng. Nghiên cứu Huang & Ontañón (2022), *A Closer Look at Invalid Action Masking in Policy Gradient Algorithms*, DOI: 10.32473/flairs.v35i.130584 chứng minh mask ảnh hưởng phân phối hành động và policy gradient. Vì vậy nhánh này tiếp tục **train** với mask mới, không dùng checkpoint cũ rồi đổi mask chỉ lúc đánh giá.

## Giới hạn kiểm thử

Đã kiểm tra syntax và các tình huống fork/preflight offline. Cần chạy bộ test trên cây source V14 đầy đủ tại Windows để kiểm tra test trực tiếp `EventCollector`, và cần kiểm chứng EP đầu trong AnyLogic/Alpyne thật. Đừng báo cáo kết quả nhánh mới như một lần train full scratch.
