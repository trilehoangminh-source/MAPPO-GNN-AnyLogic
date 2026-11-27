---
name: mappo-project-context
description: Hướng dẫn tiếp nối dự án MAPPO–GNN–AnyLogic, đọc tài liệu có chọn lọc và theo dõi phiên bản.
---

# HƯỚNG DẪN DỰ ÁN VÀ BÀN GIAO NGỮ CẢNH

Cập nhật: **25/11/2026**, múi giờ **Asia/Saigon**. File này là điểm vào cho AI làm việc trong dự án. Đầu phiên mới, người dùng hãy yêu cầu AI đọc `Readme/Skill.md`; không mặc định mọi ứng dụng AI tự động nạp file này chỉ nhờ tên file.

Source đã gom theo chức năng tại nhánh `refactor/function-based-source`, HEAD được kiểm tra khi cập nhật: `53894aaeb8955c5cf2ecf67665ee3963521a40bc`. Kiến trúc vẫn là V17-B; đổi tên file không tạo ra phiên bản thuật toán mới. Working tree còn thay đổi local, nên HEAD không đại diện cho mọi file hiện có. Nguồn đối chiếu: source hiện tại, [context Codex 24/11 lúc 22:54](Context/CONTEXT_CODEX_20261124_225405.md) và công việc xuất model/đánh giá EP470 ngày 25/11.

## PHẦN 1 — CẤU TRÚC, FILE CHÍNH VÀ PHIÊN BẢN DỰ ÁN

### 1.1. Gốc dự án và cây thư mục

Gốc Windows: `C:\MARL_AnyLogic\MAPPO_NEW\thesis`.
Các đường dẫn bên dưới tính từ gốc này. Thư mục `thesis/` bên trong là Python package, nên đường dẫn `thesis\thesis\mappo` trên Windows là đúng.

```text
thesis/                              # Gốc repository
├── Readme/
│   ├── Skill.md                     # File hướng dẫn hiện tại
│   └── Context/                     # Context bàn giao của từng phiên chat
├── scripts/                         # Khởi chạy, huấn luyện, đánh giá, vẽ biểu đồ
├── thesis/                          # Python package
│   ├── envs/                        # Observation, action, hành vi, cầu nối AnyLogic
│   ├── mappo/                       # Actor, Critic, GNN, PPO, collector
│   ├── policies/                    # Các policy và khối mạng từ nền Tobias
│   ├── pypeline_connectors/         # Các connector của dự án; giữ đúng cách viết tên
│   └── utils/                       # Công cụ cấu hình
├── envs/                            # Mô hình AnyLogic, khác thesis/envs/
│   ├── Matrix1/                     # Matrix1.alp, Matrix1_AGV17_trace.alp, Matrix1_animation.alp
│   ├── Matrix/, MiniMatrix/, ...    # Các mô hình khác; chỉ đọc khi cần
│   ├── Matrix_RL_debug.zip          # Model lịch sử của run V17-B
│   └── Matrix1_animation_RL_runmode1.zip # RLExperiment xuất ngày 25/11, runmode=1
├── tests/                           # Các kiểm thử Python
├── mappo_runs/                      # Checkpoint, metadata, metrics và held-out
├── .venv311/                        # Python dùng cho dự án
├── .git/, .vscode/, .pytest_cache/   # Git, cấu hình và cache
├── .gitignore
├── requirements.txt
└── RELEASE_V16B_TEST_REPORT.txt      # Báo cáo lịch sử V16-B
```

Đây là bản đồ các phần cần biết, không phải danh sách mọi file. Khi cần tra cứu, tìm bằng `rg --files` hoặc `rg -n` trong đúng thư mục; tránh đọc toàn bộ repository.

### 1.2. Các file cần mở theo công việc

| Công việc | File chính |
|---|---|
| Lệnh PowerShell; kiểm tra dependency trước khi dùng | `scripts/run.ps1` |
| Train, khởi tạo/resume, metadata, fingerprint, seed | `scripts/train.py` |
| Evaluator chuẩn cho metadata/source/model khớp run | `scripts/evaluate.py` |
| Dựng AnyLogic, đồ thị và mạng | `scripts/model_builder.py` — `build_model`, `TOBIAS_JOINT_REWARDS` |
| Phase, fleet và policy adapter | `scripts/stage_policy.py` — `PHASES`, `StagePolicyAdapter` |
| Actor, Critic, 7 edge features, chuyển checkpoint | `thesis/mappo/model_architecture.py` — `TrafficRoutingActor`, `EDGE_FEATURE_NAMES`, `install_base_architecture`, `install_traffic_routing`, `transfer_checkpoint` |
| PPO, ổn định update và gradient accumulation | `thesis/mappo/algorithm.py`, `ppo_stability.py`, `fullrollout_optimizer.py` |
| Collector, event/trajectory, trạng thái cuối episode | `thesis/mappo/collector.py` — `EventCollector.episode` |
| Topology và fingerprint đồ thị | `thesis/mappo/routing_graph.py` — `GraphSpec.from_hive` |
| Routing guidance dùng khi truy vết; builder hiện không bật Dijkstra | `thesis/mappo/routing_guidance.py` |
| Observation/action, reset, giao tiếp Alpyne | `thesis/envs/matrix.py`, `behaviors.py`, `base_alpyne_zoo.py`, `alpyne_bridge.py` |
| Dispatch mask và cấu hình môi trường | `thesis/envs/dispatch_mask.py`, `thesis/utils/build_config.py` |
| Animation actor-only EP490 qua Pypeline | `thesis/pypeline_connectors/matrix_animation.py`, `envs/Matrix1/Matrix1_animation.alp` |
| Tổng hợp held-out và đồ thị | `scripts/summarize_heldout.py`, `scripts/plot_metrics_gui.py`, `scripts/plot_mappo_training_timesteps.py` |
| Kiểm thử có sẵn | `tests/test_training_pipeline.py`, `test_traffic_routing.py`, `test_dispatch_mask.py`, `test_mappo.py`, `test_mappo_critic_state.py` |

Các đường dẫn cũ như `scripts/v16b.ps1`, `train_v16b_fullrollout.py`, `evaluate_v16b_traffic.py`, `v15_tobias_build.py`, `v15_staged_collector.py`, `thesis/mappo/v16b_traffic_edge.py` và các module môi trường `_v121` là tên trước refactor, không dùng làm entrypoint hiện tại. Một số tên class, comment hoặc nhãn log còn ghi V15/V16; xác nhận hành vi bằng thân hàm, `VERSION` và metadata.

### 1.3. Phiên bản và quyết định thiết kế

| Phiên bản | Thiết kế / mốc đã xác nhận | Nhánh / run root |
|---|---|---|
| V15 | Nền Tobias: Routing GATv2, Dispatch LinFE/MLP, ba Centralized Critic; Full-Rollout GA512 | `refactor/v15-minimal`; `mappo_runs/V15_FULLROLLOUT_GA512/` |
| V16-B | Bốn edge features `[L,F,O,D]` tác động attention; bản tổng kết cũ ghi warm-start từ V15 Joint12 EP850 | `feature/v16b-traffic-edge-gatv2`; `mappo_runs/V16B_TRAFFIC_EDGE_GATV2_GA512/` |
| V17-B trước refactor | Bảy edge features, edge-conditioned attention và message passing; run lịch sử warm-start từ V15 Joint16 `initial_ep_0000.pt` | `feature/v17b-edge7-message`; `mappo_runs/V17B_EDGE7_MSG_GA512/` |
| **Source hiện tại, kiến trúc V17-B** | Gom module theo chức năng; `VERSION = V17B_EDGE7_MESSAGE_GATV2_FULLROLLOUT_GA512_V1`; fingerprint khác source trước refactor | `refactor/function-based-source`; mặc định run mới `mappo_runs/TRAFFIC_EDGE_MESSAGE_GA512/` (chưa có khi cập nhật) |

V17-B giữ ba role `routing`, `dispatch_front`, `dispatch_rear`; Routing 5 action **không action mask** là chủ đích. GAT 2 lớp, 4 heads, embedding 32; duy trì reward và PPO GA512. Bảy feature theo thứ tự trong `EDGE_FEATURE_NAMES`: chiều dài, flow cùng chiều, flow ngược chiều, áp lực next-node đích, proxy xe đứng yên `last==next`, nhu cầu goal-node đích, bottleneck topology. Các bộ đếm xe khác loại ego và lấy từ actor-visible observation; proxy đứng yên không phải số đo hàng đợi vật lý.

Trạng thái và bằng chứng cần biết:

- Refactor commit `53894aa` đã được push theo context 24/11; lần cập nhật tài liệu này chỉ kiểm tra Git local, không xác minh remote mới và không commit/push.
- Run lịch sử `V17B_EDGE7_MSG_GA512/joint16/state.json` vẫn ghi EP583 khi đọc ngày 25/11. Đây là checkpoint ghi trong state, không phải kết luận training đang chạy.
- Snapshot EP470 còn tại `mappo_runs/V17B_EDGE7_MSG_GA512/joint16/snapshot_ep_0470.pt`. Blueprint còn tại **cùng thư mục**: `run_config_blueprint.json`; SHA256 khớp trường `blueprint_sha256` của experiment.
- `experiment.json` lịch sử vẫn trỏ blueprint tới `mappo_runs/V15_FULLROLLOUT_GA512/joint16/run_config_blueprint.json`, nhưng đường dẫn đó hiện thiếu. Ưu tiên bản sao V17-B đã kiểm hash khi chuẩn bị lệnh; không sửa metadata lịch sử để che việc di chuyển/mất file.
- EP490 lịch sử: 20 seed `900001–900020`, greedy, 3.600 giây/seed, không PPO update. Context ghi JobDone **13,70** (std **4,31**), Block **19,85**, LiveLock **119,90**, TargetReached **341,20**, WrongTarget **3,75**; bằng chứng ở `mappo_runs/V17B_EVAL_EP0490/joint16/heldout_joint16_20seeds_3600s.jsonl`. Chỉ đối chiếu với kết quả có cùng điều kiện; đọc lại file khi cần số liệu hiện tại.
- Ngày 25/11 đã xuất `envs/Matrix1_animation_RL_runmode1.zip` từ `Matrix1_animation.alp`, mặc định `runmode=1`. AnyLogic báo build/export thành công; kiểm tra toàn vẹn ZIP thành công. Việc xuất lại ZIP không chứng minh mọi hành vi mô phỏng giống model train cũ.
- Người dùng tự chạy đánh giá EP470 bằng source hiện tại và ZIP mới qua lệnh Python stdin; kết quả riêng ở `mappo_runs/V17B_EVAL_EP0470_ANIMATION_RL/joint16/heldout_joint16_20seeds_3600s.jsonl`. Đã quan sát hàng seed hoàn tất và thời gian mô phỏng tăng; phải đọc lại JSONL để xác nhận đủ 20 seed, không suy ra hoàn tất từ việc có thư mục.

Repository: `https://github.com/trilehoangminh-source/MAPPO-GNN-AnyLogic`.

### 1.4. Cập nhật khi cải tiến hoặc có phiên bản mới

Sau mỗi cải tiến đã thực hiện, cập nhật bảng phiên bản hoặc thêm một dòng nhật ký ngay bên dưới: ngày, phiên bản, thay đổi chính, file liên quan, nhánh/run root, kiểm tra đã chạy và việc còn chờ. Phân biệt đề xuất, code đã triển khai, kiểm thử offline và kết quả AnyLogic thật. Không nâng đề xuất thành kết quả thực nghiệm.

Giữ các mốc V15/V16/V17 để truy vết; chi tiết dài đặt trong context hoặc tài liệu chuyên đề rồi liên kết từ đây. Chỉ ghi commit/push khi có bằng chứng. Mọi việc commit, push, PR và merge phải theo yêu cầu của người dùng; việc cập nhật tài liệu không tự cấp quyền làm các bước Git đó.

| Ngày | Phiên bản / thay đổi | Bằng chứng / trạng thái |
|---|---|---|
| 24/11/2026 | Lập hướng dẫn dự án | Tạo `Readme/Skill.md`; `Readme/Context/` đã có sẵn |
| 24/11/2026 | Thêm quy trình chốt thiết kế ở phiên chat khác rồi bàn giao cho AI code | Mục 2.3; tài liệu thiết kế do người dùng gửi là căn cứ triển khai |
| 24/11/2026 | Gom source theo chức năng, giữ kiến trúc V17-B | Commit `53894aa`; chi tiết trong context 22:54, fingerprint mới khác run lịch sử |
| 25/11/2026 | Xuất model animation ở `runmode=1`, đánh giá snapshot EP470 bằng code hiện tại | ZIP mới và root `V17B_EVAL_EP0470_ANIMATION_RL`; evaluator CLI chưa được bổ sung chế độ snapshot/model mới |
| 25/11/2026 | Đồng bộ hướng dẫn với source sau refactor theo yêu cầu người dùng | Cập nhật bản đồ file, dependency còn thiếu, seed, giới hạn evaluator và quy trình xuất model; chỉ sửa tài liệu |

### 1.5. Phát triển phiên bản mới trên source hiện có, bảo tồn bằng GitHub

**Sửa trực tiếp các file source hiện có khi phát triển phiên bản mới.** Không tạo bản sao `V18`, `V19`… của cùng chức năng chỉ để giữ code phiên bản trước. Lịch sử Git và các nhánh trên GitHub đảm nhiệm việc bảo tồn source cũ. Chỉ thêm file khi thực sự cần một chức năng/module mới; không nhân bản cả bộ runner/actor/evaluator theo số phiên bản.

Trước khi sửa source để nâng phiên bản, thực hiện đúng thứ tự:

1. Kiểm tra nhánh hiện tại, working tree và các thay đổi cần bảo tồn. Commit source hiện tại lên nhánh phiên bản hiện tại rồi push nhánh đó lên GitHub (ví dụ V17 → `feature/v17b-edge7-message`). Chọn đúng file dự án; không tự đưa checkpoint, môi trường ảo, log tạm hoặc bí mật vào commit.
2. Xác nhận push thành công và SHA commit trên nhánh GitHub khớp với commit local cần bảo tồn; không chỉ dựa vào thông báo commit local hoặc remote-tracking ref cũ. Nếu push/xác minh thất bại, xử lý trước khi sửa source phiên bản mới.
3. Sau khi xác nhận đã lưu V17 trên GitHub, tạo và chuyển sang nhánh V18 theo tên nhánh đã chọn cho thiết kế mới. Chỉ lúc này mới chỉnh source hiện tại để triển khai V18. Đây là quy trình đã được người dùng yêu cầu cho các lần nâng phiên bản, không cần hỏi lại chỉ vì phải backup/push và tạo nhánh theo quy trình này; các giới hạn quyền của môi trường vẫn áp dụng.
4. Cập nhật version/metadata, tài liệu và kiểm thử liên quan trên nhánh mới. Giữ các tên theo chức năng như `train.py`, `evaluate.py`, `model_architecture.py`; không đổi hoặc sao chép thành bộ file `v18_*`. Ghi mapping phiên bản thực tế trong Skill.md/context để tránh nhầm.

Không dùng cách đóng gói tạo thêm bộ code theo từng phiên bản chỉ để giải nén song song và giữ code cũ. Nếu cần bàn giao ZIP, dùng đúng đường dẫn tương đối của các file hiện có để có thể ghi đè sau khi đã bảo tồn source trên GitHub. Không xóa hàng loạt các file nền V15/V16 đang được import; việc dọn hoặc hợp nhất chúng cần kiểm tra dependency và nằm trong phạm vi thay đổi được giao.

Git bảo tồn source; checkpoint/metrics trong `mappo_runs/` vẫn cần giữ riêng theo phiên bản thí nghiệm. Trước khi ghi đè source đang phục vụ training/resume, kiểm tra tiến trình đang chạy và cơ chế hash; không vô hiệu hóa hash hoặc ghi đè dữ liệu run cũ để tiếp tục phiên bản mới. Không tạo PR hay merge nếu chưa có yêu cầu tương ứng.

Quy trình bảo tồn trên áp dụng khi người dùng giao **nâng phiên bản/triển khai thiết kế mới**. Một yêu cầu đọc source, đánh giá hoặc sửa tài liệu không tự cấp quyền commit/push hay đổi nhánh. Phiên ngày 25/11 người dùng yêu cầu dùng source hiện tại để đánh giá, không lấy source cũ từ Git và không sửa code; sau đó cho phép đặt `runmode=1`/xuất model và cập nhật riêng file hướng dẫn này. Chưa có yêu cầu triển khai thay đổi evaluator.

### 1.6. Luồng train và seed của source hiện tại

- `scripts/train.py` dùng phase theo thứ tự `routing8`, `dispatch8`, `joint8`, `joint12`, `joint16`, `joint20`. Fleet lần lượt 8, 8, 8, 12, 16, 20; mỗi episode chạy tối đa 3.600 giây mô phỏng. `--max-episodes` là tổng EP mục tiêu trong phase, không phải số EP cần cộng thêm khi resume.
- `build_model` lắp ba role, 20 slot AGV padded, traffic GAT Routing và Dispatch LinFE. Phase `dispatch8` dùng native Tobias CollisionFreeRouting; các phase learned Routing không dùng Dijkstra guidance. Dispatch dùng mask `legacy`, Routing 5 action không mask. Blueprint lịch sử có thể còn ghi mask/kiến trúc cũ; đọc giá trị thực trong builder và actor.
- `--seed` mặc định lấy từ blueprint. Mỗi episode: `roll_seed = base_seed + phase_index * 100000 + episode - 1`; cùng seed đặt cho Python, NumPy, PyTorch, adapter và môi trường. Run lịch sử có `base_seed=43`; Joint16 có index 4 nên EP1 dùng `400043`, EP470 dùng `400512`. Đây là dãy seed xác định, không phải bốc seed tự do mỗi lần chạy.
- `Matrix.seed(None)` mới dùng `random.randint(0, 10000)` gồm cả hai đầu. Runner train truyền seed tường minh nên không dùng nhánh đó để chọn seed episode.
- `preflight` yêu cầu blueprint có `max_fleetsize=20`, `mini_batches=1` và các siêu tham số PPO. Default `--microbatch-size=512`, `--cpu-threads=2`, `--snapshot-every=10`; `max_events` lấy từ blueprint nếu không truyền. Run V17-B lịch sử ghi `max_events=20000`, thiết bị `cuda:0`.
- Dùng `--blueprint` và `--model` tường minh khi đường dẫn cũ không còn đúng. `_pick_blueprint` vẫn tìm các root V14 cũ nếu không truyền; không mặc định nó tự tìm blueprint V17-B còn tồn tại.
- `scripts/run.ps1` vẫn phụ thuộc `mappo_runs/V15_FULLROLLOUT_GA512/joint16/experiment.json`, blueprint và `initial_ep_0000.pt`. Khi cập nhật, các dependency V15 này không còn tại đường dẫn cũ, nên wrapper chưa thể dùng nguyên trạng. Không đưa lệnh init/train qua wrapper như thể đã sẵn sàng; chưa khởi tạo run mới `TRAFFIC_EDGE_MESSAGE_GA512`.

### 1.7. Đánh giá checkpoint và giới hạn CLI hiện tại

`scripts/evaluate.py` nhận `--phase`, `--blueprint`, `--model`, `--output-root`, `--device`, `--port`, `--seeds`, `--first-seed`, `--dry-run`, `--resume`. **Chưa có `--episode`, `--checkpoint` hay chế độ cho phép source/model mới.** Kiểm tra parser trước khi gửi lệnh; không đưa cờ chưa tồn tại.

- Script đọc `experiment.json` và `state.json` tại `<output-root>/<phase>/`; `_checkpoint` yêu cầu file checkpoint nằm ngay trong thư mục phase đó và SHA256 khớp state. Nó không chọn snapshot theo tên EP trong câu lệnh.
- Evaluator kiểm `VERSION`, fingerprint source, hash blueprint, hash ZIP và thiết bị so với experiment. `--blueprint`/`--model` chỉ đổi đường dẫn được đọc; **không** cho phép thay nội dung khác hash của run.
- Chọn EP cố định trong workflow source/model khớp run: chuẩn bị thư mục đánh giá riêng chứa experiment, snapshot đúng EP và state riêng có `last_episode`, đường dẫn checkpoint, hash đúng; giữ state và checkpoint của training. Cách này tự nó chưa giải quyết fingerprint khác sau refactor.
- `--output-root` vừa là nơi đọc metadata/checkpoint vừa là nơi ghi kết quả, chưa tách riêng nguồn run và nơi xuất kết quả. File kết quả: `<output-root>/<phase>/heldout_<phase>_<seeds>seeds_3600s.jsonl`.
- Mặc định 10 seed từ `900001`, chấp nhận 1–100 seed; 20 seed là `900001–900020`. Chọn action greedy, `torch.no_grad()`, không PPO update. Seed train và held-out cần tách nhau. Nếu dùng held-out nhiều lần để chọn checkpoint, nên có tập test cuối riêng khi kết luận khả năng tổng quát hóa.
- `--resume` kiểm hash checkpoint, seed trùng và seed ngoài tập yêu cầu rồi bỏ qua seed đã lưu. Đây là tính năng của evaluator chuẩn; lệnh Python stdin EP470 đã gửi **chưa có resume** và từ chối file kết quả đã tồn tại. Không hướng dẫn thêm `--resume` vào lệnh stdin hoặc xóa kết quả để chạy lại cho được.
- `--dry-run` kiểm các điều kiện trước mô phỏng, không chứng minh Java/server hoạt động hoặc checkpoint nạp vào model thực tế thành công. Sau refactor, không mặc định evaluator chuẩn chạy được với metadata cũ.

**Trường hợp EP470 + ZIP mới ngày 25/11:** người dùng muốn tự chạy terminal bằng source hiện tại và giữ nguyên file Python. Đã dùng lệnh PowerShell truyền Python qua stdin, import trực tiếp `build_model`, `StagePolicyAdapter`, `EventCollector`; không phải script mới đã lưu trong repo. Lệnh chọn snapshot EP470, blueprint bản sao V17-B đã kiểm hash, ZIP mới, `port=0`, 20 seed, 3.600 giây/seed; yêu cầu `transfer_checkpoint` trả về `v17b_strict_resume_all_states` và kiểm graph. Mỗi hàng JSONL lưu checkpoint SHA256, model SHA256 và hash source hiện tại. Đây là đánh giá bằng môi trường/code hiện tại, không giả mạo fingerprint training cũ và không ghi đè experiment/state lịch sử.

Checkpoint V17-B nạp bằng `strict load`; việc đổi tên module hoặc xuất ZIP không tự buộc train lại Joint16. Kiểm tương thích kiến trúc, observation/action, topology và điều kiện môi trường. Strict load/topology khớp chưa chứng minh động lực mô phỏng giống hoàn toàn; không trộn kết quả ZIP mới với kết quả model cũ như cùng một điều kiện thí nghiệm.

Hướng cải tiến **mới chỉ thảo luận, chưa triển khai**: thêm lựa chọn checkpoint trực tiếp, tách input run/output đánh giá, và chế độ đánh giá source/model mới có thông tin truy vết rõ ràng. Không sửa evaluator hoặc huấn luyện lại chỉ để vượt kiểm tra lineage khi chưa được giao.

Tổng hợp kết quả EP470 hiện tại bằng công cụ đã có (lệnh chỉ đọc file):

```powershell
.\.venv311\Scripts\python.exe scripts\summarize_heldout.py mappo_runs\V17B_EVAL_EP0470_ANIMATION_RL\joint16\heldout_joint16_20seeds_3600s.jsonl
```

Script tổng hợp không chờ đủ 20 seed; đọc số seed thực tế trong output trước khi kết luận hoàn tất. Khi so sánh, kiểm thêm EP, checkpoint/model hash, source, fleet, horizon và tập seed.

### 1.8. AnyLogic, ZIP RL và chẩn đoán khởi động

- Model nguồn hiện dùng: `envs/Matrix1/Matrix1_animation.alp`; mặc định `Main.runmode=1` sau thay đổi được phép ngày 25/11. Chọn **RLExperiment → Export for Reinforcement Learning** để tạo ZIP Alpyne có `model.jar`; không nhầm với xuất Simulation độc lập hoặc chỉ nén file `.alp`.
- Bản xuất: `envs/Matrix1_animation_RL_runmode1.zip`; SHA256 lúc xuất `0d7f2fd6eb295151fcbb854cc37f6f56f2b72d4bf92a031b9ace9a7d9e2265a6`. Xuất lại có thể đổi hash; luôn kiểm file thực tế và ghi hash model dùng đánh giá. Không ghi hash mới vào metadata training cũ để ép evaluator chấp nhận.
- `runmode=1` là luồng Python/Alpyne gửi action; `runmode=3` khởi tạo Pypeline animation. Connector `matrix_animation.py` hiện cố định **EP490**, không tự chuyển sang EP470 khi thay `runmode`.
- Connector animation còn trỏ blueprint tới V15 Joint16 đã mất; tính năng animation `runmode=3` cần xử lý đường dẫn khi được giao, không tuyên bố đang hoạt động chỉ vì RL ZIP `runmode=1` chạy được. Giữ thiết kế UI đã chốt: một khung chung, menu cố định, tab 2D/3D/Logic đổi nội dung trong cùng khung. Context 24/11 chưa xác nhận hết lỗi render 3D.
- `alpyne_bridge.py` dùng `MAPPO_JAVA_EXE` nếu đặt, nếu không ưu tiên JRE đi kèm AnyLogic 8.9. Timeout server và lock mặc định 180 giây, có thể cấu hình qua `MAPPO_SERVER_TIMEOUT`, `MAPPO_LOCK_TIMEOUT`. Chỉ tăng timeout khi bằng chứng cho thấy startup chậm.
- Log hiện dùng `alpynev11debug.log`, `modelv11debug.log` tại thư mục làm việc; `log_id` cố định `v11debug`, nên lần chạy khác có thể ghi đè/trộn log. Đối chiếu thời điểm/PID trước khi quy lỗi cho run hiện tại.
- Lượt đầu EP470 từng timeout vì log Java báo `Address already in use: bind` ở cổng chọn cố định. `build_model(..., port=0)` chuyển tới Alpyne để tự chọn cổng trống; evaluator CLI hiện yêu cầu `1 <= --port <= 65535` nên **không nhận `--port 0`**. Không dừng tiến trình Java/Python không xác định hoặc tăng timeout để chữa trùng cổng.
- 3.600 giây là thời gian mô phỏng, không phải thời gian thực. Collector/lệnh hiện chỉ in sau mỗi seed; số event khác nhau làm thời gian thực khác nhau. Có thể quan sát file kết quả, CPU và đọc `GET /status` trên server đúng PID/cổng để xác nhận thời gian mô phỏng tăng; không gửi action/reset/stop vào run người dùng đang chạy. Trạng thái `PAUSED` đơn lẻ có thể là đang đợi action RL, chưa đủ kết luận treo.

## PHẦN 2 — ĐỌC TÀI LIỆU VÀ BÀN GIAO THIẾT KẾ

### 2.1. Đầu một phiên chat mới

1. Đọc file này. Nếu người dùng chỉ định context cụ thể, đọc file đó; nếu không, liệt kê `Readme/Context/` và chọn file mới nhất theo thời điểm bàn giao ghi trong nội dung/tên file, không chỉ theo mtime.
2. Đọc **một context mới nhất** để nắm mục tiêu, quyết định, trạng thái và việc còn chờ. Chỉ mở context cha khi thiếu bằng chứng cụ thể; không nạp cả chuỗi lịch sử mỗi lần.
3. Nếu thiếu context, thông báo giới hạn và kiểm tra source/metadata thực tế. Ba tài liệu cũ về quy chuẩn chú thích, README V16-B và tổng kết V15–V16-B đã bị xóa local khi cập nhật; không tạo liên kết tới chúng như tài liệu còn tồn tại hoặc tự khôi phục từ Git. Context 24/11 vẫn dùng để truy vết quyết định lịch sử.
4. Khi làm việc kỹ thuật, kiểm tra nhánh, `git status` và đúng file/run liên quan. Giữ các thay đổi local của người dùng; không tự reset/checkout để làm sạch working tree.

### 2.2. Đọc đúng tài liệu để tiết kiệm token

- Khi chỉ được yêu cầu chú thích Python/PowerShell, giữ logic, interface và docstring có ảnh hưởng; kế thừa phong cách MODULE hiện có và kiểm tra không đổi executable logic khi cần. Thêm comment/khoảng trắng vẫn đổi SHA256 source, có thể làm mất khả năng resume/eval theo fingerprint.
- Lịch sử V15/V16/V17 và kết quả cũ: dùng các context đang tồn tại trong `Readme/Context/`, chỉ mở thêm bản cũ khi thiếu chi tiết cần thiết. Không dùng tên run/phiên bản cũ làm bằng chứng source hiện tại vẫn có file tương ứng.
- Đọc thân hàm và caller liên quan khi cần xác minh logic; tìm tên biến/hàm rồi chỉ đọc vùng cần thiết. Không suy đoán hành vi chỉ từ tên file.
- Với metrics/held-out, ưu tiên tổng hợp và kiểm tra số seed, EP, checkpoint hash; không in toàn bộ JSONL, checkpoint hoặc log dài vào chat.
- Mỗi kết quả phải có đường dẫn/bằng chứng đủ để phiên sau kiểm tra; không lặp lại nguyên văn các README vào mọi context.
- Trước khi gửi lệnh, đọc parser, kiểm tra file đầu vào tồn tại, đường dẫn metadata còn hiệu lực và hash tương ứng. Dùng mục 1.7 để phân biệt evaluator chuẩn với lệnh stdin EP470; không gửi lệnh có `--episode`/`--checkpoint` khi CLI chưa hỗ trợ. Không trộn nhiều checkpoint hoặc model vào một JSONL.

### 2.3. Bàn giao phương án thiết kế từ phiên chat khác để triển khai code

Người dùng thường thảo luận thuật toán/giải thuật với AI trong ứng dụng hoặc giao diện chat trên web. Quy trình làm việc đã thống nhất:

1. **Cung cấp nền dự án:** người dùng gửi `Readme/Skill.md` và context mới nhất cho AI ở phiên thảo luận để AI hiểu phiên bản, source, kết quả và quyết định đang có.
2. **Thảo luận và chốt:** sau khi thống nhất phương án, người dùng yêu cầu AI đó xuất một file `.md` ghi đầy đủ những gì đã chốt. File nên nêu mục tiêu, phiên bản nền/đích, thuật toán/công thức, dữ liệu vào/ra, tham số, phạm vi thay đổi, điều kiện phải giữ, cách chuyển checkpoint và tiêu chí kiểm tra khi có liên quan. Phân biệt rõ quyết định đã chốt với ý tưởng còn mở.
3. **Bàn giao cho AI code:** người dùng gửi file `.md` chốt thiết kế và yêu cầu triển khai. AI code phải đọc file này cùng Skill.md/context liên quan, đối chiếu source hiện tại, rồi tiến hành code theo phương án đã chốt. Không tự thay bằng phương án thuật toán khác hoặc mở lại toàn bộ cuộc thảo luận đã kết thúc.
4. **Xử lý điểm thiếu hoặc xung đột:** quyết định mới được người dùng chốt có thể thay thế thiết kế cũ trong Skill.md/context. Nếu có mâu thuẫn chưa rõ, thiếu thông tin làm thay đổi thuật toán, hoặc source không hỗ trợ như tài liệu giả định, nêu chính xác điểm cần làm rõ; tiếp tục các phần độc lập có đủ căn cứ. Không tự bịa quyết định thay người dùng. Các chi tiết triển khai thông thường có thể tự giải quyết trong phạm vi thiết kế.
5. **Ghi lại kết quả triển khai:** dẫn tên/đường dẫn tài liệu thiết kế trong nhật ký phiên bản và context tiếp theo; ghi phần đã code, các sai khác đã được thống nhất, kiểm tra thực sự đã chạy và việc còn chờ. Không ghi phương án là “đã triển khai” chỉ vì đã có file chốt thiết kế.

File `.md` chốt thiết kế mô tả **cần xây dựng gì**; context mô tả **dự án hiện đang ở đâu**. Khi bàn giao, giữ liên kết tới cả hai để AI sau tiếp tục chính xác. Việc nhận file thiết kế để đọc chưa tự đồng nghĩa với yêu cầu chạy training, push GitHub, PR hoặc merge; thực hiện các bước đó theo phạm vi người dùng giao.
