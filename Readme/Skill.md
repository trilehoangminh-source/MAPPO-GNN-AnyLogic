---
name: mappo-project-context
description: Hướng dẫn tiếp nối dự án MAPPO–GNN–AnyLogic, đọc tài liệu có chọn lọc, theo dõi phiên bản và bàn giao context sau 20 tin nhắn của người dùng.
---

# HƯỚNG DẪN DỰ ÁN VÀ BÀN GIAO NGỮ CẢNH

Cập nhật: **24/11/2026**, múi giờ **Asia/Saigon**. File này là điểm vào cho AI làm việc trong dự án. Đầu phiên mới, người dùng hãy yêu cầu AI đọc `Readme/Skill.md`; không mặc định mọi ứng dụng AI tự động nạp file này chỉ nhờ tên file.

## PHẦN 1 — CẤU TRÚC, FILE CHÍNH VÀ PHIÊN BẢN DỰ ÁN

### 1.1. Gốc dự án và cây thư mục

Gốc Windows: `C:\MARL_AnyLogic\MAPPO_NEW\thesis`.
Các đường dẫn bên dưới tính từ gốc này. Thư mục `thesis/` bên trong là Python package, nên đường dẫn `thesis\thesis\mappo` trên Windows là đúng.

```text
thesis/                              # Gốc repository
├── Readme/
│   ├── Skill.md                     # File hướng dẫn hiện tại
│   ├── Context/                     # Context bàn giao của từng phiên chat
│   ├── QUY_CHUAN_CHU_THICH_CODE_PYTHON_MAPPO_V16B.md
│   ├── README_V16B_TRAFFIC_EDGE_GATV2_VI.md
│   └── TONG_KET_PHIEN_CHAT_V15_DEN_V16B_MAPPO_GNN_ANYLOGIC.md
├── scripts/                         # Khởi chạy, huấn luyện, đánh giá, vẽ biểu đồ
├── thesis/                          # Python package
│   ├── envs/                        # Observation, action, hành vi, cầu nối AnyLogic
│   ├── mappo/                       # Actor, Critic, GNN, PPO, collector
│   ├── policies/                    # Các policy và khối mạng từ nền Tobias
│   ├── pypeline_connectors/         # Các connector của dự án; giữ đúng cách viết tên
│   └── utils/                       # Công cụ cấu hình
├── envs/                            # Mô hình AnyLogic, khác thesis/envs/
│   ├── Matrix1/                     # Có Matrix1.alp và Matrix1_AGV17_trace.alp
│   ├── Matrix/, MiniMatrix/, ...    # Các mô hình khác; chỉ đọc khi cần
│   └── Matrix_RL_debug.zip          # Model được evaluator EP490 sử dụng
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
| Lệnh PowerShell V17-B | `scripts/v16b.ps1` |
| Runner và metadata V17-B | `scripts/train_v16b_fullrollout.py` |
| Routing Actor 7 edge features và message V17-B | `thesis/mappo/v16b_traffic_edge.py` |
| Evaluator V17-B | `scripts/evaluate_v16b_traffic.py` |
| Runner nền, khởi tạo/resume, kiểm tra hash | `scripts/train_v15_tobias_staged.py`, `scripts/train_v15_fullrollout.py` |
| Dựng môi trường/mạng, adapter phase | `scripts/v15_tobias_build.py`, `scripts/v15_staged_collector.py` |
| Kiến trúc nền và gradient accumulation | `thesis/mappo/v15_tobias_arch.py`, `thesis/mappo/v15_fullrollout_accum.py` |
| PPO và collector | `thesis/mappo/algorithm.py`, `thesis/mappo/mappo_stability_v7.py`, `thesis/mappo/collector_stop_ab.py` |
| Topology/đồ thị | `thesis/mappo/gnn_routing_B3_v121.py` |
| Giao tiếp AnyLogic | `thesis/envs/behaviors_v121.py`, `matrix_v121.py`, `base_alpyne_zoo_v121.py`, `alpyne121_bridge.py` trong cùng thư mục |
| Tổng hợp held-out và đồ thị | `scripts/summarize_v15_heldout.py`, `scripts/plot_v15_metrics_gui.py`, `scripts/plot_mappo_training_timesteps.py` |
| Kiểm thử có sẵn | `tests/test_v16b_traffic_edge.py`, `tests/test_v15_tobias_staged_offline.py` và test liên quan |

Tên `v15_*`/`v16b_*` không đủ xác định phiên bản đang chạy. V17-B ghi đè một số entrypoint V16-B và tái sử dụng nền V15. Đọc nội dung source, `VERSION`, metadata và nhánh Git để xác nhận; không xóa các file nền chỉ vì tên cũ. Test có tên V16-B cần được xem lại phạm vi trước khi dùng để kết luận về V17-B.

### 1.3. Phiên bản và quyết định thiết kế

| Phiên bản | Thiết kế / mốc đã xác nhận | Nhánh / run root |
|---|---|---|
| V15 | Nền Tobias: Routing GATv2, Dispatch LinFE/MLP, ba Centralized Critic; Full-Rollout GA512 | `refactor/v15-minimal`; `mappo_runs/V15_FULLROLLOUT_GA512/` |
| V16-B | Bốn edge features `[L,F,O,D]` tác động attention; bản tổng kết cũ ghi warm-start từ V15 Joint12 EP850 | `feature/v16b-traffic-edge-gatv2`; `mappo_runs/V16B_TRAFFIC_EDGE_GATV2_GA512/` |
| **V17-B hiện tại** | **Bảy edge features, edge-conditioned attention và message passing**; script dùng nguồn V15 Joint16 `initial_ep_0000.pt` | `feature/v17b-edge7-message`; `mappo_runs/V17B_EDGE7_MSG_GA512/` |

V17-B giữ ba role `routing`, `dispatch_front`, `dispatch_rear`; Routing 5 action **không action mask** là chủ đích. GAT 2 lớp, 4 heads, embedding 32; duy trì reward và PPO GA512. Bảy feature theo thứ tự trong `EDGE_FEATURE_NAMES`: chiều dài, flow cùng chiều, flow ngược chiều, áp lực next-node đích, proxy xe đứng yên `last==next`, nhu cầu goal-node đích, bottleneck topology. Các bộ đếm xe khác loại ego và lấy từ actor-visible observation; proxy đứng yên không phải số đo hàng đợi vật lý.

Mốc đã kiểm tra trong phiên ngày 24/11/2026:

- Source V17-B đã push commit `e06b4ed` lên `origin/feature/v17b-edge7-message`, gồm bốn file V17-B chính. Không tạo PR hoặc merge. Đây không phải xác nhận mọi thay đổi local đã được push.
- `state.json` của run chính được đọc ở EP583. Phải đọc lại khi cần trạng thái mới nhất.
- Người dùng đã đánh giá EP150, EP270, EP390. Phiên này chạy xong **EP490**, 20 seed `900001–900020`, greedy, 3.600 giây/seed, không PPO update; tiến trình exit code 0.
- Kết quả EP490: JobDone trung bình **13,70** (std **4,31**), Block **19,85**, LiveLock **119,90**, TargetReached **341,20**, WrongTarget **3,75**.
- File: `mappo_runs/V17B_EVAL_EP0490/joint16/heldout_joint16_20seeds_3600s.jsonl`; 20 seed duy nhất, cùng EP490 và cùng SHA256 checkpoint. Chưa phân tích so sánh với các EP khác trong phiên này.
- Các root `V17B_EVAL_EP0150`, `EP0270`, `EP0390`, `EP0490`, `EP0520` nằm dưới `mappo_runs/` với tiền tố đầy đủ `V17B_EVAL_`. Có thư mục EP0520 chưa có nghĩa đánh giá EP520 đã hoàn tất.

Repository: `https://github.com/trilehoangminh-source/MAPPO-GNN-AnyLogic`.

### 1.4. Cập nhật khi cải tiến hoặc có phiên bản mới

Sau mỗi cải tiến đã thực hiện, cập nhật bảng phiên bản hoặc thêm một dòng nhật ký ngay bên dưới: ngày, phiên bản, thay đổi chính, file liên quan, nhánh/run root, kiểm tra đã chạy và việc còn chờ. Phân biệt đề xuất, code đã triển khai, kiểm thử offline và kết quả AnyLogic thật. Không nâng đề xuất thành kết quả thực nghiệm.

Giữ các mốc V15/V16/V17 để truy vết; chi tiết dài đặt trong context hoặc tài liệu chuyên đề rồi liên kết từ đây. Chỉ ghi commit/push khi có bằng chứng. Mọi việc commit, push, PR và merge phải theo yêu cầu của người dùng; việc cập nhật tài liệu không tự cấp quyền làm các bước Git đó.

| Ngày | Phiên bản / thay đổi | Bằng chứng / trạng thái |
|---|---|---|
| 24/11/2026 | Lập hướng dẫn dự án và quy tắc bàn giao 20 tin nhắn | Tạo `Readme/Skill.md`; `Readme/Context/` đã có sẵn |
| 24/11/2026 | Thêm quy trình chốt thiết kế ở phiên chat khác rồi bàn giao cho AI code | Mục 2.7; tài liệu thiết kế do người dùng gửi là căn cứ triển khai |

## PHẦN 2 — QUY TẮC PHIÊN CHAT, ĐỌC TÀI LIỆU VÀ CONTEXT

### 2.1. Đầu một phiên chat mới

1. Đọc file này. Nếu người dùng chỉ định context cụ thể, đọc file đó; nếu không, dùng liên kết “Context mới nhất” ở cuối file. Khi liên kết chưa có hoặc không hợp lệ, liệt kê `Readme/Context/` và chọn file theo thời điểm bàn giao ghi trong nội dung/tên file, không chỉ theo mtime.
2. Đọc **một context mới nhất** để nắm mục tiêu, quyết định, trạng thái và việc còn chờ. Chỉ mở context cha khi thiếu bằng chứng cụ thể; không nạp cả chuỗi lịch sử mỗi lần.
3. Nếu chưa có context, dùng [tổng kết V15–V16-B](TONG_KET_PHIEN_CHAT_V15_DEN_V16B_MAPPO_GNN_ANYLOGIC.md) làm lịch sử; lấy mục V17-B ở phần 1 làm mốc mới hơn, rồi kiểm tra source khi bắt đầu việc kỹ thuật.
4. Khởi tạo bộ đếm **riêng của phiên mới**. Tin nhắn đầu tiên của người dùng tính là 1, kể cả tin yêu cầu đọc context. Không kế thừa số đếm của phiên trước.
5. Khi làm việc kỹ thuật, kiểm tra nhánh, `git status` và đúng file/run liên quan. Giữ các thay đổi local của người dùng; không tự reset/checkout để làm sạch working tree.

### 2.2. Đọc đúng tài liệu để tiết kiệm token

- Chú thích Python/PowerShell: đọc [quy chuẩn chú thích](QUY_CHUAN_CHU_THICH_CODE_PYTHON_MAPPO_V16B.md). Chỉ thêm comment/khoảng trắng an toàn, kiểm tra AST/token theo quy chuẩn; thêm comment vẫn đổi SHA256 source.
- Lịch sử và lý do từ V15 sang V16-B: đọc [tổng kết cũ](TONG_KET_PHIEN_CHAT_V15_DEN_V16B_MAPPO_GNN_ANYLOGIC.md).
- Cách triển khai V16-B cũ: đọc [README V16-B](README_V16B_TRAFFIC_EDGE_GATV2_VI.md); các lệnh/nguồn warm-start trong đó là lịch sử, phải đối chiếu V17-B trước khi chạy.
- Đọc thân hàm và caller liên quan khi cần xác minh logic; tìm tên biến/hàm rồi chỉ đọc vùng cần thiết. Không suy đoán hành vi chỉ từ tên file.
- Với metrics/held-out, ưu tiên tổng hợp và kiểm tra số seed, EP, checkpoint hash; không in toàn bộ JSONL, checkpoint hoặc log dài vào chat.
- Mỗi kết quả phải có đường dẫn/bằng chứng đủ để phiên sau kiểm tra; không lặp lại nguyên văn các README vào mọi context.
- Evaluator hiện chọn checkpoint qua `state.json`. Để đánh giá một EP cố định, dùng metadata đánh giá riêng và checkpoint đúng EP như `V17B_EVAL_EP0490`; giữ run huấn luyện chính. Không trộn nhiều checkpoint vào một JSONL, không bỏ qua kiểm tra hash để chạy cho được.

### 2.3. Cách đếm 20 lần chat

**Một lần chat = một tin nhắn thật của người dùng trong cùng khung chat.**

- Tính cả yêu cầu mới, câu hỏi trạng thái, đính chính và tin nhắn ngắn.
- Không tính lời trả lời của AI, commentary, tool call/output, system/developer message, snapshot trang tự chèn hoặc context kỹ thuật tự động.
- Nếu một tin nhắn có nhiều yêu cầu, vẫn tính một lần. Tin đầu phiên được tính dù AI đọc Skill.md ở thời điểm muộn hơn.
- Theo dõi số đếm trong phiên; hiển thị ngắn `Chat: n/20` trong câu trả lời cuối. Nếu lịch sử đã rút gọn, dùng mốc đếm được bàn giao; nếu không đủ dữ liệu, nói rõ số đếm chưa chắc chắn thay vì bịa.
- Việc AI được chạy tiếp sau khi chờ tool không làm tăng số đếm. Chỉ đặt lại bộ đếm khi thực sự sang khung chat mới.

### 2.4. Hành động khi đạt tin nhắn thứ 20

Ngay trong lượt xử lý tin thứ 20, **tự tạo file context, không chờ người dùng nhắc lại**:

1. Báo “Đã đạt 20 tin nhắn trong phiên này; tui lưu context để tiếp tục ở chat mới.”
2. Tạo file UTF-8 trong `Readme/Context/` theo tên `CONTEXT_YYYYMMDD_HHMMSS_V17B.md`, thay phiên bản và thời gian theo thực tế (Asia/Saigon). Không ghi đè context của phiên trước; thêm hậu tố nếu tên trùng.
3. Ghi bản tóm tắt tự đủ thông tin theo mục 2.5. Nếu task vẫn đang chạy, ghi checkpoint/tiến trình, lệnh và file kết quả, kèm trạng thái “đang chạy”; không khẳng định hoàn tất.
4. Cập nhật liên kết “Context mới nhất” ở cuối Skill.md và các mục phiên bản có thay đổi. Đọc lại file đã ghi để kiểm tra đường dẫn, trạng thái và thông tin bàn giao.
5. Gửi liên kết file context trong câu trả lời, kèm câu khởi động phiên mới: “Đọc Readme/Skill.md và context mới nhất được dẫn trong đó, rồi tiếp tục việc còn dang dở.”

Người dùng có thể yêu cầu xuất context sớm. Nếu vẫn chat tiếp sau mốc 20, giữ số đếm 21, 22… và cập nhật context của cùng phiên khi có thay đổi quan trọng, không tự coi đó là phiên mới. Tiếp tục công việc đã được giao; không buộc người dùng đóng chat. Một tiến trình còn chạy phải được kiểm tra lại ở phiên sau trước khi khởi chạy lần nữa.

### 2.5. Nội dung tối thiểu của mỗi context

Context phải ngắn, tự đủ để phiên sau làm tiếp, thông thường khoảng 800–1.500 từ và điều chỉnh theo độ phức tạp. Tóm tắt trạng thái, không chép toàn bộ hội thoại. Gồm:

- **Thông tin bàn giao:** thời gian, phiên bản, thư mục gốc, nhánh và commit đã xác nhận; số tin nhắn của phiên này, quy tắc đếm và liên kết context cha (nếu có).
- **Mục tiêu và yêu cầu người dùng:** nhiệm vụ đang làm, phạm vi đã cho phép và quyết định phải giữ.
- **Thiết kế hiện tại:** phần mới so với context trước, thông tin nền cần để hiểu độc lập; dẫn README cho phần chi tiết.
- **Việc đã hoàn tất:** file thay đổi, kiểm tra đã chạy, kết quả thực tế và bằng chứng. Ghi rõ thay đổi nào local, đã commit hoặc đã push.
- **Trạng thái vận hành:** run root, phase/EP, checkpoint/hash khi cần, seeds/port, lệnh đang chạy và nơi ghi kết quả; phân biệt trạng thái lần kiểm tra cuối với trạng thái hiện tại chưa xác minh.
- **Việc còn chờ và bước tiếp theo:** lỗi/chướng ngại, câu hỏi chưa giải quyết, hành động cụ thể để tiếp tục, những gì không nên chạy lặp lại.
- **Điểm vào đọc tiếp:** chỉ các file/hàm/tài liệu cần thiết. Nêu thông tin còn thiếu và mức độ chắc chắn; không đưa bí mật/token truy cập vào context.

Phiên A tạo context A ở mốc 20 → phiên B đọc Skill.md và context A, đếm lại từ 1 → phiên B tạo context B ở mốc 20 → phiên C đọc Skill.md và context B. Context B giữ đủ trạng thái và quyết định còn hiệu lực từ A để C không cần đọc lại toàn bộ A.

### 2.6. Chỉ mục bàn giao hiện tại

- **Context mới nhất:** chưa tạo; `Readme/Context/` đang trống ở thời điểm lập file này.
- **Mốc đếm phiên đang tạo file:** tin nhắn yêu cầu tạo `Readme/Skill.md` ngày 24/11/2026 là **9/20**. Tin nhắn thật kế tiếp trong cùng khung chat là 10/20; các tin kỹ thuật `external_codex_apps_*` tự chèn không tính.
- **Mốc công việc:** EP490 đã evaluate xong 20/20 seed; yêu cầu hiện tại là lập Skill.md. Khi tiếp tục ở phiên mới, bộ đếm bắt đầu từ 1; không dùng 9/20 làm bộ đếm cho phiên mới.

### 2.7. Bàn giao phương án thiết kế từ phiên chat khác để triển khai code

Người dùng thường thảo luận thuật toán/giải thuật với AI trong ứng dụng hoặc giao diện chat trên web. Quy trình làm việc đã thống nhất:

1. **Cung cấp nền dự án:** người dùng gửi `Readme/Skill.md` và context mới nhất cho AI ở phiên thảo luận để AI hiểu phiên bản, source, kết quả và quyết định đang có.
2. **Thảo luận và chốt:** sau khi thống nhất phương án, người dùng yêu cầu AI đó xuất một file `.md` ghi đầy đủ những gì đã chốt. File nên nêu mục tiêu, phiên bản nền/đích, thuật toán/công thức, dữ liệu vào/ra, tham số, phạm vi thay đổi, điều kiện phải giữ, cách chuyển checkpoint và tiêu chí kiểm tra khi có liên quan. Phân biệt rõ quyết định đã chốt với ý tưởng còn mở.
3. **Bàn giao cho AI code:** người dùng gửi file `.md` chốt thiết kế và yêu cầu triển khai. AI code phải đọc file này cùng Skill.md/context liên quan, đối chiếu source hiện tại, rồi tiến hành code theo phương án đã chốt. Không tự thay bằng phương án thuật toán khác hoặc mở lại toàn bộ cuộc thảo luận đã kết thúc.
4. **Xử lý điểm thiếu hoặc xung đột:** quyết định mới được người dùng chốt có thể thay thế thiết kế cũ trong Skill.md/context. Nếu có mâu thuẫn chưa rõ, thiếu thông tin làm thay đổi thuật toán, hoặc source không hỗ trợ như tài liệu giả định, nêu chính xác điểm cần làm rõ; tiếp tục các phần độc lập có đủ căn cứ. Không tự bịa quyết định thay người dùng. Các chi tiết triển khai thông thường có thể tự giải quyết trong phạm vi thiết kế.
5. **Ghi lại kết quả triển khai:** dẫn tên/đường dẫn tài liệu thiết kế trong nhật ký phiên bản và context tiếp theo; ghi phần đã code, các sai khác đã được thống nhất, kiểm tra thực sự đã chạy và việc còn chờ. Không ghi phương án là “đã triển khai” chỉ vì đã có file chốt thiết kế.

File `.md` chốt thiết kế mô tả **cần xây dựng gì**; context mô tả **dự án hiện đang ở đâu**. Khi bàn giao, giữ liên kết tới cả hai để AI sau tiếp tục chính xác. Việc nhận file thiết kế để đọc chưa tự đồng nghĩa với yêu cầu chạy training, push GitHub, PR hoặc merge; thực hiện các bước đó theo phạm vi người dùng giao.
