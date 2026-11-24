# QUY CHUẨN CHÚ THÍCH MÃ NGUỒN PYTHON — DỰ ÁN MAPPO–TRAFFIC-GATv2 / ANYLOGIC

**Phiên bản quy chuẩn:** 1.0  
**Ngày chốt:** 19/11/2026  
**Phạm vi:** Mã nguồn Python của dự án MARL–MAPPO–Traffic-Aware Edge-GATv2 (V15, V16-B và các phiên bản sau), cùng những file PowerShell điều khiển chạy chương trình nếu được gửi kèm.  
**File mẫu chuẩn:** `train_v15_tobias_staged_CHU_THICH_TIENg_VIET.py` — bản có nhãn `MODULE 00` đến `MODULE 15`, có các module con `14A`, `14H1`…

> **Nguyên tắc quan trọng nhất:** **CHỈ THÊM CHÚ THÍCH VÀ KHOẢNG TRẮNG HỢP LỆ; KHÔNG THAY ĐỔI LOGIC THỰC THI.** Mục tiêu là giúp người học hiểu chính xác mã gốc, chứ không phải tối ưu, viết lại hoặc sửa thuật toán.

## 1. Mục đích và đối tượng sử dụng

Người đọc có nền tảng **C, kỹ thuật điều khiển – tự động hóa, RL và MARL**, nhưng **yếu cú pháp Python/PyTorch**. Chú thích phải làm rõ chương trình đang thực hiện **việc gì, tại sao cần bước đó, dữ liệu vào/ra là gì, bước đó nằm ở đâu trong toàn hệ thống**.

Ưu tiên tài liệu và **mã nguồn do người dùng cung cấp**, không suy đoán chức năng chỉ dựa trên tên biến. Trước khi mô tả một hàm phức tạp, hãy đọc thân hàm và các hàm liên quan. Khi không đủ dữ liệu để xác nhận, viết rõ **“Cần kiểm tra ở file/hàm …”**, không khẳng định thay.

## 2. Hình thức phân chia module — BẮT BUỘC

### 2.1. Module cấp cao

Mỗi khối chức năng độc lập phải có **tiêu đề bắt đầu**, **chú thích giải thích**, **code gốc**, và **tiêu đề kết thúc**:

```python
#================ MODULE 02 - TÍNH SHA256 CHO FILE ====================
# Mục đích: phát hiện file nguồn hoặc checkpoint bị thay đổi.
# Đầu vào: path — đường dẫn file.
# Đầu ra: chuỗi SHA256.
# Liên hệ C: đọc từng khối byte rồi cập nhật bộ băm.
def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for c in iter(lambda:stream.read(1024*1024),b''):
            h.update(c)
    return h.hexdigest()
#=============== KẾT THÚC MODULE 02 - TÍNH SHA256 CHO FILE ================
```

- Dùng `#` với **Python**, **KHÔNG dùng `//`** (Python sẽ báo lỗi cú pháp). Với PowerShell dùng `#`.
- Tiêu đề **VIẾT HOA, BẰNG TIẾNG VIỆT**, giữ các thuật ngữ kỹ thuật cần thiết như `PPO`, `GAE`, `Actor`, `Critic`, `GATv2`, `checkpoint`.
- Đánh số **MODULE 00, MODULE 01, MODULE 02…** trong phạm vi mỗi file, theo thứ tự logic của file (thường là thứ tự xuất hiện).
- Dấu `=` kéo dài để đường phân cách dễ nhìn, gần với mẫu; độ dài không bắt buộc giống tuyệt đối.
- **Giữa hai module cấp cao: khoảng 3–5 dòng trống sạch**, không có khoảng trắng vô nghĩa trên các dòng trống.
- Tên module nói về **chức năng** (“KIỂM TRA CHECKPOINT”) thay vì chỉ nhắc tên hàm (`_checkpoint`).

### 2.2. Module con bên trong hàm lớn

Khi một hàm có nhiều giai đoạn, chia `MODULE 14A`, `14B` hoặc `14H1`, `14H2`… theo đúng thứ tự thực thi:

```python
#================ MODULE 14 - ĐIỀU PHỐI TRAINING =================
def run(argv=None):
    #========== MODULE 14A - ĐỌC THAM SỐ VÀ KIỂM TRA ==========
    a=parse_args(argv)
    cfg,blueprint,simulator,fingerprints,out,index=preflight(a)

    #========== MODULE 14B - KHỞI TẠO / RESUME ==========
    # ... giữ nguyên các lệnh gốc trong phần này ...

    #========== MODULE 14H - VÒNG LẶP EPISODE ==========
    while state['last_episode']<a.max_episodes:
        #========== MODULE 14H1 - THU THẬP TRAJECTORY ==========
        paths,meta=collector.episode(...)

        #========== MODULE 14H2 - LỌC TRAJECTORY ==========
        selected=trained_trajectories(paths,a.phase)

        #========== MODULE 14H3 - CẬP NHẬT PPO ==========
        updates=model.train(selected)

#================ KẾT THÚC MODULE 14 =================
```

**Lưu ý:** Mẫu `...` ở trên chỉ để minh họa cách chia mục. **File bàn giao phải chứa toàn bộ code gốc, tuyệt đối không được thay đoạn mã thật bằng `...`**. Chú thích nằm đúng cấp thụt lề của khối mà nó giải thích. Không chèn comment giữa `@decorator` và `def`, hoặc ở vị trí làm sai cú pháp dòng tiếp nối.

### 2.3. Module bắt đầu/kết thúc khi nào?

- Nên tạo **một module cấp cao cho mỗi hàm hoặc lớp có chức năng riêng**. Có thể gộp các hàm hỗ trợ rất ngắn liên quan mật thiết trong một module có tên rõ ràng.
- Các phần `import`, hằng số, cấu hình cũng cần module riêng.
- Với lớp `class` dài, chia module con cho khởi tạo, `forward`, tạo feature, xử lý tensor, trả kết quả.
- Với code lồng nhau, **không** chèn các dòng trống tùy tiện khiến việc đọc cấu trúc khó hơn; ưu tiên chú thích ngay trước từng bước.
- Cuối mỗi module lớn phải có dòng **`KẾT THÚC MODULE ...`** tương ứng, không thiếu và không đặt nhầm vị trí.

## 3. Nội dung chú thích tiếng Việt — PHẢI GIẢI THÍCH LOGIC

### 3.1. Thông tin tối thiểu của mỗi module

Với module có đủ ý nghĩa nghiệp vụ, diễn giải:

1. **Mục đích:** Module thực hiện việc gì trong hệ thống.
2. **Đầu vào:** Dữ liệu, tham số, hình dạng tensor nếu quan trọng.
3. **Đầu ra:** Giá trị trả về hoặc tác động (lưu file, cập nhật mạng, gọi AnyLogic…).
4. **Logic:** Các bước xử lý quan trọng theo đúng trình tự code thật.
5. **Liên hệ C / RL / MARL:** Chỉ thêm khi giúp người đọc hiểu tốt hơn.

Không cần biến mọi module ngắn thành đoạn văn dài; độ chi tiết tỷ lệ với độ phức tạp.

### 3.2. Cách diễn đạt

- Dùng **tiếng Việt trước**; thuật ngữ tiếng Anh giữ trong ngoặc hoặc dùng song song khi cần: `observation (quan sát)`, `trajectory (chuỗi chuyển trạng thái)`, `Centralized Critic`.
- Giải nghĩa cú pháp Python ít quen thuộc: `self`, `def`, `class`, `dict`, `list`, `tuple`, `with`, `yield`, `*args`, `**kwargs`, `zip`, `enumerate`, `lambda`, `None`… khi xuất hiện ở bước đáng chú ý.
- So sánh với C khi có ích: `dict` gần bảng ánh xạ theo khóa (không đồng nhất với `struct`), `class` chứa dữ liệu và phương thức, `for ... in` duyệt phần tử, indentation thay `{}`.
- Nêu kích thước tensor theo dạng `[batch, n_nodes, features]` **khi đã xác minh từ code**; giải thích `reshape`, `gather`, `scatter`, `softmax`, `detach`, `backward` theo ý nghĩa toán học nếu cần.
- Phân biệt việc **tạo dữ liệu**, **chọn action**, **ước lượng value**, **tính loss**, **cập nhật tham số**. Không gọi mọi vòng lặp là “PPO đang cập nhật”.
- Không sử dụng chú thích mơ hồ như “xử lý dữ liệu”, “chạy hàm”, “tính toán” nếu có thể nói cụ thể hơn.

### 3.3. Ví dụ chú thích sát logic MAPPO

```python
#========== MODULE 14H1 - THU THẬP DỮ LIỆU MỘT EPISODE ==========
# Chạy môi trường AnyLogic qua EventCollector.
# paths: các trajectory dùng để tính return/GAE/PPO.
# meta: thống kê môi trường (JobDone, Block, Reward...).
paths,meta=collector.episode(seed=roll_seed,train=True,max_event_steps=a.max_events)

#========== MODULE 14H2 - CHỌN ROLE ĐƯỢC HUẤN LUYỆN ==========
# Chỉ giữ trajectory thuộc những role active của phase hiện tại.
selected=trained_trajectories(paths,a.phase)

#========== MODULE 14H3 - GỌI CẬP NHẬT PPO ==========
# Lúc này mới gọi quá trình tối ưu mạng; chi tiết GAE/loss nằm trong code optimizer.
updates=model.train(selected)
```

Trong ví dụ trên, **giữ nguyên lệnh**; nếu file gốc có các kiểm tra ở giữa các lệnh, giữ đúng toàn bộ các kiểm tra đó. Không tự viết rằng một lệnh đã làm GAE nếu hàm hiện tại chỉ gọi sang hàm khác.

## 4. Độ ưu tiên khi chú thích dự án này

### 4.1. `scripts/`: luồng điều khiển chương trình

Ưu tiên giải thích: lệnh PowerShell khởi chạy → runner V16-B → wrapper Full-Rollout → runner huấn luyện V15 → dựng môi trường/mạng → EventCollector → PPO update → metrics/checkpoint/evaluation. Nói rõ **V16-B tái sử dụng code V15** ở vị trí nào; không hiểu nhầm tên `v15_...` là file thừa.

### 4.2. `thesis/envs/`: môi trường AnyLogic

Giải thích: `HiveContext`, `Matrix`, `Observation`, `Action`, các trường của AGV, hướng của cạnh, callback/sự kiện bất đồng bộ, ánh xạ AGV vật lý với role.

### 4.3. `thesis/mappo/`: phần lõi MARL

Ưu tiên chú thích kỹ:

- **Traffic-Aware Edge-GATv2:** cách xây node/edge feature, đặc trưng `length`, lưu lượng cùng chiều, ngược chiều, áp lực node đích; chiều cạnh và thứ tự `edge_index`; attention/message passing.
- **Actor–Critic / CTDE:** Actor dùng observation nào, Critic dùng global state nào; vì sao dữ liệu tập trung chỉ phục vụ huấn luyện/ước lượng value.
- **EventCollector:** cách ghi action/log-prob/value, ghép reward với lần quyết định tiếp theo của đúng AGV, xử lý `done`, alias và trajectory bất đồng bộ.
- **GAE / PPO:** reward, bootstrap, return, advantage, ValueNorm, entropy, Approx KL, Actor Loss, Value Loss, backpropagation và Adam.
- **Full-Rollout Gradient Accumulation:** phân biệt episode, rollout, batch, microbatch và optimizer step.
- **Checkpoint:** nguồn warm-start, khởi tạo tham số edge mới, resume, snapshot, hashes.

**Không tự ý thay đổi thiết kế:** Routing **không dùng action mask** là quyết định có chủ đích trong V15/V16-B; chú thích phải ghi đúng, không tự khuyên sửa mask trong code. V16-B giữ ba role `routing`, `dispatch_front`, `dispatch_rear`.

## 5. Những việc TUYỆT ĐỐI KHÔNG được làm

- Không sửa, thêm, xóa hoặc đảo thứ tự **câu lệnh chạy**.
- Không sửa biểu thức, tên biến, tham số hàm, default value, đường dẫn, seed, reward, observation, action space, action mask, mạng GNN, PPO, learning rate, giá trị cấu hình.
- Không tối ưu/refactor, không tách hàm, không thêm logging hoặc `print`, không thay `import`, không thêm code debug.
- Không thay chuỗi ký tự hoặc docstring đang có (docstring là dữ liệu có thể được truy cập bằng `__doc__`; chỉnh nó có thể thay đổi hành vi phụ).
- Không thay đổi thứ tự `import` hay vị trí `from __future__ import ...`.
- Không sửa indentation, toán tử hoặc cấu trúc khối điều khiển; **chỉ thêm comment và các dòng trống an toàn**.
- Không xóa file `.py`, `.ps1` trong ZIP nguồn nếu người dùng yêu cầu chú thích tất cả; không thay thế mã bằng đoạn tóm tắt hoặc pseudocode.
- Không đổi encoding nội dung mã; ưu tiên lưu UTF-8, giữ newline của file gốc nếu có thể.
- Nếu phát hiện lỗi code, **ghi nhận trong báo cáo**; không sửa lén trong bản chú thích.

## 6. Kiểm tra chất lượng trước khi bàn giao

**Bắt buộc kiểm tra cho từng file `.py`:**

1. File gốc và file sau chú thích đều đọc được, không bị mất phần mã.
2. Python **parse/compile** thành công (ví dụ `ast.parse(...)` và `compile(...)` để tránh phát sinh `.pyc` trong thư mục nguồn).
3. So sánh **AST** trước/sau (`ast.dump(..., include_attributes=False)`): phải giống nhau.
4. So sánh **token mang ý nghĩa thực thi** trước/sau (bỏ comment/những token dòng trống): tên biến, giá trị, toán tử, chuỗi, thứ tự lệnh không đổi.
5. Mọi module có nhãn mở/đóng phù hợp; chú thích không lạc indentation.
6. Với PowerShell, xác nhận script vẫn giữ nguyên lệnh, chỉ thêm `#` ở vị trí comment hợp lệ; không tự tuyên bố đã kiểm thử thực thi nếu chưa thực sự chạy.
7. Không khẳng định “đã chạy AnyLogic” chỉ vì compile hoặc unit test đạt; **phân biệt kiểm tra cú pháp với kiểm thử tích hợp**.

**Báo cáo kèm ZIP** nên ghi số file, file giữ nguyên, file được chú thích, kết quả AST/token, các trường hợp không kiểm tra được, cách giải nén.

## 7. Quy cách xuất file và ZIP

Khi người dùng gửi cả thư mục:

- **Giữ nguyên cây thư mục tương đối** của ZIP nguồn. Nếu có `scripts/...`, trả lại `scripts/...`; nếu có `thesis/mappo/...`, trả lại `thesis/mappo/...`.
- Đóng ZIP **để giải nén trực tiếp vào thư mục gốc dự án** `C:\MARL_AnyLogic\MAPPO_NEW\thesis` khi người dùng yêu cầu.
- Không tạo thừa tầng lồng nhau kiểu `thesis/thesis/thesis/...`.
- Không đóng `__pycache__`, `.pyc`, file tạm hoặc `.bak` vào bản chú thích mới, trừ khi người dùng yêu cầu; các file này không phải code cần giải thích.
- Có thể kèm `HUONG_DAN_DOC_CODE.md` ở cấp gốc ZIP, ghi thứ tự file nên học.
- Đặt tên ZIP dễ hiểu: `SCRIPTS_CHU_THICH_TIENG_VIET.zip`, `THESIS_CHU_THICH_TIENG_VIET.zip` hoặc tương tự.
- Khi người dùng yêu cầu **ghi đè source**, giữ đúng tên file và đường dẫn trong ZIP; khi yêu cầu **bản đọc riêng**, tạo thư mục `_CHU_THICH` riêng.

## 8. Lưu ý đặc biệt: SHA256, checkpoint và resume training

Dự án V15/V16-B có cơ chế so hash **mã nguồn/checkpoint** để bảo vệ tính tái lập. **Thêm comment cũng làm đổi SHA256**, dù AST vẫn giống. Vì vậy:

- Nếu có run đang train hoặc cần resume: **không ghi đè source gốc**; sử dụng bản chú thích riêng để đọc.
- Nếu người dùng **chấp nhận train lại** từ EP0: có thể thay bản source đã chú thích vào đúng chỗ, nhưng phải **dừng training trước, backup thư mục `mappo_runs` hiện tại, và khởi tạo run mới**. Không xóa checkpoint V15 dùng để warm-start.
- Việc train lại V16-B từ EP0 theo cấu hình hiện hành **không đồng nghĩa train từ trọng số ngẫu nhiên**: có thể warm-start từ checkpoint V15, ví dụ Joint12 EP850. Kiểm tra chính xác nguồn trong config/log trước khi phát biểu.
- Không tự sửa hoặc vô hiệu hóa cơ chế hash để “cho resume được” nếu không có yêu cầu rõ ràng.

## 9. Mẫu yêu cầu dùng ở phiên chat mới (sao chép nguyên đoạn này)

> Tôi gửi file/folder source của dự án MAPPO–Traffic-GATv2–AnyLogic. Hãy **chỉ thêm chú thích tiếng Việt theo QUY CHUẨN CHÚ THÍCH MÃ NGUỒN**: chia mỗi file thành các khối `#================ MODULE 00 - TÊN ================` và `#================ KẾT THÚC MODULE 00 ================`; trong hàm lớn chia `MODULE 14A`, `14H1`…; giữa các module cấp cao có 3–5 dòng trống; giải thích **mục đích, đầu vào/đầu ra, logic, ý nghĩa RL/MARL và liên hệ C/Python**. Giữ nguyên tuyệt đối code thực thi, tên biến, thứ tự lệnh, thuật toán và cấu trúc thư mục. Kiểm tra AST/token trước/sau, báo cáo kết quả và đóng ZIP giải nén trực tiếp vào `C:\MARL_AnyLogic\MAPPO_NEW\thesis` nếu tôi yêu cầu. **Không sửa code**, không refactor, không tự thêm chức năng. Nếu file nguồn đang được dùng để resume training, cảnh báo thay đổi SHA256.

## 10. Tiêu chí nghiệm thu cuối

Một file chú thích **đạt chuẩn** khi người đọc có thể nhìn vào tiêu đề module để biết nhiệm vụ của khối, đọc 2–5 câu comment để hiểu dữ liệu và vị trí trong thuật toán, rồi lần theo **đúng toàn bộ câu lệnh gốc**. File đó phải parse/compile được, AST và token thực thi không đổi, không phá môi trường huấn luyện. Với ZIP, giải nén phải cho đúng cấu trúc thư mục đã yêu cầu.

---

**Quy ước ưu tiên:** Trong các phiên chat sau, nếu có xung đột giữa cách chú thích chung và file mẫu người dùng gửi, **ưu tiên file mẫu người dùng gửi**, đồng thời tuân thủ nguyên tắc không đổi logic và thông báo rõ mọi giới hạn kiểm thử.
