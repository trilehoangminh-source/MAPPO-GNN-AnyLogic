#===================== GIẢI THÍCH FILE: CONNECTOR AGV SLIPCOURSE ======================
# Giao diện action đơn giản cho mô hình minh họa; không phải đường train V16-B chính.
# BẢN ĐỌC HIỂU: chỉ bổ sung comment/khoảng trắng; KHÔNG thay đổi câu lệnh gốc.
# Lưu ý: comment vẫn làm thay đổi SHA256 của file (cần init run mới nếu dùng train).
#================================ HẾT GIỚI THIỆU FILE =================================





#========================= MODULE 01 - CÁC THƯ VIỆN / IMPORT ==========================
# import/from trong Python tương tự #include ở C, nhưng có hệ thống module và package.
from stable_baselines3 import PPO
#===================== KẾT THÚC MODULE 01 - CÁC THƯ VIỆN / IMPORT =====================




#======================= MODULE 02 - HẰNG SỐ VÀ CẤU HÌNH CHUNG ========================
# Các biến module được đọc bởi các hàm/lớp bên dưới; gần với biến toàn cục trong C.
model_path = "../../models/AgvSlipCourse/PPO/PPO-1653220759.1479478-42-100000"
model = PPO.load(model_path)
#=================== KẾT THÚC MODULE 02 - HẰNG SỐ VÀ CẤU HÌNH CHUNG ===================




#============================= MODULE 03 - HÀM GET_ACTION =============================
# Nhận observation và trả action tương thích connector của module.
def get_action(observation):
    action, _ = model.predict(observation)
    return action
#======================== KẾT THÚC MODULE 03 - HÀM GET_ACTION =========================


