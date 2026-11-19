#======================== GIẢI THÍCH FILE: GÓI THƯ VIỆN MAPPO =========================
# Đánh dấu thư mục mappo là Python package để các file khác import.
# BẢN ĐỌC HIỂU: chỉ bổ sung comment/khoảng trắng; KHÔNG thay đổi câu lệnh gốc.
# Lưu ý: comment vẫn làm thay đổi SHA256 của file (cần init run mới nếu dùng train).
#================================ HẾT GIỚI THIỆU FILE =================================


"""Event-driven MAPPO implementation for Tobias's Matrix/AnyLogic simulation."""



#========================= MODULE 01 - CÁC THƯ VIỆN / IMPORT ==========================
# import/from trong Python tương tự #include ở C, nhưng có hệ thống module và package.
from .algorithm import MAPPO, MAPPOConfig
from .collector import EventCollector
#===================== KẾT THÚC MODULE 01 - CÁC THƯ VIỆN / IMPORT =====================


