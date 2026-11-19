#===================== GIẢI THÍCH FILE: CONNECTOR HIỂN THỊ MATRIX =====================
# Hỗ trợ chọn hành động cho mô hình minh họa/visualization.
# BẢN ĐỌC HIỂU: chỉ bổ sung comment/khoảng trắng; KHÔNG thay đổi câu lệnh gốc.
# Lưu ý: comment vẫn làm thay đổi SHA256 của file (cần init run mới nếu dùng train).
#================================ HẾT GIỚI THIỆU FILE =================================




#========================= MODULE 01 - CÁC THƯ VIỆN / IMPORT ==========================
# import/from trong Python tương tự #include ở C, nhưng có hệ thống module và package.
from alpyne.data.spaces import Observation
from ..envs.matrix import Matrix
#===================== KẾT THÚC MODULE 01 - CÁC THƯ VIỆN / IMPORT =====================



# ===== CAU HINH MO PHONG =====




#======================= MODULE 02 - HẰNG SỐ VÀ CẤU HÌNH CHUNG ========================
# Các biến module được đọc bởi các hàm/lớp bên dưới; gần với biến toàn cục trong C.
FLEETSIZE = 2

env = Matrix(
    model_path=None,
    fleetsize=FLEETSIZE,
    max_fleetsize=20,

    # Su dung Behavior co san cua Tobias
    pseudo_routing=True,
    pseudo_dispatcher=True,
    pseudo_dispatcher_clever=True,
    pseudo_dispatcher_rule="shortest_quene",

    sim_config={
        "dispatch": True,
        "routing_ma": True,
        "dispatching_ma": True,

        "routing_interval": 2,
        "dispatching_interval": 120,
        "block_timeout": 120,

        "fixed_fleets": False,
        "remove_all_blocked": False,

        "availability": 0.9,
        "mttr": 5,
        "io_quote": 0.9,
    },
)

env_config = env.config
#=================== KẾT THÚC MODULE 02 - HẰNG SỐ VÀ CẤU HÌNH CHUNG ===================




# ===== GUI CAU HINH CHO ANYLOGIC =====




#============================= MODULE 03 - HÀM GET_CONFIG =============================
# Trả cấu hình thí nghiệm/mô phỏng cho connector.
def get_config(name: str):

    # Tuong thich ten tham so trong Main cua Tobias
    if name == "obsrout_include_part_info":
        name = "obs_include_part_info"

    if name in env_config.names():
        return env_config.values()[
            env_config.names().index(name)
        ]

    raise KeyError(f"Unknown configuration: {name}")
#======================== KẾT THÚC MODULE 03 - HÀM GET_CONFIG =========================




# ===== PYTHON TINH ACTION CHO AGV =====




#============================= MODULE 04 - HÀM GET_ACTION =============================
# Nhận observation và trả action tương thích connector của module.
def get_action(
    observation,
    caller,
    n_nodes,
    context,
    statTitles,
    statValues,
    time
):
    
    print("PYTHON DANG TINH ACTION - TIME:", time, flush=True)
    with open(r"C:\MARL_AnyLogic\pypeline_debug.txt", "a") as f:
        f.write(f"Python get_action called at time = {time}\n")

    alpyne_obs = Observation(
        obs=observation,
        caller=caller,
        n_nodes=n_nodes,
        networkcontext=context,
        rew=0,
        statTitles=statTitles,
        statValues=statValues,
    )

    # Tim Agent dang yeu cau quyet dinh
    agent = env._agent_selection_fn(alpyne_obs)

    # Lay Behavior cua Agent
    behavior = env.agent_behaviors[agent]

    # Cap nhat context (mang luoi, vi tri...)
    if behavior.is_for_learning(alpyne_obs):
        raise RuntimeError(
            f"Agent {agent} requires an RL policy"
        )

    # Tinh Action theo thuat toan cua Tobias
    return behavior.get_action(alpyne_obs, time=time)
#======================== KẾT THÚC MODULE 04 - HÀM GET_ACTION =========================


