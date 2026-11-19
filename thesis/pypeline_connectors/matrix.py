#======================= GIẢI THÍCH FILE: CONNECTOR MATRIX GỐC ========================
# Nhận cấu hình, áp policy hoặc logic được cấu hình vào môi trường Matrix.
# BẢN ĐỌC HIỂU: chỉ bổ sung comment/khoảng trắng; KHÔNG thay đổi câu lệnh gốc.
# Lưu ý: comment vẫn làm thay đổi SHA256 của file (cần init run mới nếu dùng train).
#================================ HẾT GIỚI THIỆU FILE =================================





#========================= MODULE 01 - CÁC THƯ VIỆN / IMPORT ==========================
# import/from trong Python tương tự #include ở C, nhưng có hệ thống module và package.
import json
import ray.rllib.algorithms.ppo as ppo
import ray.rllib.algorithms.dqn as dqn
from ..utils.double_trainer import DoubleTrainer, TripleTrainer
from alpyne.data.spaces import Observation
from ..utils.utils import get_config, setup_ray
#===================== KẾT THÚC MODULE 01 - CÁC THƯ VIỆN / IMPORT =====================



### SETUP ###
setup_ray(port=51160, unidirectional=True)



#======================= MODULE 02 - HẰNG SỐ VÀ CẤU HÌNH CHUNG ========================
# Các biến module được đọc bởi các hàm/lớp bên dưới; gần với biến toàn cục trong C.
pseudo = True
model_path = "../../models/matrix_routing/09_rew_block_-20_8_20_2023-05-01_22-56-02"#minimatrix_dispatching/06_mat_rout__4_30_2023-01-13_18-43-09"
checkpoint = 400
trainer_cls = dqn.DQN
fleetsize = 16#None
### END SETUP ###


checkpoint_path = model_path + f"/checkpoint_{str(checkpoint).rjust(6, '0')}"
path = "C:/Users/Wegma/Documents/Masterarbeit/thesis/"
with open(model_path + "/config.json") as json_file:
    hparams = json.load(json_file)
hparams["n_envs"] = 1
hparams["run_class"] = "pypeline"
if fleetsize is not None:
    hparams["env_args"]["fleetsize"] = fleetsize
    hparams["env_args"]["fleetsize_upper"] = None
if pseudo:
    hparams["env_args"]["max_seconds"] = 10
    hparams["env_args"]["pseudo_dispatcher"] = True
    hparams["env_args"]["pseudo_dispatcher_clever"] = True
    #hparams["env_args"]["pseudo_dispatcher_rule"] = "shortest_quene"
    hparams["env_args"]["pseudo_routing"] = True
    
config, logger_creator, checkpoint_dir = get_config(path, **hparams)
config["num_gpus"] = 1
trainer = trainer_cls(config, logger_creator=logger_creator)
if not pseudo:
    trainer.restore(checkpoint_path)

env = trainer.workers.local_worker().env.env
if pseudo:  
    env.reset(options = {"dont_collect": True})
env_config = env.config
#=================== KẾT THÚC MODULE 02 - HẰNG SỐ VÀ CẤU HÌNH CHUNG ===================




#============================= MODULE 03 - HÀM GET_CONFIG =============================
# Trả cấu hình thí nghiệm/mô phỏng cho connector.
def get_config(id: str):
    if id in env_config.names():
        return env_config.values()[env_config.names().index(id)]
    return "NOT FOUND"
#======================== KẾT THÚC MODULE 03 - HÀM GET_CONFIG =========================




#============================= MODULE 04 - HÀM GET_ACTION =============================
# Nhận observation và trả action tương thích connector của module.
def get_action(observation, caller, n_nodes, context, statTitles, statValues, time):
    alpyne_obs = Observation(
        obs=observation, caller=caller, n_nodes=n_nodes, networkcontext=context, rew=0, statTitles = statTitles, statValues = statValues
    )
    agent = env._agent_selection_fn(alpyne_obs)
    if env.agent_behaviors[agent].is_for_learning(alpyne_obs):
        env._save_observation(
            alpyne_obs,
            agent,
            *env.agent_behaviors[agent].convert_from_observation(alpyne_obs),
        )
        print(agent)
        print(config["multiagent"]["policy_mapping_fn"](agent, None, None))
        action = trainer.compute_single_action(env.last()[0], explore=False, policy_id=config["multiagent"]["policy_mapping_fn"](agent, None, None))

        return env.agent_behaviors[env.agent_selection].convert_to_action(
            action, caller
        )

    else:
        return env.agent_behaviors[agent].get_action(alpyne_obs, time=time)
#======================== KẾT THÚC MODULE 04 - HÀM GET_ACTION =========================


