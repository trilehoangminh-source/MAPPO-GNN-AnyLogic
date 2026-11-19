#=============== GIẢI THÍCH FILE: MÔI TRƯỜNG EVENT-DRIVEN / PETTINGZOO ================
# Chuyển giao lượt điều khiển giữa agent, nhận observation, chuyển action xuống AnyLogic và lưu reward.
# BẢN ĐỌC HIỂU: chỉ bổ sung comment/khoảng trắng; KHÔNG thay đổi câu lệnh gốc.
# Lưu ý: comment vẫn làm thay đổi SHA256 của file (cần init run mới nếu dùng train).
#================================ HẾT GIỚI THIỆU FILE =================================





#========================= MODULE 01 - CÁC THƯ VIỆN / IMPORT ==========================
# import/from trong Python tương tự #include ở C, nhưng có hệ thống module và package.
from abc import abstractmethod
from typing import List
from typing import Union, Tuple, Dict, Optional

import numpy as np
from gymnasium import spaces

from .alpyne121_bridge import ModelRun
from .alpyne121_bridge import Observation, Action, Configuration

from pettingzoo import AECEnv
#===================== KẾT THÚC MODULE 01 - CÁC THƯ VIỆN / IMPORT =====================



# ZooAgentBehavior là cái khuôn mô tả behavior của một agent 



#========================== MODULE 02 - LỚP ZOOAGENTBEHAVIOR ==========================
# Giao diện hành vi của từng agent, phân biệt học RL và chạy chính sách có sẵn.
class ZooAgentBehavior:
    """
    Baseclass to describe the behavior of an agent. Any custom behavior should inherit from this. 
    A BaseAlpyneZoo-environment has one of these behaviors for every agent. It can be used for learning agents, agents that just use some static behavior or anything in between. 
    """


    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self) -> None:
        pass
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #======================== MODULE CON 02 - IS_FOR_LEARNING =========================
    # Mục đích: Cho biết agent có được huấn luyện bằng RL không.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def is_for_learning(self, alpyne_obs) -> bool:   #agent này có phải là learning agent ko ? 
        """
        Will be called when the corresponding agent has to act. 
        If True: Delegate to gym-inteface for learning
        If False: Choose the action by get_action and go on with the simulation. 
        """
        return True
    #==================== KẾT THÚC MODULE CON 02 - IS_FOR_LEARNING ====================




    #=========================== MODULE CON 03 - GET_ACTION ===========================
    # Mục đích: Nhận observation và trả action tương thích connector của module.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    @abstractmethod
    def get_action(self, alpyne_obs, time):
        """
        Calculate the action absed on the alpyne observation object and the current time.
        """
        raise NotImplementedError()
    #====================== KẾT THÚC MODULE CON 03 - GET_ACTION =======================




    #===================== MODULE CON 04 - GET_OBSERVATION_SPACE ======================
    # Mục đích: Trả mô tả kích thước/miền giá trị observation.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    @abstractmethod
    def get_observation_space(self) -> spaces.Space:
        """
        Describe the agent's observation space
        """
        raise NotImplementedError()
    #================= KẾT THÚC MODULE CON 04 - GET_OBSERVATION_SPACE =================




    #======================== MODULE CON 05 - GET_ACTION_SPACE ========================
    # Mục đích: Trả mô tả tập hành động của agent.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    @abstractmethod
    def get_action_space(self) -> spaces.Space:
        """
        Describe the agent's action space
        """
        raise NotImplementedError()
    #=================== KẾT THÚC MODULE CON 05 - GET_ACTION_SPACE ====================




    #======================= MODULE CON 06 - CONVERT_TO_ACTION ========================
    # Mục đích: Đóng gói action RL sang packet gửi simulator AnyLogic.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    @abstractmethod
    def convert_to_action(
        self, action: "BaseAlpyneZoo.PyActionType", agent: str
    ) -> Action:
        """
        Convert the action from the learning algorithm / get_action to an alypne action object
        """
        raise NotImplementedError()
    #=================== KẾT THÚC MODULE CON 06 - CONVERT_TO_ACTION ===================




    #==================== MODULE CON 07 - CONVERT_FROM_OBSERVATION ====================
    # Mục đích: Biến đổi packet AnyLogic thành observation/reward theo giao diện RL.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    @abstractmethod
    def convert_from_observation(self, alpyne_obs):
        """
        Convert the alpyne observation object to an observation compatible with get_observation_space
        """
        raise NotImplementedError()
    #=============== KẾT THÚC MODULE CON 07 - CONVERT_FROM_OBSERVATION ================

#===================== KẾT THÚC MODULE 02 - LỚP ZOOAGENTBEHAVIOR ======================



# Một Agent cụ thể được gắn với một Behavior cụ thể.



#============================== MODULE 03 - LỚP ZOOAGENT ==============================
# Ghép tên agent vật lý và đối tượng xử lý hành vi.
class ZooAgent:
    """
    
    """


    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(
        self, name: str, behavior: ZooAgentBehavior, does_training: bool = True
    ) -> None:
        self.name = name
        self.behavior = behavior
        self.does_training = does_training
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================

#========================= KẾT THÚC MODULE 03 - LỚP ZOOAGENT ==========================


# # ZooAgentBehavior
#        ↓
# "Quy định một Agent Behavior phải làm được gì"

# ZooAgent
#        ↓
# "Một Agent cụ thể + Behavior của nó"



# Anylogic -> chờ simulation xong -> lấy observation -> xác định agent nào tới lượt -> nếu agent đó là learning agent thì trả observation về cho gym interface, nếu không thì tính action và gửi action về simulation.
# AECEnv là Environment API của PettingZoo.



#=========================== MODULE 04 - LỚP BASEALPYNEZOO ============================
# Adapter vòng đời PettingZoo AEC giữa Python và AnyLogic; môi trường là event-driven, không đồng bộ theo tick.
class BaseAlpyneZoo(AECEnv):
    """
    An abstract PettingZoo environment.

    """

    # The possible types that the gym Space objects can represent.
    PyObservationType = PyActionType = Union[
        np.ndarray,
        int,
        float,
        Tuple[np.ndarray, int, float, tuple, dict],
        Dict[str, Union[np.ndarray, int, float, tuple, dict]],
    ]



    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, agent_behaviors = None):
        """
        Construct a new environment for the provided sim.

        Note that the configuration passed as part of its creation is what will be used for all episodes.
        If it's desired to have this be changed, the configuration values
        should be assigned to callable values or tuples consisting of (start, stop, step).
        See the `ModelRun` or `Configuration` class documentation for more information.

        :param sim: a created - but not yet started - instance of your model
        :raise ValueError: if the run has been started
        """
        self._initialize_cache(agents=agent_behaviors)      #Chuẩn bị bộ nhớ để quản lý các Agent
        self.agent_selection = None  #hiện tại chưa có Agent nào được chọn.
        self.observable = False  #hiện tại Environment chưa có Observation mới để cung cấp.
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #====================== MODULE CON 02 - _AGENT_SELECTION_FN =======================
    # Mục đích: Hàm nội bộ agent selection fn; được gọi bởi các hàm cấp cao hơn.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    @abstractmethod
    def _agent_selection_fn(self, alpyne_obs):
        raise NotImplementedError
    #================== KẾT THÚC MODULE CON 02 - _AGENT_SELECTION_FN ==================

    #Observation hiện tại thuộc về Agent nào / Agent nào đang tới lượt?



    #=========================== MODULE CON 03 - GET_AGENTS ===========================
    # Mục đích: Truy xuất agents từ trạng thái/dữ liệu đã lưu.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    @abstractmethod
    def get_agents(self) -> List[ZooAgent]:
        raise NotImplementedError
    #====================== KẾT THÚC MODULE CON 03 - GET_AGENTS =======================

    # Environment này có những Agent nào?

    # Đưa simulation về trạng thái bắt đầu của một Episode.


    #============================= MODULE CON 04 - RESET ==============================
    # Mục đích: Đưa trạng thái hoạt động về đầu episode/thời điểm khởi tạo.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def reset(
        self,
        config: Configuration = None,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
        agents=None,
        dont_collect = False,
    ) -> "BaseAlpyneZoo.PyObservationType":
        """
        A method required as part of the pettingzoo interface to revert the sim to the start.

        :param config: The config to start the run with. If None, the sim will use the same configuration object as it was created with.
        """
        if config is not None:
            config.seed = seed if seed is not None else config.seed
            if options is not None:
                for option in options.keys():
                    if option in config.names():
                        config.__setattr__(option, options[option])
        self.agents.clear()
        self._initialize_cache()
        self.sim.wait_for_completion()
        self.sim.reset(config)
        self.observable = False
        if not dont_collect:
            self._collect()
            if return_info:
                return self.observe(self.agent_selection)
    #========================= KẾT THÚC MODULE CON 04 - RESET =========================




    #============================== MODULE CON 05 - STEP ==============================
    # Mục đích: Gửi action và chuyển mô phỏng tới sự kiện tiếp theo.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def step(
        self, action: "BaseAlpyneZoo.PyActionType"
    ) -> Tuple["BaseAlpyneZoo.PyObservationType", float, bool, Optional[dict]]:
        """
        A method required as part of the pettingzoo interface to run one step of the sim.
        Take an action in the sim and advance the sim to the start of the next step.

        :param action: The action to send to the sim (in the type expressed by your action space)
        """
        alpyne_action = self.agent_behaviors[self.agent_selection].convert_to_action(
            action, self.agent_selection
        )
        self.sim.take_action(alpyne_action)
        self._collect()
    #========================= KẾT THÚC MODULE CON 05 - STEP ==========================



    # Lấy thông tin của Agent hiện đang tới lượt hành động.


    #============================== MODULE CON 06 - LAST ==============================
    # Mục đích: Xử lý last theo logic của lớp BaseAlpyneZoo.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def last(self, observe=True) -> "BaseAlpyneZoo.PyObservationType":
        """
        A method required as part of the pettingzoo interface to gather observation, reward, done and info for the agent that will act next.

        :return: Observation, reward, done and info for the agent that will act next
        """
        if not self.observable:
            self._collect()
        obs = self.observations[self.agent_selection] if observe else None
        reward = self.rewards[self.agent_selection]
        done = self.dones[self.agent_selection]
        info = self.infos[self.agent_selection]
        if all([self.dones[agent] for agent in self.agents]):   #Điều khiện để kết thúc Episode
            # All terminate at once
            self.agents.clear()
        return obs, reward, done, info
    #========================= KẾT THÚC MODULE CON 06 - LAST ==========================



    # Lấy Observation của một Agent cụ thể.


    #============================ MODULE CON 07 - OBSERVE =============================
    # Mục đích: Xử lý observe theo logic của lớp BaseAlpyneZoo.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def observe(self, agent: str) -> "BaseAlpyneZoo.PyObservationType":
        """Return the observation that agent currently can make."""
        if not self.observable:
            self._collect()
        return self.observations[agent]
    #======================== KẾT THÚC MODULE CON 07 - OBSERVE ========================




    #======================= MODULE CON 08 - OBSERVATION_SPACE ========================
    # Mục đích: Xử lý observation space theo logic của lớp BaseAlpyneZoo.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def observation_space(self, agent: str) -> spaces.Space:
        return self.agent_behaviors[agent].get_observation_space()
    #=================== KẾT THÚC MODULE CON 08 - OBSERVATION_SPACE ===================




    #========================== MODULE CON 09 - ACTION_SPACE ==========================
    # Mục đích: Xử lý action space theo logic của lớp BaseAlpyneZoo.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def action_space(self, agent) -> spaces.Space:
        return self.agent_behaviors[agent].get_action_space()
    #===================== KẾT THÚC MODULE CON 09 - ACTION_SPACE ======================




    #========================== MODULE CON 10 - REMOVE_AGENT ==========================
    # Mục đích: Xử lý remove agent theo logic của lớp BaseAlpyneZoo.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def remove_agent(self, agent):
        self.agents.remove(agent)
    #===================== KẾT THÚC MODULE CON 10 - REMOVE_AGENT ======================




    #=========================== MODULE CON 11 - ADD_AGENT ============================
    # Mục đích: Xử lý add agent theo logic của lớp BaseAlpyneZoo.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def add_agent(self, agent: ZooAgent):
        if not agent.name in self.agents:
            self.agents.append(agent.name)
            self.dones[agent.name] = False
            self.infos[agent.name] = dict()
            self.rewards[agent.name] = 0
            self.observations[agent.name] = None
            self.agent_behaviors[agent.name] = agent.behavior
    #======================= KẾT THÚC MODULE CON 11 - ADD_AGENT =======================



    # Nó liên tục tìm kiếm trạng thái tiếp theo mà Environment cần cung cấp cho learning agent.


    #============================ MODULE CON 12 - _COLLECT ============================
    # Mục đích: Hàm nội bộ collect; được gọi bởi các hàm cấp cao hơn.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def _collect(self):
        while True:
            self.sim.wait_for_completion()
            alpyne_obs = self.sim.get_observation() #lấy observation từ Alpyne 
            agent = self._agent_selection_fn(alpyne_obs) #observation này thuộc agent nào? → xác định agent nào tới lượt
            if self.agent_behaviors[agent].is_for_learning(alpyne_obs):
                self._save_observation(
                    alpyne_obs,
                    agent,
                    *self.agent_behaviors[agent].convert_from_observation(alpyne_obs)
                )
                break
            else:
                action = self.agent_behaviors[agent].get_action(
                    alpyne_obs, time=self.run.get_state()[1]["model_time"]
                )
                self.sim.take_action(action)

        if self.sim.is_terminal() or self._terminal_alternative(alpyne_obs):
            [self.dones.update({agent: True}) for agent in self.agents]
        self.observable = True
    #======================= KẾT THÚC MODULE CON 12 - _COLLECT ========================



    #bộ nhớ tạm của Environment cho bước hiện tại.


    #======================= MODULE CON 13 - _SAVE_OBSERVATION ========================
    # Mục đích: Hàm nội bộ save observation; được gọi bởi các hàm cấp cao hơn.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def _save_observation(self, alpyne_obs, agent, obs, reward, done, info):
        self.agent_selection = agent   #Ghi nhận: hiện tại đang tới lượt Agent này.
        self.observations[agent] = obs #Cất Observation vào đúng ngăn của Agent 2.
        self.rewards[agent] = reward    #Cất Reward của Agent đó.
        self.dones[agent] = done    #Agent này đã kết thúc chưa?
        self.infos[agent] = info #info = thông tin bổ sung, không phải phần chính của RL.
    #=================== KẾT THÚC MODULE CON 13 - _SAVE_OBSERVATION ===================




    #===================== MODULE CON 14 - _TERMINAL_ALTERNATIVE ======================
    # Mục đích: Hàm nội bộ terminal alternative; được gọi bởi các hàm cấp cao hơn.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def _terminal_alternative(self, observation: Observation) -> bool:
        """Optional method to add *extra* terminating conditions"""
        return False
    #================= KẾT THÚC MODULE CON 14 - _TERMINAL_ALTERNATIVE =================




    #======================= MODULE CON 15 - _INITIALIZE_CACHE ========================
    # Mục đích: Hàm nội bộ initialize cache; được gọi bởi các hàm cấp cao hơn.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def _initialize_cache(self, agents: List[ZooAgent] = None):
        if agents is None:
            agents = self.get_agents()
        self.agents = [agent.name for agent in agents if agent.does_training]
        self.dones = {agent: False for agent in self.agents}
        self.infos = {agent: dict() for agent in self.agents}
        self.rewards = {agent: 0 for agent in self.agents}
        self.observations = {agent: None for agent in self.agents}
        self.agent_behaviors = {agent.name: agent.behavior for agent in agents}
    #=================== KẾT THÚC MODULE CON 15 - _INITIALIZE_CACHE ===================




    #============================= MODULE CON 16 - _START =============================
    # Mục đích: Hàm nội bộ start; được gọi bởi các hàm cấp cao hơn.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def _start(self, sim: ModelRun):
        # complain if the sim was already started
        if sim.id:
            raise ValueError("The provided model run should not have been started!")

        self.sim = sim.run()  # submit new run
        self.sim.wait_for_completion()  # wait until start is finished setting up
    #======================== KẾT THÚC MODULE CON 16 - _START =========================

#======================= KẾT THÚC MODULE 04 - LỚP BASEALPYNEZOO =======================


