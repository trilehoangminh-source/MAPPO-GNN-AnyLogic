import random
from .alpyne121_bridge import AlpyneClient
from .base_alpyne_zoo_v121 import ZooAgent, BaseAlpyneZoo
from .behaviors_v121 import (
    MultiAgent,
    SingleAgent,
    RandomStationDispatcher,
    HiveContext,
    CollisionFreeRouting,
    CollisionFreeRoutingHive,
    Dummy,
    MultiDispAgent,
    CleverMatrixDispatcher,
    ApplyPolicyAgent,
    SingleDispAgent
)
from ..utils.build_config_v121 import build_config
from PIL import ImageDraw, Image
import gymnasium.spaces


class Matrix(BaseAlpyneZoo):

    def __init__(
        self,
        model_path: str = None,
        startport=51150,
        fleetsize: int = 1,
        fleetsize_upper = None,
        max_fleetsize: int = 1,
        sim_config: dict = dict(),
        max_steps: int = None,
        max_seconds: int = None,
        pseudo_dispatcher=True,
        pseudo_dispatcher_clever=False,
        pseudo_dispatcher_rule="random",
        pseudo_dispatcher_distance=None,
        pseudo_routing=False,
        routing_agent_death=False,
        death_on_target=False,
        dispatching_agent_death=False,
        transform_dispatching_partobs = False,
        routing_policy:str = None,
        dispatching_policy:str = None,
        direction_reward = 0,
        client = None,
        warmup_time = None,
    ):
        self.fleetsize =self.fleetsize_under =fleetsize   #fleetsize = số AGV mà Environment muốn mô phỏng.
        self.fleetsize_upper = fleetsize_upper
        self.max_fleetsize = max_fleetsize #→ giới hạn tối đa mà các Behavior/Observation được chuẩn bị để xử lý
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.stepcounter = 0
        self.routing_policy = routing_policy
        self.dispatching_policy = dispatching_policy
        self.direction_reward = direction_reward
        self.warmup_time = warmup_time if warmup_time is not None else 0
        self.warmup_statistics = None

        self.metadata = dict(is_parallelizable=True)
        self.statistics = None
        self.dispatch = sim_config["dispatch"]
        self.pseudo_dispatcher = pseudo_dispatcher      #Một cơ chế thay thế/giả lập quyết định dispatching bằng logic có sẵn, thay vì learning policy.
        self.pseudo_routing = pseudo_routing            #Có dùng routing logic có sẵn thay vì routing policy học được hay không.
        self.pseudo_disp_cls = CleverMatrixDispatcher if pseudo_dispatcher_clever else RandomStationDispatcher  #lựa chọn class làm dispater_pseudo
        self.pseudo_dispatcher_rule = pseudo_dispatcher_rule
        self.pseudo_dispatcher_distance = pseudo_dispatcher_distance
        self.routing_ma = sim_config["routing_ma"]
        self.dispatching_ma = sim_config["dispatching_ma"]
        self.all_ma = self.routing_ma and (self.dispatching_ma or not self.dispatch)  
        self.routing_agent_death = routing_agent_death #Khi một Agent hoàn thành nhiệm vụ/đạt target hoặc rời khỏi hệ thống, Environment có loại Agent đó khỏi danh sách đang hoạt động rồi tạo alias/Agent thay thế hay không.
        self.death_on_target = death_on_target
        self.dispatching_agent_death = dispatching_agent_death
        if routing_agent_death:
            self.routing_hwm = self.fleetsize
        if dispatching_agent_death:
            self.dispatching_hwm = 1000 + self.fleetsize
        self.agent_hive = None
        self.ma_disp_cls = MultiAgent if not transform_dispatching_partobs else MultiDispAgent
        self.sa_disp_cls = SingleAgent if not transform_dispatching_partobs else SingleDispAgent

        self.client = client
        self.started = False
        self.model_path = model_path
        self.startport = startport

        self.config = self.get_config(sim_config)
        self.object_agents = self.get_agents()
        self.possible_agents = [agent.name for agent in self.object_agents]
        if routing_agent_death:
            self.possible_agents.extend([str(i) for i in range(self.fleetsize, 1000)])
        if dispatching_agent_death:
            self.possible_agents.extend(
                [str(i) for i in range(self.fleetsize + 1000, 2000)]
            )
        super().__init__(self.object_agents)
        if not (self.pseudo_dispatcher or self.dispatching_policy is not None) or not (self.pseudo_routing or self.routing_policy is not None):
            self.reset()

    #Nó nhận Observation từ AnyLogic, nhìn vào caller để xác định decision request này thuộc về routing agent hay dispatching agent nào, đồng thời xử lý trường hợp Agent bị thay thế bằng alias.
    def _agent_selection_fn(self, alpyne_obs):
        if (self.pseudo_routing or self.routing_policy is not None) and (self.pseudo_dispatcher or self.dispatching_policy is not None) and self._terminal_alternative(alpyne_obs):
            return "dummy"
        if alpyne_obs.caller < 1000 and self.routing_agent_death:
            return self.routing_aliases[str(alpyne_obs.caller)]
        elif alpyne_obs.caller >= 1000 and self.dispatching_agent_death:
            return self.dispatching_aliases[str(alpyne_obs.caller)]
        else:
            return str(alpyne_obs.caller)

    def step(self, action):
        self.stepcounter += 1
        if self.agent_selection != "dummy" and int(self.agent_selection) < 1000 and self.routing_agent_death:
            self.agent_selection = self.routing_aliases_rev[self.agent_selection]
        elif self.agent_selection != "dummy" and int(self.agent_selection) >= 1000 and self.dispatching_agent_death:
            self.agent_selection = self.dispatching_aliases_rev[self.agent_selection]
        return super().step(action)

    def get_agents(self):
        agents = []
        self.agent_hive = (
            HiveContext() if not self.pseudo_routing else CollisionFreeRoutingHive()
        )
        if (self.pseudo_routing or self.routing_policy is not None) and (self.pseudo_dispatcher or self.dispatching_policy is not None):
            agents.append(ZooAgent("dummy", Dummy(self.agent_hive,self.max_fleetsize, 8, all_ma=self.all_ma,)))
        if self.pseudo_routing:
            if self.routing_ma:
                agents.extend(
                    [
                        ZooAgent(str(i), CollisionFreeRouting(self.agent_hive), False)
                        for i in range(self.fleetsize)
                    ]
                )
                if self.routing_agent_death:
                    self.routing_hwm = self.fleetsize
                    self.routing_aliases = {
                        str(i): str(i) for i in range(self.fleetsize)
                    }
                    self.routing_aliases_rev = {
                        str(i): str(i) for i in range(self.fleetsize)
                    }
            else:
                raise ValueError("pseudo routing only available for multiagent")
        else:
            if self.routing_ma:
                if self.routing_policy is None:
                    agents.extend(
                        [
                            ZooAgent(
                                str(i),
                                MultiAgent(
                                    self.agent_hive,
                                    self.max_fleetsize,
                                    8,
                                    all_ma=self.all_ma,
                                    die_on_target=self.death_on_target,
                                    direction_reward=self.direction_reward,
                                ),
                            )
                            for i in range(self.fleetsize)
                        ]
                    )
                else:
                    agents.extend(
                        [
                            ZooAgent(
                                str(i),
                                ApplyPolicyAgent(self.agent_hive, self.routing_policy, 
                                    MultiAgent(
                                        self.agent_hive,
                                        self.max_fleetsize,
                                        8,
                                        all_ma=self.all_ma,
                                        die_on_target=self.death_on_target,
                                        direction_reward=self.direction_reward,
                                    ),
                                )
                            )
                            for i in range(self.fleetsize)
                        ]
                    )
                if self.routing_agent_death:
                    self.routing_hwm = self.fleetsize
                    self.routing_aliases = {
                        str(i): str(i) for i in range(self.fleetsize)
                    }
                    self.routing_aliases_rev = {
                        str(i): str(i) for i in range(self.fleetsize)
                    }
            else:
                agents.append(
                    ZooAgent(
                        "2000", SingleAgent(self.agent_hive, self.max_fleetsize, 8)
                    )
                )
        if self.dispatch:
            if self.pseudo_dispatcher:
                if self.dispatching_ma:
                    agents.extend(
                        [
                            ZooAgent(
                                str(i),
                                self.pseudo_disp_cls(
                                    self.agent_hive, 1, self.pseudo_dispatcher_distance, rule = self.pseudo_dispatcher_rule
                                ),
                                False,
                            )
                            for i in range(1000, 1000 + self.fleetsize)
                        ]
                    )
                else:
                    agents.append(
                        ZooAgent(
                            "2001",
                            self.pseudo_disp_cls(
                                self.agent_hive,
                                self.fleetsize,
                                self.pseudo_dispatcher_distance,
                                rule = self.pseudo_dispatcher_rule
                            ),
                            False,
                        )
                    )
            else:
                if self.dispatching_ma:
                    if self.dispatching_policy is None:
                        agents.extend(
                            [
                                ZooAgent(
                                    str(i),
                                    self.ma_disp_cls(
                                        self.agent_hive,
                                        self.max_fleetsize,
                                        6 if not self.config.obs_include_agv_target else 8,
                                        True,
                                        all_ma=self.all_ma,
                                    ),
                                )
                                for i in range(1000, 1000 + self.fleetsize)
                            ]
                        )
                    else:
                        agents.extend(
                            [
                                ZooAgent(
                                    str(i),
                                    ApplyPolicyAgent(
                                        self.agent_hive, self.dispatching_policy if not isinstance(self.dispatching_policy, list) else self.dispatching_policy[i%2],
                                        self.ma_disp_cls(
                                            self.agent_hive,
                                            self.max_fleetsize,
                                            6 if not self.config.obs_include_agv_target else 8,
                                            True,
                                            all_ma=self.all_ma,
                                        ),
                                    )
                                )
                                for i in range(1000, 1000 + self.fleetsize)
                            ]
                        )
                    if self.dispatching_agent_death:
                        self.dispatching_hwm = self.fleetsize + 1000
                        self.dispatching_aliases = {
                            str(i): str(i) for i in range(1000, 1000 + self.fleetsize)
                        }
                        self.dispatching_aliases_rev = {
                            str(i): str(i) for i in range(1000, 1000 + self.fleetsize)
                        }
                else:
                    agents.append(
                        ZooAgent(
                            "2001",
                            self.sa_disp_cls(
                                self.agent_hive,
                                self.max_fleetsize,
                                6 if not self.config.obs_include_agv_target else 8,
                                True,
                            ),
                        )
                    )
        return agents

    def get_config(self, config_args):
        conf = build_config(config_args, self.fleetsize)
        conf.obs_include_nodes_in_reach = True
        conf.coordinates = True
        return conf

    def seed(self, seed: int = None):
        if seed is not None:
            self.config.seed = seed
        else:
            self.config.seed = random.randint(0, 10000)

    def _start(self):
        if self.client is None:
            port = self.startport
            self.client = AlpyneClient(self.model_path, blocking = True, port=port, verbose=False)

        self.run = self.client.create_reinforcement_learning(self.config)
        self.started = True
        super()._start(self.run)

    def reset(
        self,
        config=None,
        seed=None,
        return_info=False,
        options={},
    ) -> "BaseAlpyneZoo.PyObservationType":
        # Station observations are event-specific: discard previous-episode
        # information before starting the next episode.
        self._mappo_station_cache = None
        self._mappo_station_source = "missing"
        if not self.started:
            self._start()
        self.stepcounter = 0
        self.seed(seed=seed)
        if self.fleetsize_upper is not None:
            self.config.fleetsize = self.fleetsize = random.randint(self.fleetsize_under, self.fleetsize_upper)
        if self.routing_ma and self.routing_agent_death:
            self.routing_hwm = self.fleetsize
            self.routing_aliases = {str(i): str(i) for i in range(self.fleetsize)}
            self.routing_aliases_rev = {str(i): str(i) for i in range(self.fleetsize)}
        if (
            self.dispatch
            and not self.pseudo_dispatcher
            and self.dispatching_ma
            and self.dispatching_agent_death
        ):
            self.dispatching_hwm = self.fleetsize + 1000
            self.dispatching_aliases = {
                str(i): str(i) for i in range(1000, 1000 + self.fleetsize)
            }
            self.dispatching_aliases_rev = {
                str(i): str(i) for i in range(1000, 1000 + self.fleetsize)
            }
        # if self.pseudo_dispatcher and self.pseudo_routing:
        #     return_val = super().reset(self.config, seed, return_info, options)
        #     return self.observation_space("dummy").sample(), 0, False, {}
        else:
            return super().reset(self.config, seed, return_info, options, dont_collect = options.get("dont_collect", False))

    def action_space(self, agent) -> gymnasium.spaces.Space:
        return super().action_space(agent)

    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        return super().observation_space(agent)

    def get_mappo_state(self):
        """Physical AGV + station state for the centralized critic.

        Routing observations from the *original* AnyLogic export contain
        only AGV rows, so station values may be cached or absent.
        Routing observations from the corrected export also contain
        registered Station rows; those values are fresh at every event.
        This method preserves the existing ``agvs/stat/ego`` interface.
        """
        if not hasattr(self, "_mappo_current_state"):
            raise RuntimeError("MAPPO state not available; call reset/last first")
        state = self._mappo_current_state
        return {key: value.copy() if hasattr(value, "copy") else value
                for key, value in state.items()}

    def get_mappo_station_source(self):
        """Diagnostic only: fresh_routing, fresh_dispatch, cached, missing."""
        return getattr(self, "_mappo_station_source", "missing")

    def get_mappo_time(self):
        """AnyLogic model time, saved at the most recent decision event."""
        return float(getattr(self, "_mappo_current_time", 0.0))

    def _capture_mappo_state(self, alpyne_obs):
        """Capture AGV rows and registered Station rows when provided.

        Tobias's original Java Controller.getObservationRouting() emits AGV
        rows only; this method remains backwards-compatible with that layout.
        The updated Java function additionally emits Station rows for routing,
        making the station portion of the critic state fresh every decision.
        A Python-only change CANNOT make stations fresh with the old export.
        """
        import numpy as np

        raw = np.asarray(alpyne_obs.obs, dtype=np.float32)
        if raw.ndim != 2:
            raise ValueError(
                f"MAPPO expected 2D observation; got {raw.shape} "
                f"from caller={alpyne_obs.caller}"
            )

        n_agv = int(self.fleetsize)
        n_station_slots = len(self.agent_hive.stations or ())
        n_station_observed = int(self.agent_hive.statistics["n_stat"])
        reported_n_agv = int(self.agent_hive.statistics["n_agv"])
        if (n_agv < 1 or n_agv > self.max_fleetsize
                or n_station_slots < 1
                or n_station_observed < 0
                or n_station_observed > n_station_slots
                or reported_n_agv != n_agv):
            raise ValueError(
                f"Invalid MAPPO sizes: configured_agv={n_agv}, "
                f"reported_agv={reported_n_agv}, max_fleetsize={self.max_fleetsize}, "
                f"station_slots={n_station_slots}, "
                f"registered_stations={n_station_observed}"
            )

        caller = int(alpyne_obs.caller)
        ego = caller % 1000
        if not 0 <= ego < n_agv:
            raise ValueError(
                f"Invalid physical AGV index: caller={caller}, "
                f"ego={ego}, fleetsize={n_agv}"
            )
        is_dispatching = caller >= 1000 and caller != 2000
        station_rows_received = raw.shape[0] - n_agv
        if is_dispatching:
            valid_row_counts = (n_station_observed,)
        else:
            # Legacy routing = AGVs only; corrected routing = AGVs + stations.
            valid_row_counts = (0, n_station_observed)
        if station_rows_received not in valid_row_counts:
            raise ValueError(
                f"Unexpected Matrix observation: caller={caller}, "
                f"shape={raw.shape}; station rows received="
                f"{station_rows_received}, expected one of {valid_row_counts}"
            )

        feature_len = max(
            int(self.agent_hive.statistics["rout_len"]),
            int(self.agent_hive.statistics["disp_len"]),
            raw.shape[1],
        )
        agvs = np.zeros((self.max_fleetsize, feature_len), dtype=np.float32)
        agvs[:n_agv, :raw.shape[1]] = raw[:n_agv]

        if station_rows_received > 0:
            # Fresh registered station observations, including on Routing
            # events if the corrected AnyLogic export is used.
            stat = np.zeros((n_station_slots, feature_len), dtype=np.float32)
            stat[:n_station_observed, :raw.shape[1]] = raw[n_agv:]
            self._mappo_station_cache = stat.copy()
            self._mappo_station_source = (
                "fresh_dispatch" if is_dispatching else "fresh_routing"
            )
        else:
            cached = getattr(self, "_mappo_station_cache", None)
            if cached is None:
                stat = np.zeros((n_station_slots, feature_len), dtype=np.float32)
                self._mappo_station_source = "missing"
            elif cached.shape != (n_station_slots, feature_len):
                raise ValueError(
                    f"Station cache shape {cached.shape} does not match "
                    f"({n_station_slots}, {feature_len})"
                )
            else:
                stat = cached.copy()
                self._mappo_station_source = "cached"

        self._mappo_current_state = {
            "agvs": agvs,
            "stat": stat,
            "ego": ego,
        }
        try:
            self._mappo_current_time = float(self.run.get_state()[1]["model_time"])
        except (AttributeError, KeyError, IndexError, TypeError):
            self._mappo_current_time = 0.0

    def _save_observation(self, alpyne_obs, agent, obs, reward, done, info):
        # The raw packet includes ALL station rows after the corrected Java
        # export. Save them for the CENTRAL critic before changing actor obs.
        self._capture_mappo_state(alpyne_obs)
        # Keep the original Routing ACTOR input exactly as before: in Tobias's
        # baseline, Routing stat observations were zero. Its GNN extractor uses
        # obs["stat"] even with with_stations=False; feeding fresh station data
        # to the actor would silently change the old policy and violate the
        # intended separation of actor vs centralized critic inputs.
        if int(alpyne_obs.caller) < 1000 and isinstance(obs, dict) and "stat" in obs:
            import numpy as np
            obs = obs.copy()
            obs["stat"] = np.zeros_like(obs["stat"])
        if "statTitles" in alpyne_obs.names():
            self.statistics = {
                title: value
                for title, value in zip(alpyne_obs.statTitles, alpyne_obs.statValues)
            }
            self.statistics["fleetsize" ] = self.fleetsize
        agent_died = ("in_system" in info.keys() and not info["in_system"]) or (
            "at_target" in info.keys() and info["at_target"]
        )
        if (
            agent != "dummy"
            and int(agent) < 1000
            and self.routing_agent_death
            and agent_died
            and agent in self.agents
        ):
            self.remove_agent(agent)
            name = str(self.routing_hwm)
            original_name = self.routing_aliases_rev[agent]
            self.routing_aliases[original_name] = name
            self.routing_aliases_rev[name] = original_name  # bidirectional
            if self.routing_policy is None:
                self.add_agent(
                    ZooAgent(
                        name,
                        MultiAgent(
                            self.agent_hive,
                            self.max_fleetsize,
                            8,
                            all_ma=self.all_ma,
                            die_on_target=self.death_on_target,
                            direction_reward=self.direction_reward,
                        ),
                    )
                )
            else:
                self.add_agent(
                    ZooAgent(
                        name,
                        ApplyPolicyAgent(self.agent_hive, self.routing_policy, 
                            MultiAgent(
                                self.agent_hive,
                                self.max_fleetsize,
                                8,
                                all_ma=self.all_ma,
                                die_on_target=self.death_on_target,
                                direction_reward=self.direction_reward,
                            ),
                        )
                    )
                )
            self.routing_hwm += 1
        elif (
            agent != "dummy"
            and int(agent) >= 1000
            and self.dispatching_agent_death
            and agent_died
            and agent in self.agents
        ):
            self.remove_agent(agent)
            name = str(self.dispatching_hwm)
            original_name = self.dispatching_aliases_rev[agent]
            self.dispatching_aliases[original_name] = name
            self.dispatching_aliases_rev[name] = original_name  # bidirectional
            if self.dispatching_policy is None:
                self.add_agent(
                    ZooAgent(
                        name,
                        self.ma_disp_cls(
                            self.agent_hive,
                            self.max_fleetsize,
                            6 if not self.config.obs_include_agv_target else 8,
                            True,
                            all_ma=self.all_ma,
                        ),
                    )
                )
            else:
                self.add_agent(
                    ZooAgent(
                        name,
                        ApplyPolicyAgent(
                            self.agent_hive, self.dispatching_policy if not isinstance(self.dispatching_policy, list) else self.dispatching_policy[int(original_name)%2],
                            self.ma_disp_cls(
                                self.agent_hive,
                                self.max_fleetsize,
                                6 if not self.config.obs_include_agv_target else 8,
                                True,
                                all_ma=self.all_ma,
                            ),
                        )
                    )
                )
            self.dispatching_hwm += 1
        return super()._save_observation(alpyne_obs, agent, obs, reward, done, info)

    def _terminal_alternative(self, observation) -> bool:
        """Optional method to add *extra* terminating conditions"""
        terminal_max_steps = (
            (self.stepcounter >= self.max_steps)
            if self.max_steps is not None
            else False
        )
        time = 0 if self.max_seconds is None else self.run.get_state()[1]["model_time"]
        terminal_max_seconds = (
            time >= self.max_seconds if self.max_seconds is not None else False
        )
        if time < self.warmup_time:
            self.warmup_statistics = self.statistics.copy()
        return terminal_max_steps or terminal_max_seconds

    def close(self):
        self.sim.stop()

    def render(self, dict_mode=False):
        obs, _, _, _ = self.last()
        obs_dict = self.agent_behaviors[self.agent_selection]._state_dict(obs)
        if dict_mode:
            return obs_dict
        x_max = 400
        y_max = 200
        r = 5
        im = Image.new("RGB", (x_max + 10, y_max + 10))
        draw = ImageDraw.Draw(im)
        color_agv = (255, 0, 0)
        color_stat = (0, 50, 0)
        color_main = (0, 0, 255)
        for station in obs_dict["stations"].values():
            x = station["position"][0] * x_max
            y = station["position"][1] * y_max
            draw.ellipse((x - 2 * r, y - 2 * r, x + 2 * r, y + 2 * r), fill=color_stat)
        agvs = list(obs_dict["agvs"].items())
        agvs.reverse()
        for name, agv in agvs:
            if agv["in_system"] == 1:
                color = color_main if name == "0" else color_agv
                xl = agv["last_node"][0] * x_max
                yl = agv["last_node"][1] * y_max
                xn = agv["next_node"][0] * x_max
                yn = agv["next_node"][1] * y_max
                if "target_node" in agv.keys():
                    xt = agv["target_node"][0] * x_max
                    yt = agv["target_node"][1] * y_max
                    draw.ellipse(
                        (xt - r / 2, yt - r / 2, xt + r / 2, yt + r / 2), fill=color
                    )
                draw.rectangle((xl - r, yl - r, xl + r, yl + r), fill=color)
                draw.ellipse((xn - r, yn - r, xn + r, yn + r), fill=color)
                draw.text((xl - r, yl - r), name)
                draw.line((xl, yl, xn, yn), fill=color)
        del draw

        return im
