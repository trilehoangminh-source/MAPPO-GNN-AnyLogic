#==================== EVENT COLLECTOR (ROUTING KHÔNG MASK) ====================
# Ghép reward vào quyết định trước theo đúng AGV vật lý; giữ các episode/lifetime riêng khi thay alias.
# BẢN ĐỌC HIỂU: chỉ bổ sung comment/khoảng trắng; KHÔNG thay đổi câu lệnh gốc.
# Lưu ý: comment vẫn làm thay đổi SHA256 của file (cần init run mới nếu dùng train).
#================================ HẾT GIỚI THIỆU FILE =================================


"""Event-driven Tobias/AnyLogic MAPPO collector with lifecycle-safe GAE segments.

Reward for an action is consumed only upon the NEXT observation of that same
(role, physical AGV). A change of PettingZoo alias due to an AGV leaving the
system is a terminal boundary for that AGV's old trajectory.

Final actions of agents not called again before episode cutoff have no observed
reward. They are reported/excluded, never assigned a fake 0.

Current routing policy:
- Routing's GATv2 samples from all five logits, without applying action_mask.
- We still retain the original mask for diagnostics and STOP-rate metrics.
- A routing direction missing from the original mask is recorded, not rejected
  by the collector. AnyLogic is responsible for processing that direction.
- Dispatch Front/Rear still enforce their original action mask.
"""



#========================= MODULE 01 - CÁC THƯ VIỆN / IMPORT ==========================
# import/from trong Python tương tự #include ở C, nhưng có hệ thống module và package.
from __future__ import annotations

from collections import Counter, defaultdict
import numpy as np
#===================== KẾT THÚC MODULE 01 - CÁC THƯ VIỆN / IMPORT =====================




#=========================== MODULE 02 - LỚP EVENTCOLLECTOR ===========================
# Ghi các transition theo thời gian sự kiện và ghép reward vào quyết định đúng agent.
class EventCollector:


    #============================ MODULE CON 01 - __INIT__ ============================
    # Mục đích: Khởi tạo thuộc tính, tham số và bộ nhớ cho đối tượng.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def __init__(self, model, environment, stop_mask_mode="baseline"):
        if stop_mask_mode not in ("baseline", "forbid_when_move"):
            raise ValueError(f"Unknown stop_mask_mode: {stop_mask_mode}")
        self.stop_mask_mode = stop_mask_mode
        self.model = model
        self.env = environment
        self.unfinished_last_episode = 0
    #======================= KẾT THÚC MODULE CON 01 - __INIT__ ========================




    #============================ MODULE CON 02 - _PREPARE ============================
    # Mục đích: Chuyển obs thành float32; giữ mask để chẩn đoán nhưng Routing V16-B vẫn không mask.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def _prepare(self, observation, role):
        mask = np.asarray(observation['action_mask'], dtype=np.float32).reshape(-1)
        n = self.model.roles[role].actor.action_dim
        if len(mask) < n:
            raise ValueError(f'{role}: mask has {len(mask)} entries; expected {n}')
        mask = mask[:5] if role == 'routing' else mask[:n]
        # Mask is still recorded on Routing observations for diagnostics, but
        # TobiasGraphRoutingActor.dist() intentionally ignores it. Dispatch
        # policies continue to apply it. The optional STOP A/B mode below
        # therefore only changes Routing diagnostic mask entries in this version.
        mask = mask.copy()
        if (role == 'routing' and self.stop_mask_mode == 'forbid_when_move'
                and mask[0] > 0.5 and (mask[1:5] > 0.5).any()):
            mask[0] = 0.0
        if not (mask > 0.5).any():
            raise RuntimeError(f'{role}: no action allowed by environment mask')
        return {
            'agvs': np.asarray(observation['agvs'], dtype=np.float32).copy(),
            'stat': np.asarray(observation['stat'], dtype=np.float32).copy(),
            'action_mask': mask.copy(),
        }
    #======================= KẾT THÚC MODULE CON 02 - _PREPARE ========================




    #======================= MODULE CON 03 - _IS_AGENT_REPLACED =======================
    # Mục đích: Nhận biết AGV bị thay alias/lifetime để không nối hai trajectory khác đời AGV.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def _is_agent_replaced(self, role, info):
        """Match conditions Matrix._save_observation uses to replace aliases."""
        if not isinstance(info, dict):
            return False
        if role == 'routing' and getattr(self.env, 'routing_agent_death', False):
            return (('in_system' in info and not bool(info['in_system']))
                    or ('at_target' in info and bool(info['at_target'])))
        if role != 'routing' and getattr(self.env, 'dispatching_agent_death', False):
            return (('in_system' in info and not bool(info['in_system']))
                    or ('at_target' in info and bool(info['at_target'])))
        return False
    #================== KẾT THÚC MODULE CON 03 - _IS_AGENT_REPLACED ===================




    #============================ MODULE CON 04 - EPISODE =============================
    # Mục đích: Chạy một episode event-driven; ghép reward với action trước của cùng physical AGV; xuất trajectory và thống kê.
    # Cú pháp: self là đối tượng đang xử lý (gần tương tự con trỏ tới struct trong C).
    def episode(self, seed=None, train=True, max_event_steps=None):

        #========================= MỤC NHỎ - KHỞI TẠO EPISODE =========================
        # Reset AnyLogic; tạo pending action và các GAE segment riêng theo lifetime.
        self.env.reset(seed=seed)
        pending = {}  # (role, physical index) -> action awaiting its reward
        active = defaultdict(list)  # GAE segment for current lifetime only
        segments = []  # separate trajectories; NEVER concatenate lifetimes

        def finish_segment(key):
            steps = active.pop(key, None)
            if steps:
                segments.append(steps)

        events = 0
        delivered_reward = 0.0
        terminal = False
        alias_boundaries = Counter()
        action_histograms = defaultdict(Counter)
        stopped_with_move = 0
        stopped_without_move = 0
        routing_move_available_decisions = 0
        routing_stop_masked_decisions = 0
        mask_violations = 0
        routing_invalid_selected = 0
        ended_with = 'agents_exhausted'


        #========================= MỤC NHỎ - VÒNG LẶP SỰ KIỆN =========================
        # Mỗi lần chỉ có một agent được gọi, KHÔNG phải tick đồng bộ toàn hệ thống.
        while self.env.agents:
            obs, reward, done, info = self.env.last()
            agent_id = str(self.env.agent_selection)
            state = self.env.get_mappo_state()
            physical = int(state['ego'])
            role = self.model.role_for(agent_id, physical, self.model.cfg.two_fleets)
            if role not in self.model.roles:
                raise RuntimeError(f'Unexpected role={role}, agent={agent_id}')
            key = (role, physical)
            sim_time = float(self.env.get_mappo_time())
            replaced = self._is_agent_replaced(role, info)


            #=============== MỤC NHỎ - GHÉP REWARD VỚI QUYẾT ĐỊNH TRƯỚC ===============
            # Khi cùng physical AGV được gọi lại, mới đóng transition và gán reward/next_value.
            if key in pending:
                previous = pending.pop(key)
                previous['reward'] = float(reward)
                # Aliased agents are NEW PettingZoo lives even if their
                # physical slot stays identical. Do not bootstrap across lives.
                previous['done'] = bool(done or replaced)
                previous['next_value'] = (
                    0.0 if previous['done'] else float(self.model.value(role, state))
                )
                previous['elapsed_seconds'] = float(sim_time - previous['time'])
                if previous['elapsed_seconds'] < -1e-5:
                    raise RuntimeError(f'AnyLogic time moved backward for {key}')
                active[key].append(previous)
                delivered_reward += float(reward)


            #=============== MỤC NHỎ - NGẮT LIFETIME KHI ALIAS THAY ĐỔI ===============
            # Đóng segment để GAE không bootstrap qua hai vòng đời khác nhau.
            if replaced:
                alias_boundaries[role] += 1
                finish_segment(key)

            if done:
                terminal = True
                ended_with = 'environment_terminal'
                finish_segment(key)
                break


            #==================== MỤC NHỎ - THỐNG KÊ MASK ROUTING =====================
            # Đọc mask để đếm Invalid/Stop; actor Routing không bị ép theo mask.
            if role == 'routing':
                original_mask = np.asarray(obs['action_mask'], dtype=np.float32).reshape(-1)[:5]
                move_available = bool((original_mask[1:5] > 0.5).any())
                routing_move_available_decisions += int(move_available)
                if (self.stop_mask_mode == 'forbid_when_move' and move_available
                        and original_mask[0] > 0.5):
                    routing_stop_masked_decisions += 1
            actor_obs = self._prepare(obs, role)

            #=============== MỤC NHỎ - CHỌN HÀNH ĐỘNG BẰNG ACTOR/CRITIC ===============
            # Lưu action, log-prob và ValueNorm-denormalized critic value.
            action, logp, value = self.model.act(role, actor_obs, state, greedy=not train)
            action = int(action)
            # Always enforce tensor action bounds (Routing has five logits).
            # Only Dispatch requires action-mask legality. Unmasked Routing
            # actions can be invalid directions and must pass to AnyLogic,
            # exactly as in the no-mask Tobias routing experiment.
            if not 0 <= action < self.model.roles[role].actor.action_dim:
                raise RuntimeError(f'{role} AGV={physical}: action {action} out of bounds')
            if actor_obs['action_mask'][action] <= 0.5:
                mask_violations += 1
                if role == 'routing':
                    routing_invalid_selected += 1
                else:
                    raise RuntimeError(f'{role} AGV={physical}: action {action} violates mask '
                                       f'{actor_obs["action_mask"].tolist()}')

            action_histograms[role][str(action)] += 1
            if role == 'routing' and action == 0:
                if (actor_obs['action_mask'][1:5] > 0.5).any():
                    stopped_with_move += 1
                else:
                    stopped_without_move += 1

            pending[key] = {
                'role': role, 'observation': actor_obs, 'state': state,
                'action': action, 'logp': float(logp), 'value': float(value),
                'time': sim_time,
            }

            #================= MỤC NHỎ - GỬI HÀNH ĐỘNG SANG ANYLOGIC ==================
            # Tiếp tục simulator tới event kế tiếp rồi lặp vòng.
            self.env.step(action)
            events += 1
            if max_event_steps is not None and events >= max_event_steps:
                ended_with = 'max_event_steps'
                break


        #========================= MỤC NHỎ - ĐÓNG CÁC SEGMENT =========================
        # Thống kê pending action cuối episode chưa nhận được reward.
        for key in list(active):
            finish_segment(key)
        self.unfinished_last_episode = len(pending)
        statistics = dict(self.env.statistics or {})
        pending_by_role = Counter(role for role, _ in pending)

        return segments, {
            'events': events,
            'closed_transitions': sum(len(t) for t in segments),
            'discarded_pending': len(pending),
            'discarded_pending_by_role': dict(pending_by_role),
            'delivered_reward': delivered_reward,
            'reached_terminal': terminal,
            'end_reason': ended_with,
            'trajectories': len(segments),
            'alias_boundaries_by_role': dict(alias_boundaries),
            'statistics': statistics,
            'action_diagnostics': {
                'actions_by_role': {k: dict(v) for k, v in action_histograms.items()},
                'routing_stop_with_move_available': stopped_with_move,
                'routing_stop_without_move_available': stopped_without_move,
                'selected_mask_violations': mask_violations,
                'routing_invalid_selected': routing_invalid_selected,
                'routing_action_mask_applied': False,
                'dispatch_action_mask_applied': True,
                'stop_mask_mode': self.stop_mask_mode,
                'routing_move_available_decisions': routing_move_available_decisions,
                'routing_stop_masked_decisions': routing_stop_masked_decisions,
            },
        }
    #======================== KẾT THÚC MODULE CON 04 - EPISODE ========================

#====================== KẾT THÚC MODULE 02 - LỚP EVENTCOLLECTOR =======================


