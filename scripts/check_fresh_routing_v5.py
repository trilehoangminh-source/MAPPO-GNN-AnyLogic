"""Kiểm tra Routing mới có thực sự chứa AGV + Station (Alpyne 1.2.1).

Chạy trong thesis với .venv311 và Java 17:
    python -u scripts/check_fresh_routing_v5.py

Chỉ thử tối đa 60 quyết định với chính sách chẩn đoán đơn giản; KHÔNG học PPO.
Không sửa model, không ghi checkpoint. Dispatch chọn một node Station thật
trong networkcontext do mô hình xuất; Routing dừng ngay khi quan sát đầu tiên.
"""
from __future__ import annotations

from pathlib import Path
import traceback

from alpyne.sim import AnyLogicSim
from alpyne.constants import EngineState

MODEL = Path('envs/Matrix_critic_fresh.zip')
JAVA17 = Path(r'C:\Program Files\AnyLogic 8.9 Personal Learning Edition\jre\bin\java.exe')
FLEETSIZE = 8
PORT = 51360
MAX_EVENTS = 60


def station_node_ids(networkcontext):
    """Đúng quy ước HiveContext._catch_context() trong source Tobias.

    networkcontext gồm các dòng node (cột áp chót == 0) và các dòng arc.
    ID node = vị trí của nó trong các dòng node; cột thứ 3 == 1 là Station.
    """
    if networkcontext is None:
        return []
    node_rows = [row for row in networkcontext if len(row) >= 4 and float(row[-2]) == 0.0]
    return [i for i, row in enumerate(node_rows) if float(row[2]) == 1.0]


def print_routing_result(caller, matrix):
    rows = len(matrix)
    widths = sorted({len(row) for row in matrix}) if rows else []
    print(f'[TEST] ROUTING caller={caller}', flush=True)
    print(f'[TEST] ROUTING obs: rows={rows}, widths={widths}', flush=True)
    if rows <= FLEETSIZE:
        print('[TEST] CHUA DAT: Routing van chi co hang AGV (hoac thieu hang).', flush=True)
        return
    station_rows = matrix[FLEETSIZE:]
    nonzero = sum(any(abs(float(v)) > 1e-12 for v in row) for row in station_rows)
    print(f'[TEST] SO HANG STATION: {len(station_rows)}', flush=True)
    print(f'[TEST] STATION KHAC 0: {nonzero}/{len(station_rows)}', flush=True)
    print(f'[TEST] STATION DAU TIEN (12 so): {list(station_rows[0][:12])}', flush=True)
    if nonzero:
        print('[TEST] PASS: Routing da nhan hang Station thuc te!', flush=True)
    else:
        print('[TEST] Routing co hang Station, nhung du lieu deu 0; can kiem tra tiep.', flush=True)


def main():
    if not MODEL.is_file():
        raise FileNotFoundError(f'Khong tim thay ZIP: {MODEL.resolve()}')
    if not JAVA17.is_file():
        raise FileNotFoundError(f'Khong tim thay Java 17: {JAVA17}')

    print('[TEST] Khoi dong Alpyne 1.2.1 / Java 17...', flush=True)
    sim = AnyLogicSim(
        str(MODEL), port=PORT, java_exe=str(JAVA17),
        max_server_await_time=120,
        auto_lock=True, lock_defaults={'timeout': 120},
        engine_overrides={'stop_time': 120, 'seed': 43},
    )
    config = {
        'runmode': 1, 'fleetsize': FLEETSIZE, 'seed': 43,
        'dispatch': True, 'routing_ma': True, 'dispatching_ma': True,
        'fixed_fleets': True, 'routing_interval': 2.0,
        'dispatching_interval': 360.0,
        'coordinates': True, 'obs_include_nodes_in_reach': True,
        'station_availability': 0.9, 'station_io_quote': 0.9,
        'station_mttr': 5.0,
    }
    status = sim.reset(**config)
    print(f'[TEST] Reset OK: state={status.state}, time={status.time}', flush=True)
    nodes = []

    for i in range(MAX_EVENTS + 1):
        if status.state == EngineState.ERROR:
            raise RuntimeError(f'AnyLogic MODEL_ERROR: {status.message}')
        if status.state == EngineState.FINISHED or status.stop:
            print(f'[TEST] Mo phong ket thuc truoc Routing: step={i}, time={status.time}', flush=True)
            return
        if status.state != EngineState.PAUSED:
            raise RuntimeError(f'Engine state khong mong doi: {status.state} ({status.message})')

        obs = status.observation
        if obs is None:
            raise RuntimeError('Status khong co observation.')
        caller = int(obs['caller'])
        matrix = obs['obs']
        if matrix is None:
            raise RuntimeError(f'Observation obs=None, caller={caller}')

        # networkcontext duoc AnyLogic chi gui o lan takeAction dau tien.
        new_nodes = station_node_ids(obs.get('networkcontext'))
        if new_nodes:
            nodes = new_nodes
            print(f'[TEST] Tim thay {len(nodes)} node Station, ID dau: {nodes[:8]}', flush=True)

        if i < 12:
            print(f'[TEST] Decision {i}: caller={caller}, time={status.time}, rows={len(matrix)}', flush=True)

        if 0 <= caller < 1000:
            print_routing_result(caller, matrix)
            return

        if not nodes:
            print('[TEST] Chua co networkcontext/station ID: dung de tranh gui action sai.', flush=True)
            print('[TEST] networkcontext:', type(obs.get('networkcontext')).__name__, flush=True)
            return

        # Controller.action_target trong Tobias can node ID thuc, khong phai
        # chi so station; dung node dau tien chi de mo khoa buoc Routing.
        if 1000 <= caller < 2000:
            actions = [int(nodes[0])]
        elif caller == 2001:
            actions = [int(nodes[0])] * FLEETSIZE
        elif caller == 2000:
            actions = [0] * FLEETSIZE
        else:
            raise RuntimeError(f'Unknown caller code: {caller}')

        status = sim.take_action(receiver=caller, actions=actions)

    print(f'[TEST] Chua thay Routing trong {MAX_EVENTS} actions; khong ket luan la loi.', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        print('[TEST] Loi khi chay AnyLogic:', flush=True)
        traceback.print_exc()
        raise
