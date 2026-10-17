"""Kiểm tra observation Routing/Station của Matrix AnyLogic với Alpyne 1.2.1.

Chạy bằng Python 3.11 trong thư mục gốc thesis:
    python -u scripts/check_fresh_routing.py

Chỉ reset mô phỏng và đọc observation đầu tiên; KHÔNG train MAPPO.
"""
from __future__ import annotations

from pathlib import Path
import traceback

from alpyne.sim import AnyLogicSim

MODEL = Path("envs/Matrix_critic_fresh.zip")
JAVA17 = Path(r"C:\Program Files\AnyLogic 8.9 Personal Learning Edition\jre\bin\java.exe")
FLEETSIZE = 8
PORT = 51355


def config_key(name: str) -> str:
    """Schema model dùng RL_x; Alpyne SimConfiguration dùng x."""
    return name[3:] if name.startswith("RL_") else name


def build_config(sim: AnyLogicSim) -> dict:
    # Đọc giá trị mặc định từ schema Java, nhưng CHUYỂN sang tên của
    # SimConfiguration trước khi truyền cho sim.reset().
    config = {
        config_key(name): field.value
        for name, field in sim.schema.inputs.items()
        if field.value is not None
    }

    # Dùng cấu hình gần với smoke test MAPPO cũ.
    overrides = {
        "runmode": 1,                 # RL Experiment; 3 là Pypeline
        "fleetsize": FLEETSIZE,
        "seed": 43,
        "dispatch": True,
        "routing_ma": True,
        "dispatching_ma": True,
        "fixed_fleets": True,
        "routing_interval": 2.0,
        "dispatching_interval": 360.0,
        "coordinates": True,
        "obs_include_nodes_in_reach": True,
        "station_availability": 0.9,
        "station_io_quote": 0.9,
        "station_mttr": 5.0,
    }

    allowed = {config_key(name) for name in sim.schema.inputs}
    missing = sorted(set(overrides) - allowed)
    if missing:
        raise RuntimeError(f"Các tham số không có trong RL Experiment: {missing}")

    config.update(overrides)
    # Phòng trường hợp sau này schema có field không được hỗ trợ.
    config = {key: val for key, val in config.items() if key in allowed}
    print("[TEST] Reset keys:", ", ".join(sorted(config)), flush=True)
    return config


def main() -> None:
    if not MODEL.is_file():
        raise FileNotFoundError(f"Không tìm thấy mô hình: {MODEL.resolve()}")
    if not JAVA17.is_file():
        raise FileNotFoundError(f"Không tìm thấy Java 17: {JAVA17}")

    print("[TEST] Khởi động AnyLogic bằng Java 17 / Alpyne 1.2.1", flush=True)
    sim = AnyLogicSim(
        str(MODEL),
        port=PORT,
        java_exe=str(JAVA17),
        max_server_await_time=120,
        auto_lock=True,
        lock_defaults={"timeout": 120},
        engine_overrides={"stop_time": 120, "seed": 43},
    )

    config = build_config(sim)
    print("[TEST] runmode=1, fleetsize=8, sim_seconds=120", flush=True)
    status = sim.reset(**config)
    print("[TEST] Engine state:", status.state, flush=True)

    obs = status.observation
    if obs is None:
        print("[TEST] Chưa có observation; xem model.log và alpyne.log.", flush=True)
        return

    caller = obs.get("caller")
    matrix = obs.get("obs")
    print("[TEST] Caller:", caller, flush=True)
    if matrix is None:
        print("[TEST] Trường 'obs' là None; chưa xác minh dữ liệu Station.", flush=True)
        return

    rows = len(matrix)
    widths = sorted({len(row) for row in matrix}) if rows else []
    print("[TEST] Observation shape: rows=", rows, ", row widths=", widths, sep="", flush=True)
    print("[TEST] Routing cũ: 8 hàng AGV; Routing mới: 8 AGV + N hàng Station.", flush=True)

    if caller is None:
        print("[TEST] Thiếu caller; chưa phân biệt Routing/Dispatching.", flush=True)
    elif int(caller) >= 1000:
        print("[TEST] Observation đầu tiên là Dispatching; chưa xác nhận Routing mới.", flush=True)
    elif rows == FLEETSIZE:
        print("[TEST] Routing chỉ có hàng AGV; cần kiểm tra bản export.", flush=True)
    elif rows > FLEETSIZE:
        station_rows = matrix[FLEETSIZE:]
        nonzero = sum(
            any(abs(float(x)) > 1e-12 for x in row)
            for row in station_rows
        )
        print(f"[TEST] ROUTING CÓ {len(station_rows)} hàng Station!", flush=True)
        print(f"[TEST] Station nonzero: {nonzero}/{len(station_rows)}", flush=True)
        print("[TEST] Station row đầu (tối đa 12 số):",
              list(station_rows[0][:12]), flush=True)
    else:
        print("[TEST] Số hàng thấp hơn fleetsize; cần kiểm tra thêm.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[TEST] Lỗi; chi tiết:", flush=True)
        traceback.print_exc()
        raise
