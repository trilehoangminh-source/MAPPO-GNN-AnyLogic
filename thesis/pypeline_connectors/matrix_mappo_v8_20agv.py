"""AnyLogic Desktop animation: MAPPO V8 Residual Edge-GAT, 20 AGVs.

Only inference (no RL training). AnyLogic's Main.runmode=3 calls get_config
at startup and get_action for every routing/dispatch decision over Pypeline.

Setup: copy the trained checkpoint to
    mappo_runs/V8_RESIDUAL_EDGEGAT_JOBDONE2_FROM205/desktop_brain.pt
and set Main.RL_pypeline_name = "matrix_mappo_v8_20agv".
The checkpoint is checked against the 20-AGV run_config.json and live graph.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ["MAPPO_DISPATCH_MASK_MODE"] = "operation_first"

from thesis.envs.alpyne121_bridge import Observation
from .v7_inference import verify_checkpoint, make_inference_policy, greedy_action

ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = ROOT / "mappo_runs" / "V8_RESIDUAL_EDGEGAT_JOBDONE2_FROM205"
CHECKPOINT = Path(os.environ.get("V8_20AGV_DESKTOP_CHECKPOINT", str(RUN_DIR / "desktop_brain.pt")))
RUN_CONFIG = Path(os.environ.get("V8_20AGV_DESKTOP_RUN_CONFIG", str(RUN_DIR / "run_config.json")))

if not CHECKPOINT.is_file():
    raise FileNotFoundError(f"V8 20AGV checkpoint not found: {CHECKPOINT}; copy it to desktop_brain.pt")
if not RUN_CONFIG.is_file():
    raise FileNotFoundError(f"V8 20AGV run_config.json not found: {RUN_CONFIG}")

_config, _checkpoint_data = verify_checkpoint(CHECKPOINT, RUN_CONFIG)
if int(_config["fleetsize"]) != 20 or int(_config["max_fleetsize"]) != 20:
    raise RuntimeError("Expected a V8 trained 20-AGV run_config.json (fleet=20, max=20)")

_env = None
_model = None
_collector = None
_call_count = 0


def _make_model_free_environment():
    """Build Tobias converters WITHOUT launching an external Alpyne Java server."""
    from thesis.envs.matrix_v121 import Matrix
    from scripts.train_mappo_long_B3_v121 import SIM_CONFIG

    sim_config = SIM_CONFIG.copy()
    # Record the V8 JD2 experiment's reward value; AnyLogic GUI also needs
    # Main.RL_reward_part_completed=2 if reward statistics should match.
    sim_config["reward_part_completed"] = 2.0
    original_reset = Matrix.reset
    Matrix.reset = lambda self, *args, **kwargs: None
    try:
        env = Matrix(
            model_path=None,
            fleetsize=20,
            max_fleetsize=20,
            sim_config=sim_config,
            max_seconds=None,
            pseudo_routing=False,
            pseudo_dispatcher=False,
            routing_agent_death=True,
            death_on_target=False,
            transform_dispatching_partobs=True,
            direction_reward=-0.1,
        )
    finally:
        Matrix.reset = original_reset
    env._mappo_station_cache = None
    env._mappo_station_source = "missing"
    return env


def _env_instance():
    global _env
    if _env is None:
        _env = _make_model_free_environment()
    return _env


def get_config(key: str):
    config = _env_instance().config
    if key in config.names():
        return config.values()[config.names().index(key)]
    raise KeyError(f"Pypeline unknown configuration key: {key}")


def _ensure_model(env):
    global _model, _collector
    if _model is not None:
        return
    from thesis.mappo.routing_guidance_v121 import GuidedEventCollector
    # Strict checkpoint graph fingerprint verification happens HERE, after
    # AnyLogic has supplied the first live networkcontext packet.
    _model = make_inference_policy(CHECKPOINT, RUN_CONFIG, env.agent_hive)
    _collector = GuidedEventCollector(
        _model,
        env,
        stop_mask_mode=_config["stop_mask_mode"],
        routing_guide_strength=1.5,  # V8 train/eval use 1.5, NOT legacy V7 metadata
    )
    print(f"[V8 20AGV ANIMATION] LOADED: {CHECKPOINT} | fleet=20 | Residual Edge-GAT | Dijkstra=1.5", flush=True)


def get_action(observation, caller, n_nodes, context, statTitles, statValues, time):
    global _call_count
    env = _env_instance()
    packet = Observation({
        "obs": observation,
        "caller": int(caller),
        "n_nodes": n_nodes,
        "networkcontext": context,
        "rew": 0,
        "statTitles": statTitles,
        "statValues": statValues,
    })
    agent = env._agent_selection_fn(packet)
    if agent == "dummy":
        raise RuntimeError("Expected MARL per-AGV caller, got dummy central agent")
    behavior = env.agent_behaviors[agent]
    # Pull graph and current workstation statistics from AnyLogic's packet.
    behavior.is_for_learning(packet)
    obs, reward, done, info = behavior.convert_from_observation(packet)
    env.agent_selection = agent
    env._save_observation(packet, agent, obs, reward, done, info)
    _ensure_model(env)

    physical = int(caller) - 1000 if int(caller) >= 1000 else int(caller)
    role = "routing" if int(caller) < 1000 else ("dispatch_front" if physical % 2 == 0 else "dispatch_rear")
    prepared = _collector._prepare(obs, role)
    action = greedy_action(_model, role, prepared)
    result = behavior.convert_to_action(action, int(caller))
    fields = result.fields
    receiver = int(fields["receiver"])
    actions = [int(x) for x in fields["actions"]]
    if len(actions) != 1 or receiver != int(caller) or any(x < 0 for x in actions):
        raise RuntimeError(f"Invalid action from V8 MAPPO: {fields!r}")

    # This exact textual format is parsed by Main.take_action_pypeline().
    java_text = f"Action(actions={actions}, receiver={receiver})"
    _call_count += 1
    if _call_count <= 15 or _call_count % 300 == 0:
        print(f"[V8 20AGV ANIMATION] t={time} caller={caller} role={role} action={action} count={_call_count}", flush=True)
    return java_text
