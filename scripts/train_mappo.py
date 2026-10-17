
"""Run event-driven CTDE MAPPO on Tobias's Matrix AnyLogic model.

The baseline DQN+PPO notebooks remain unchanged.
This version includes a startup retry for Alpyne 0.2.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import logging
from urllib.error import HTTPError, URLError
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from thesis.mappo import MAPPO, MAPPOConfig, EventCollector


# Original combined-training environment configuration
SIM_CONFIG = dict(
    dispatch=True,
    routing_ma=True,
    dispatching_ma=True,
    reward_reached_target=10,
    reward_wrong_target=-1,
    reward_removed_for_block=-20,
    reward_target_distance=0,
    reward_invalid=-0.1,
    reward_duration=-0.5,
    reward_pass=0.2,
    block_timeout=60,
    station_separate=False,
    reward_accepted_in_station=0.5,
    reward_declined_in_station=-0.25,
    reward_geo_operation=0.25,
    reward_reduce=-0.005,
    routing_interval=2,
    dispatching_interval=360,
    io_quote=0.9,
    availability=0.9,
    mttr=5,
    fixed_fleets=True,
    remove_all_blocked=False,
)


# ============================================================
# FIX: ALPYNE STARTUP RETRY
# ============================================================

def enable_alpyne_startup_retry(timeout_seconds=180):
    """Wait until the Java API becomes ready.

    This changes Alpyne startup handling only in this Python
    process. The installed Alpyne package is not modified.
    """
    import alpyne.client.alpyne_client as client_module

    AlpyneClient = client_module.AlpyneClient

    if getattr(
        AlpyneClient,
        "_mappo_startup_retry_enabled",
        False
    ):
        return

    def init_with_retry(
        self,
        model_loc,
        blocking=False,
        port=51150,
        verbose=False
    ):
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.WARNING,
            format=(
                "%(asctime)s [%(name)s @ %(lineno)s]"
                "[%(levelname)8s] %(message)s"
            ),
            handlers=[logging.StreamHandler()],
        )

        self.log = logging.getLogger(client_module.__name__)
        self._blocking = blocking
        self._verbose = verbose

        print(
            f"[Alpyne] Starting Java model on port {port}...",
            flush=True
        )

        self._proc = self._start_app(
            model_loc,
            blocking,
            port,
            verbose
        )

        self._http_client = client_module.HttpClient(
            f"http://127.0.0.1:{port}"
        )

        deadline = time.monotonic() + timeout_seconds
        next_message = time.monotonic()
        last_error = None

        while time.monotonic() < deadline:

            # Detect an early Java process failure
            if self._proc.poll() is not None:
                error_text = ""

                if self._proc.stderr is not None:
                    raw = self._proc.stderr.read()

                    if isinstance(raw, bytes):
                        error_text = raw.decode(
                            errors="replace"
                        )
                    else:
                        error_text = str(raw)

                raise RuntimeError(
                    "Alpyne Java process exited early "
                    f"(code={self._proc.returncode}). "
                    f"Java stderr: {error_text[-2000:]}"
                )

            # Try connecting to the Alpyne HTTP API
            try:
                _, _, output = self._http_client.get(
                    "/versions/number/0"
                )

            except HTTPError as exc:
                raise RuntimeError(
                    f"Alpyne API responded with HTTP "
                    f"{exc.code} on port {port}."
                ) from exc

            except (
                URLError,
                OSError,
                TimeoutError,
                ConnectionError
            ) as exc:
                last_error = exc
                now = time.monotonic()

                if now >= next_message:
                    print(
                        f"[Alpyne] Waiting for API "
                        f"on port {port}: {exc}",
                        flush=True
                    )
                    next_message = now + 10

                time.sleep(1)
                continue

            # API is reachable; verify its model version
            try:
                self.version = (
                    client_module.ModelVersion.from_json(
                        output
                    )
                )

            except Exception as exc:
                raise RuntimeError(
                    "Alpyne API is reachable, but its "
                    "model-version response could not "
                    "be parsed. Check Matrix.zip."
                ) from exc

            print(
                "[Alpyne] Server API ready!",
                flush=True
            )
            return

        # Stop Java on startup timeout
        if self._proc.poll() is None:
            self._proc.terminate()

        raise TimeoutError(
            f"Alpyne API did not become ready on "
            f"port {port} within {timeout_seconds} "
            f"seconds. Last error: {last_error}"
        )

    AlpyneClient.__init__ = init_with_retry
    AlpyneClient._mappo_startup_retry_enabled = True


# ============================================================
# ENVIRONMENT
# ============================================================

def build_env(args):

    # Apply the startup fix BEFORE Matrix creates AlpyneClient
    enable_alpyne_startup_retry(args.startup_timeout)

    from thesis.envs.matrix import Matrix

    return Matrix(
        model_path=str(Path(args.model).resolve()),
        startport=args.port,
        fleetsize=args.fleetsize,
        fleetsize_upper=None,
        max_fleetsize=args.max_fleetsize,
        sim_config=SIM_CONFIG.copy(),
        max_seconds=args.sim_seconds,
        max_steps=args.max_events,
        pseudo_routing=False,
        pseudo_dispatcher=False,
        routing_agent_death=True,
        death_on_target=False,
        transform_dispatching_partobs=True,
        direction_reward=-0.1,
    )


# ============================================================
# BUILD MAPPO MODEL
# ============================================================

def make_model(env, args):

    from thesis.mappo.algorithm import MAPPO

    cfg = MAPPOConfig(
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip,
        actor_lr=args.lr,
        critic_lr=args.lr,
        epochs=args.epochs,
        mini_batches=args.mini_batches,
        hidden=args.hidden,
        env_type=args.env_type,
        routing_backbone=args.routing_backbone,
        dispatch_backbone=args.dispatch_backbone,
        two_fleets=not args.single_dispatch_policy,
    )

    env.reset(seed=args.seed)

    sample = env.last()[0]

    obs_space = env.observation_space(
        env.agent_selection
    )

    critic_state = env.get_mappo_state()

    n_station = critic_state["stat"].shape[0]
    feat = critic_state["agvs"].shape[1]

    roles = {
        "routing": (obs_space, 5)
    }

    if cfg.two_fleets:
        roles["dispatch_front"] = (
            obs_space,
            n_station
        )
        roles["dispatch_rear"] = (
            obs_space,
            n_station
        )
    else:
        roles["dispatching"] = (
            obs_space,
            n_station
        )

    return MAPPO(
        roles,
        feat,
        cfg,
        device=args.device
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=str(ROOT / "envs" / "Matrix.zip")
    )

    parser.add_argument(
        "--env-type",
        default="matrix",
        choices=["matrix", "minimatrix"]
    )

    parser.add_argument(
        "--port",
        type=int,
        default=51150
    )

    parser.add_argument(
        "--startup-timeout",
        type=int,
        default=180
    )

    parser.add_argument(
        "--fleetsize",
        type=int,
        default=8
    )

    parser.add_argument(
        "--max-fleetsize",
        type=int,
        default=20
    )

    parser.add_argument(
        "--sim-seconds",
        type=int,
        default=3600
    )

    parser.add_argument(
        "--max-events",
        type=int,
        default=20000
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=30
    )

    parser.add_argument(
        "--episodes-per-update",
        type=int,
        default=1
    )

    parser.add_argument(
        "--routing-backbone",
        choices=["original_gnn", "pooled"],
        default="original_gnn"
    )

    parser.add_argument(
        "--dispatch-backbone",
        choices=["original_linear", "pooled"],
        default="original_linear"
    )

    parser.add_argument(
        "--single-dispatch-policy",
        action="store_true"
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.90
    )

    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95
    )

    parser.add_argument(
        "--clip",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5
    )

    parser.add_argument(
        "--mini-batches",
        type=int,
        default=1
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--hidden",
        type=int,
        default=32
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=43
    )

    parser.add_argument(
        "--device",
        default="cpu"
    )

    parser.add_argument(
        "--output",
        default=str(
            ROOT / "mappo_runs" / "trial_01"
        )
    )

    parser.add_argument(
        "--resume",
        default=None
    )

    args = parser.parse_args()

    if args.fleetsize > args.max_fleetsize:
        parser.error(
            "fleetsize must be <= max-fleetsize"
        )

    if args.startup_timeout < 1:
        parser.error(
            "startup-timeout must be positive"
        )

    if args.episodes_per_update < 1:
        parser.error(
            "episodes-per-update must be positive"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out = Path(args.output)
    out.mkdir(
        parents=True,
        exist_ok=True
    )

    env = None

    try:
        print(
            "[MAPPO] Creating environment...",
            flush=True
        )

        env = build_env(args)

        print(
            "[MAPPO] Environment initialized. "
            "Building networks...",
            flush=True
        )

        model = make_model(env, args)

        print(
            "[MAPPO] Networks ready. "
            "Collecting rollout...",
            flush=True
        )

        if args.resume:
            model.load(args.resume)

        collector = EventCollector(
            model,
            env
        )

        trajectories = []

        with (
            out / "metrics.jsonl"
        ).open(
            "a",
            encoding="utf-8"
        ) as logfile:

            for e in range(args.episodes):

                pieces, meta = collector.episode(
                    seed=args.seed + e,
                    train=True,
                    max_event_steps=args.max_events
                )

                trajectories.extend(pieces)
                update = {}

                if (
                    (e + 1) %
                    args.episodes_per_update
                ) == 0:

                    update = model.train(
                        trajectories
                    )

                    trajectories = []

                    model.save(
                        out / "latest_mappo.pt"
                    )

                log = dict(
                    episode=e + 1,
                    environment=meta,
                    policy_updates=update
                )

                logfile.write(
                    json.dumps(
                        log,
                        default=str
                    ) + "\n"
                )

                logfile.flush()

                print(
                    f'Episode {e + 1}: '
                    f'transitions={meta["closed_transitions"]}, '
                    f'pending={meta["discarded_pending"]}, '
                    f'events={meta["events"]}, '
                    f'training={list(update)}',
                    flush=True
                )

        if trajectories:

            model.train(
                trajectories
            )

            model.save(
                out / "latest_mappo.pt"
            )

    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
