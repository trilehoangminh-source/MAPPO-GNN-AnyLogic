"""Read-only inspection of the exact dispatch action -> AnyLogic node mapping.

Works with the existing Alpyne 1.2.1 MAPPO project's train_mappo_stop_ab.py.
No checkpoint is changed and no RL actions are sent.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from types import SimpleNamespace

import train_mappo_stop_ab as training
from thesis.envs.part_variants import MatrixPart


def resolve_names():
    variants = MatrixPart().stat_node
    result = {}
    for name, value in variants.items():
        for node in value if isinstance(value, (tuple, list)) else (value,):
            if int(node) in result:
                raise ValueError(f"Duplicated node {node} in MatrixPart.stat_node")
            result[int(node)] = name
    return result


def print_dispatch_map(stations, name_for_node):
    if not stations:
        raise RuntimeError("No stations in HiveContext; cannot interpret dispatch indices")
    print("\n== LIVE DISPATCH ACTION MAPPING ==", flush=True)
    print(f"Stations marked by networkcontext: {len(stations)}", flush=True)
    unknown = []
    for action_index, (node_id, coords) in enumerate(stations):
        name = name_for_node.get(int(node_id), "UNKNOWN")
        if name == "UNKNOWN":
            unknown.append(node_id)
        focus = "  <=== CHOSEN BY B GREEDY" if action_index in (21, 22) else ""
        print(f"action={action_index:02d} -> node={node_id:03d} -> "
              f"{name:>10s} -> coords={coords}{focus}", flush=True)
    print("\n== CHECKS ==", flush=True)
    if len(stations) > 22:
        for ix in (21, 22):
            node, xy = stations[ix]
            print(f"action[{ix}] => node {node}, name={name_for_node.get(int(node),'UNKNOWN')}, "
                  f"coordinate={xy}", flush=True)
    else:
        print("Unexpected station count: indices 21/22 might be out of range", flush=True)
    print("Expected station nodes from MatrixPart:", len(name_for_node), flush=True)
    print("Unknown live nodes:", unknown, flush=True)
    diff = sorted(set(name_for_node) ^ {int(k) for k, _ in stations})
    print("Difference vs MatrixPart node inventory:", diff, flush=True)
    if diff:
        print("WARNING: Static MatrixPart mapping differs from the live AnyLogic export. "
              "Do not trust static action-name labels until corrected!", flush=True)
    else:
        print("OK: Live nodes match MatrixPart static inventory", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--port', type=int, default=51650)
    p.add_argument('--seed', type=int, default=1001)
    p.add_argument('--model', type=Path, default=Path('envs/Matrix_critic_fresh.zip'))
    p.add_argument('--startup-timeout', type=int, default=180)
    p.add_argument('--lock-timeout', type=int, default=180)
    p.add_argument('--java-exe', type=str,
                   default=r'C:\Program Files\AnyLogic 8.9 Personal Learning Edition\jre\bin\java.exe')
    args = p.parse_args()
    if not args.model.is_file():
        p.error(f'Cannot find model: {args.model}')
    runtime = SimpleNamespace(
        model=str(args.model.resolve()), port=args.port, seed=args.seed,
        startup_timeout=args.startup_timeout, lock_timeout=args.lock_timeout,
        java_exe=args.java_exe, fleetsize=8, max_fleetsize=20,
        sim_seconds=120, max_events=20000,
    )
    env = None
    try:
        # build_env initializes the model and collects its first observation.
        env = training.build_env(runtime)
        if env.agent_hive.stations is None:
            # The first event may lack networkcontext; a regular reset requests it.
            env.reset(seed=args.seed)
        if env.agent_hive.stations is None:
            raise RuntimeError("Hive stations still unavailable after reset. "
                               "Check the AnyLogic networkcontext on the first event.")
        print_dispatch_map(list(env.agent_hive.stations.items()), resolve_names())
    finally:
        if env is not None:
            env.close()


if __name__ == '__main__':
    main()
