"""V10-A Dispatch diagnostics on identical observations, without PPO updates.

Copy this script to thesis/scripts/diagnose_v10_dispatch.py.
Run it from the thesis root, with no other AnyLogic simulations active.
Compares frozen V8 EP342 Dispatch actors with trained V10 Dispatch actors
on the SAME observations encountered by V10 during heldout execution.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, default=ROOT / "mappo_runs" / "V10_A_FROM_V8")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[900001])
    p.add_argument("--sim-seconds", type=int, default=1800)
    p.add_argument("--device", choices=["cpu", "cuda:0"], default="cuda:0")
    p.add_argument("--port", type=int, default=51881)
    args = p.parse_args()
    if not args.seeds or len(args.seeds) != len(set(args.seeds)):
        p.error("Provide nonempty, unique seeds")

    import torch
    import thesis.mappo.algorithm as algo
    from v10_common import read_json, setup_model, sha256
    from thesis.mappo.routing_guidance_v121 import GuidedEventCollector

    run = args.run.resolve()
    meta = read_json(run / "experiment.json")
    cfg = read_json(run / "v8_source_run_config.json")
    if meta["variant"] != "A":
        raise ValueError(f"Expected V10-A, got {meta['variant']!r}")
    frozen = run / "v8_source_checkpoint.pt"
    if sha256(frozen) != meta["source_sha256"]:
        raise RuntimeError("Frozen V8 SHA256 does not match experiment metadata")
    if args.checkpoint is None:
        cp = Path(read_json(run / "state.json")["checkpoint"]).resolve()
    else:
        cp = args.checkpoint.resolve()
    if not cp.is_file() or cp.parent != run:
        raise FileNotFoundError("Checkpoint must exist inside selected run")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu")
    print(f"[SOURCE V8] {frozen.name} sha256={sha256(frozen)[:16]}", flush=True)
    print(f"[V10] {cp.name} sha256={sha256(cp)[:16]}", flush=True)

    env = None
    try:
        env, model, spec, _ = setup_model(
            cfg, frozen, "A", args.device, args.port, args.sim_seconds,
            int(meta["max_events"]), int(meta["hidden"]), int(meta["heads"]),
            int(meta["layers"]), v10_checkpoint=cp,
        )
        source_data = torch.load(frozen, map_location="cpu", weights_only=False)
        old_actors = {}
        for role in ("dispatch_front", "dispatch_rear"):
            # Deep-copy ONLY the V8 MLP from the trained adapter, then load
            # the frozen V8 source actor (not the V10-updated base weights).
            current = model.roles[role].actor
            if not hasattr(current, "base") or not hasattr(current, "station_head"):
                raise RuntimeError(f"{role}: V10 Transformer adapter is NOT installed")
            original = deepcopy(current.base)
            original.load_state_dict(source_data["roles"][role]["actor"], strict=True)
            original.eval()
            old_actors[role] = original

        original_act = model.act
        current_stats = {}
        counts = defaultdict(dict)

        def record_act(role, obs, state, greedy=False):
            result = original_act(role, obs, state, greedy=greedy)
            if role not in old_actors:
                return result
            s = current_stats[role]
            with torch.inference_mode():
                x = algo.batchify(algo.as_actor(obs, model.device))
                new_actor = model.roles[role].actor
                p_v10 = new_actor.dist(x).probs[0]
                p_v8 = old_actors[role].dist(x).probs[0]
                valid = x["action_mask"][0] > 0.5
                n_valid = int(valid.sum().item())
                if n_valid < 1:
                    raise RuntimeError("No valid Dispatch actions")
                a10 = int(p_v10.argmax().item())
                a8 = int(p_v8.argmax().item())
                if greedy and int(result[0]) != a10:
                    raise RuntimeError("Greedy V10 action mismatch: check evaluation wiring")
                # Total variation = 0.5 sum |p_V10 - p_V8|, range [0,1].
                tv = float(0.5 * (p_v10 - p_v8).abs().sum().item())
                v10_raw = new_actor.logits(x)[0]
                mlp_raw = new_actor.base.policy(new_actor.base.encoder(x))[0]
                correction = (v10_raw - mlp_raw)[valid]
                spread = float((correction.max() - correction.min()).item())

            s["decisions"] += 1
            s["allowed_sum"] += n_valid
            s["forced"] += int(n_valid == 1)
            s["multiple"] += int(n_valid >= 2)
            s["different_argmax"] += int(a10 != a8)
            s["tv_sum"] += tv
            s["tv_max"] = max(s["tv_max"], tv)
            s["correction_spread_sum"] += spread
            s["correction_spread_max"] = max(s["correction_spread_max"], spread)
            if n_valid >= 2:
                s["different_argmax_multi"] += int(a10 != a8)
            return result

        model.act = record_act
        collector = GuidedEventCollector(
            model, env, stop_mask_mode=cfg["stop_mask_mode"],
            routing_guide_strength=1.5,
        )
        for seed in args.seeds:
            current_stats = {role: dict(
                decisions=0, allowed_sum=0, forced=0, multiple=0,
                different_argmax=0, different_argmax_multi=0,
                tv_sum=0., tv_max=0.,
                correction_spread_sum=0., correction_spread_max=0.,
            ) for role in old_actors}
            _, metadata = collector.episode(
                seed=seed, train=False, max_event_steps=int(meta["max_events"])
            )
            print(f"\n===== SEED {seed} | JobDone={metadata.get('statistics',{}).get('jobDone')} =====", flush=True)
            for role, s in current_stats.items():
                n = max(1, s["decisions"])
                print(
                    f"{role}: decisions={s['decisions']} "
                    f"one-valid={s['forced']} ({s['forced']/n:.1%}) "
                    f"multiple-valid={s['multiple']} "
                    f"V8-vs-V10-greedy-different={s['different_argmax']} "
                    f"(among multiple-valid: {s['different_argmax_multi']}) "
                    f"mean-valid-actions={s['allowed_sum']/n:.2f} "
                    f"mean-TV={s['tv_sum']/n:.8f} "
                    f"max-TV={s['tv_max']:.8f} "
                    f"mean-adapter-logit-spread={s['correction_spread_sum']/n:.8f} "
                    f"max-adapter-logit-spread={s['correction_spread_max']:.8f}",
                    flush=True,
                )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
