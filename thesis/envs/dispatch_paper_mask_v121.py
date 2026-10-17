"""Paper-inspired, operation-first AGV dispatch action decoder.

Reference inspirations:
 - Li et al. (2025) efficient action decoding / eligibility pruning.
 - Zheng et al. (2026) operation-aware cooperative scheduling.
This is a NEW adaptation to Tobias Matrix, not source code from either paper.

Do not mistake a processing Station candidate for available capacity; capacity-aware
masking requires further validated Station-state semantics.
"""
from __future__ import annotations

import numpy as np


def priority_station_names(info, part):
    """Use Tobias CleverMatrixDispatcher's precedence order:
    rework -> next GEO -> currently possible procedures -> buffer fallback.
    """
    if info is None:  # AGV has no Part: go to inlet GEO nodes for pickup
        return ["vgeo1", "hgeo1"], "pickup_geos"
    if bool(info.get("rework", False)):
        return ["rework"], "rework"
    if info.get("next_geo") is not None:
        side = "v" if str(info.get("variant", "")).startswith("vb") else "h"
        return [f"{side}geo{int(info['next_geo'])}"], "next_geo"
    nodes = []
    for proc, amount in zip(info.get("proc", []), info.get("amount", [])):
        if float(amount) > 0.0:
            nodes.extend(part.proc_stat.get(proc, ()))
    return list(dict.fromkeys(nodes)), ("operations" if nodes else "no_pending_operations")


def build_operation_first_dispatch_mask(info, part, live_station_nodes):
    """Return mask aligned to live hive.stations order and diagnostic metadata.

    Buffers are a conservative fallback if *no stage-eligible station is present*.
    Do NOT claim this checks station load, occupancy, breakdowns, or prerequisites.
    """
    live = list(live_station_nodes)
    if len(set(live)) != len(live):
        raise ValueError("Duplicate node ids in AnyLogic stations")
    names, reason = priority_station_names(info, part)
    candidate = set()
    for name in names:
        node = part.stat_node.get(name)
        if isinstance(node, int):
            candidate.add(node)
        elif isinstance(node, (list, tuple)):
            candidate.update(int(n) for n in node)
    station_idxs = [i for i, node in enumerate(live) if node in candidate]
    mask = np.zeros(len(live), dtype=np.float32)
    if station_idxs:
        mask[station_idxs] = 1.0
        used_buffer_fallback = False
    else:
        buffers = part.stat_node.get("puffer", [])
        if isinstance(buffers, int):
            buffers = [buffers]
        buffer_set = set(buffers)
        fallback_idxs = [i for i, node in enumerate(live) if node in buffer_set]
        if not fallback_idxs:
            raise RuntimeError(
                f"No station or buffer fallback: reason={reason} names={names} live_nodes={live}"
            )
        mask[fallback_idxs] = 1.0
        used_buffer_fallback = True
    meta = {
        "reason": reason,
        "station_names": names,
        "allowed_actions": np.flatnonzero(mask).tolist(),
        "buffer_fallback": used_buffer_fallback,
        "part_present": info is not None,
    }
    return mask, meta
