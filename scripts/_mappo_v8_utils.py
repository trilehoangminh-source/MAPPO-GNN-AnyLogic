"""Pure helpers for safe, resumable V8 overnight MAPPO runs."""
from __future__ import annotations
import json
import math
import os
from pathlib import Path


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, path)


def score_kpis(stats):
    """Heuristic checkpoint *ranking*, NOT a replacement for a held-out evaluation."""
    get = lambda name: float(stats.get(name, 0.0))
    keys = ('jobDone', 'wrongTarget', 'block', 'liveLock')
    if not all(math.isfinite(get(k)) for k in keys):
        raise ValueError('Nonfinite KPI encountered')
    return get('jobDone') - .05*get('wrongTarget') - .01*get('block') - .01*get('liveLock')


def average_report(rows):
    if not rows:
        raise ValueError('Cannot evaluate empty results')
    stats = [r['environment']['statistics'] for r in rows]
    def avg(key): return sum(float(s.get(key, 0)) for s in stats)/len(stats)
    return dict(n=len(rows), jobDone=avg('jobDone'), wrongTarget=avg('wrongTarget'),
                block=avg('block'), liveLock=avg('liveLock'),
                score=sum(score_kpis(s) for s in stats)/len(stats))


def read_jsonl(path):
    path = Path(path)
    if not path.exists(): return []
    lines = [s for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
    return [json.loads(s) for s in lines]


def check_resume_state(state, ckpt_path):
    if not isinstance(state.get('last_episode'), int) or state['last_episode'] < 30:
        raise ValueError('Unexpected last_episode in resume state')
    if not Path(ckpt_path).is_file():
        raise FileNotFoundError('Saved checkpoint missing: '+str(ckpt_path))
    return True


def repair_metrics_on_resume(path, last_episode):
    """Discard only uncommitted trailing log rows after crash; fail on missing committed rows."""
    path = Path(path)
    rows = read_jsonl(path)
    committed = [r for r in rows if int(r["episode"]) <= last_episode]
    ids = [int(r["episode"]) for r in committed]
    if ids and (len(set(ids)) != len(ids) or ids != sorted(ids)):
        raise RuntimeError("Duplicate or unordered metric episodes; inspect manually")
    # A transferred V8 run begins at its SOURCE episode (e.g. 144 for 15->20
    # AGVs). Before its first 20-AGV episode commits, zero local metrics is valid.
    # Do not confuse the transfer boundary with the V7 Episode-30 boundary.
    first_episode = 30
    metadata_path = path.parent / "overnight_config.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        first_episode = int(metadata.get("first_completed_episode", first_episode))
    if last_episode < first_episode:
        raise RuntimeError("Resume state precedes this run's transfer boundary")
    if last_episode == first_episode and ids:
        raise RuntimeError("Unexpected committed metrics at transfer boundary")
    if last_episode > first_episode and (not ids or ids[-1] != last_episode):
        raise RuntimeError("Missing committed metrics for latest checkpoint")
    if len(committed) != len(rows):
        print("[V8 RECOVERY] Removing uncommitted metric tail", len(rows)-len(committed), flush=True)
        txt = "".join(json.dumps(r,ensure_ascii=False,default=str)+"\n" for r in committed)
        tmp = path.with_name(path.name+".tmp")
        tmp.write_text(txt,encoding="utf-8")
        os.replace(tmp,path)
    return len(committed)
