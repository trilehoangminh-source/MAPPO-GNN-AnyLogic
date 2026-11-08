"""Greedy, 10-seed held-out evaluation of V14 Safe-Yield+Stall reward checkpoint.

No PPO optimization. Results stored in the NEW experiment directory only.
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import os
import random
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from train_v14_safe_yield_stall import (VERSION as SOURCE_VERSION, DEFAULT_OUT,
    REWARD_MODULE,sha256,read_json,SafeYieldConfig,SafeYieldStallReward)
from v14_deep_mlp_common import build_v14

EVAL_VERSION='V14_SAFE_YIELD_STALL_HELDOUT_V1'
SEEDS=list(range(900001,900011))
METRICS=('jobDone','block','liveLock','wrongTarget','targetReached',
         'block_time_s','stop_ratio_pct','yield_bonuses','stall_penalties','reward')


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=DEFAULT_OUT)
    p.add_argument('--checkpoint',type=Path,default=None)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51881)
    p.add_argument('--seeds',type=int,nargs='+',default=SEEDS)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args(argv)
    if not 1<=a.port<=65535 or not a.seeds or len(set(a.seeds))!=len(a.seeds):
        p.error('Invalid port or duplicate/empty seeds')
    return a


def preflight(a):
    run=a.run.resolve();meta=read_json(run/'experiment.json')
    if (meta.get('version')!=SOURCE_VERSION or meta.get('reward_mode')!='direct_environment_safe_yield_stall'
        or meta.get('stop_mask_mode')!='baseline' or int(meta['fleetsize'])!=20
        or int(meta['sim_seconds'])!=3600 or int(meta['gat_layers'])!=4):
        raise RuntimeError('Not a 20-AGV/3600s V14 Safe-Yield run')
    if sha256(REWARD_MODULE)!=meta['reward_implementation_sha256']:
        raise RuntimeError('Reward .py changed since fork initialized')
    cfg_path=run/'run_config_blueprint.json';cfg=read_json(cfg_path)
    if sha256(cfg_path)!=meta['source_blueprint_sha256'] or sha256(Path(cfg['model_file']).resolve())!=meta['anylogic_sha256']:
        raise RuntimeError('AnyLogic blueprint/model changed')
    cp=(a.checkpoint or Path(read_json(run/'state.json')['checkpoint'])).resolve()
    if cp.parent!=run or cp.suffix!='.pt' or not cp.is_file():
        raise RuntimeError('Checkpoint must be an existing .pt INSIDE new V14 run')
    cp_hash=sha256(cp)
    if int(read_json(run/'state.json')['last_episode'])==130 and cp_hash!=meta['source_checkpoint_sha256']:
        raise RuntimeError('Initial EP130 checkpoint changed after forking')
    output=run/f'heldout_safe_yield_{cp.stem}_{cp_hash[:10]}_{len(a.seeds)}seeds_3600s.jsonl'
    return run,meta,cfg,cp,cp_hash,output


def format_row(seed,envdata):
    s=envdata.get('statistics') or {};a=envdata.get('action_diagnostics') or {}
    counts=(a.get('actions_by_role') or {}).get('routing') or {}
    routed=sum(int(v) for v in counts.values());stop=int(counts.get('0',0))
    shaping=envdata.get('reward_shaping') or {}
    def read(k): return float(s[k]) if s.get(k) is not None else None
    return dict(seed=int(seed),jobDone=read('jobDone'),block=read('block'),
                liveLock=read('liveLock'),wrongTarget=read('wrongTarget'),
                targetReached=read('targetReached'),
                block_time_s=sum(float(v or 0) for k,v in s.items() if k.startswith('_blocksec_agv')),
                stop_ratio_pct=100*stop/routed if routed else None,
                stop_count=stop,routing_decisions=routed,
                yield_bonuses=int(shaping.get('yield_proxy_bonuses_applied',0)),
                stall_penalties=int(shaping.get('stall_penalties_applied',0)),
                reward=float(envdata.get('delivered_reward',0)),
                reward_adjustment=float(shaping.get('total_reward_adjustment',0)),
                end_reason=envdata.get('end_reason'),
                reached_terminal=bool(envdata.get('reached_terminal',False)))


def write_summary(path,rows,ordered_seeds):
    if not all(any(int(r['seed'])==s for r in rows) for s in ordered_seeds):
        return
    means={}
    for k in METRICS:
        x=[float(r[k]) for r in rows if r.get(k) is not None]
        means[k+'_mean']=statistics.mean(x) if x else None
        means[k+'_sd']=statistics.stdev(x) if len(x)>1 else None
    summary=path.with_name(path.stem+'_summary.csv')
    with summary.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(means));w.writeheader();w.writerow(means)
    print('[SAFE YIELD EVAL SUMMARY]',{k:means[k+'_mean'] for k in METRICS},flush=True)
    print('[SAFE YIELD EVAL SUMMARY FILE]',summary,flush=True)


def main(argv=None):
    a=parse_args(argv)
    run,meta,cfg,cp,cp_hash,output=preflight(a)
    if a.dry_run:
        print(f'[SAFE YIELD EVAL DRY RUN OK] checkpoint={cp}, sha256={cp_hash}')
        print(f'[SAFE YIELD EVAL DRY RUN OK] 20 AGV, 3600s, {len(a.seeds)} seeds, greedy, output={output}')
        return
    rows=[]
    if output.exists():
        if not a.resume:
            raise FileExistsError('Output exists; use --resume: '+str(output))
        for line in output.read_text(encoding='utf-8').splitlines():
            if line.strip():
                r=json.loads(line)
                if (r.get('eval_version')!=EVAL_VERSION or
                    r.get('checkpoint_sha256')!=cp_hash or
                    int(r.get('seed',-1)) not in a.seeds or
                    any(z['seed']==r['seed'] for z in rows)):
                    raise RuntimeError('Incomplete/invalid/mixed previous evaluation rows')
                rows.append(r)
    import numpy as np
    import torch
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit
    from v14_reward_collector import V14RewardEventCollector
    if str(a.device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    os.environ['MAPPO_DISPATCH_MASK_MODE']='operation_first'
    os.environ['MAPPO_MAX_MINIBATCH']=str(int(meta['max_minibatch']))
    logging.getLogger('alpyne.data').setLevel(logging.ERROR)
    logging.getLogger('alpyne.sim').setLevel(logging.ERROR)
    torch.set_num_threads(int(meta.get('cpu_threads',2)))
    shaper=SafeYieldStallReward(SafeYieldConfig(**meta['reward_config']))
    shaper.install()
    env=None
    try:
        env,model,spec,_=build_v14(cfg,a.device,a.port,3600,int(meta['max_events']),
            gat_layers=int(meta['gat_layers']),gat_heads=int(meta['gat_heads']),
            dispatch_width=int(meta['dispatch_width']),embedding=int(meta['embedding']),
            station_id_dim=int(meta['station_id_dim']),checkpoint=cp)
        auditor=RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        collector=V14RewardEventCollector(model,env,shaper,stop_mask_mode='baseline')
        with output.open('a' if a.resume else 'x',encoding='utf-8') as f:
            for seed in a.seeds:
                if any(r['seed']==seed for r in rows):
                    print('[SKIP] seed',seed);continue
                random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                auditor.reset()
                with torch.no_grad():
                    _,envdata=collector.episode(seed=seed,train=False,max_event_steps=int(meta['max_events']))
                if envdata.get('end_reason')!='environment_terminal' or not envdata.get('reached_terminal'):
                    raise RuntimeError(f'Seed {seed}: incomplete episode')
                if GraphSpec.from_hive(env.agent_hive).fingerprint!=spec.fingerprint:
                    raise RuntimeError('Graph changed during evaluation')
                if auditor.report().get('candidate_map_rate',0)<.99:
                    raise RuntimeError('Graph mapping below 99%')
                t=float(env.run.get_state()[1]['model_time'])
                if t<3599.0:
                    raise RuntimeError(f'Seed {seed}: model time too short ({t})')
                row=format_row(seed,envdata)
                row.update(eval_version=EVAL_VERSION,checkpoint=str(cp),
                           checkpoint_sha256=cp_hash,model_time_s=t,
                           reward_implementation_sha256=meta['reward_implementation_sha256'],
                           evaluation=True,greedy=True,sim_seconds=3600,fleet_size=20)
                f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
                rows.append(row)
                print(f'[SAFE YIELD EVAL seed={seed}] JobDone={row["jobDone"]} Block={row["block"]} '
                      f'LiveLock={row["liveLock"]} Stop%={row["stop_ratio_pct"]:.2f}',flush=True)
    finally:
        if env is not None:
            env.close()
    write_summary(output,rows,a.seeds)
    print('[SAFE YIELD EVAL DONE]',output,flush=True)

if __name__=='__main__':
    main()
