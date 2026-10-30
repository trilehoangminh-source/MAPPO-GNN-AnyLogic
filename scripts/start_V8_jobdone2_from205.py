"""Create a separate V8 EP205 -> Reward+2 run and launch the V8 trainer.

Usage (from thesis folder):
  python -u scripts/start_V8_jobdone2_from205.py --hours 5 --device cuda:0
  python -u scripts/start_V8_jobdone2_from205.py --resume --hours 5 --device cuda:0

Never modifies the source V8_DEBUG_EP199 run or the on-disk reward settings.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'mappo_runs'
SRC = RUNS / 'V8_DEBUG_EP199'
DST = RUNS / 'V8_RESIDUAL_EDGEGAT_JOBDONE2_FROM205'

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024*1024), b''):
            h.update(part)
    return h.hexdigest()

def preflight():
    if not SRC.is_dir():
        raise RuntimeError(f'Cannot find source EP205 run: {SRC}')
    for n in ('state.json', 'overnight_config.json', 'run_config.json', 'metrics.jsonl'):
        if not (SRC / n).is_file():
            raise RuntimeError(f'Missing V8 source file: {SRC / n}')
    state, meta, cfg = (read_json(SRC / name) for name in ('state.json', 'overnight_config.json', 'run_config.json'))
    if state.get('last_episode') != 205:
        raise RuntimeError(f'Source run must still be EP205; got {state.get("last_episode")}. Do not edit original run.')
    if int(meta.get('first_completed_episode', -1)) != 144 or int(meta.get('fleetsize', -1)) != 20:
        raise RuntimeError('Unexpected source V8 transfer boundary or fleet size')
    if cfg.get('routing_encoder') != 'residual_edge_gat' or cfg.get('routing_backbone') != 'directed_graph_gnn':
        raise RuntimeError('Source is NOT residual Edge-GAT')
    if int(cfg.get('graph_nodes', 0)) != 105 or int(cfg.get('graph_edges', 0)) != 328:
        raise RuntimeError('Unexpected source V8 graph topology')
    model = Path(meta['model_file']).resolve()
    if not model.is_file() or model != Path(cfg['model_file']).resolve():
        raise RuntimeError('AnyLogic ZIP not found / not consistent with run_config.json')
    cp = Path(state['checkpoint']).resolve()
    if cp.parent != SRC.resolve() or not cp.is_file():
        raise RuntimeError(f'EP205 checkpoint missing or not local to source: {cp}')
    prior_cp = Path(meta['source_checkpoint']).resolve()
    prior_run = prior_cp.parent
    if not prior_cp.is_file() or not (prior_run / 'state.json').is_file() or not (prior_run / 'overnight_config.json').is_file() or not (prior_run / 'run_config.json').is_file():
        raise RuntimeError('Source 15-AGV transfer run missing (possibly moved during cleanup): '+str(prior_run)+'\nRestore it from archive before starting.')
    prev_state = read_json(prior_run / 'state.json')
    if Path(prev_state['checkpoint']).resolve() != prior_cp or int(prev_state['last_episode']) != 144:
        raise RuntimeError('15-AGV transfer checkpoint changed since original V8 run; refusing unsafe resume')
    cfg_v7 = Path(meta['source_config']).resolve()
    if not cfg_v7.is_file():
        raise RuntimeError('Original V7 architecture configuration missing: '+str(cfg_v7))
    # If available, verify that today's AnyLogic ZIP equals a saved EP205 ZIP.
    backups = sorted(RUNS.glob('BACKUP_EP205_*/Matrix_RL_debug.zip'))
    if backups:
        if sha256(backups[-1]) != sha256(model):
            raise RuntimeError('AnyLogic ZIP differs from the EP205 backup. Restore the matching model ZIP first.')
        print('[CHECK] AnyLogic ZIP matches saved EP205 model SHA256', flush=True)
    else:
        print('[WARNING] No BACKUP_EP205_*/Matrix_RL_debug.zip found; cannot prove ZIP byte-identical to EP205.', flush=True)
    return state, meta, model, prior_run, cfg_v7

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hours', type=float, default=5)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--resume', action='store_true', help='Continue existing Reward+2 run; do not copy EP205 again')
    p.add_argument('--max-minibatch', type=int, default=0, help='0 preserves V8 PPO update; 128 if CUDA OOM')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if not 0 < args.hours <= 72:
        p.error('hours must be between 0 and 72')
    if args.max_minibatch < 0:
        p.error('max-minibatch must be 0 or positive')
    _, meta, model, prior_run, cfg_v7 = preflight()
    state_path = DST / 'state.json'
    if args.resume:
        if not DST.is_dir() or not state_path.is_file():
            raise RuntimeError('No Reward+2 run to resume: '+str(DST))
        dest_state=read_json(state_path)
        if dest_state['last_episode'] < 205 or not Path(dest_state['checkpoint']).is_file():
            raise RuntimeError('Reward+2 checkpoint missing or invalid')
        tag=read_json(DST/'reward_ablation.json')
        if tag.get('reward_part_completed') != 2.0 or tag.get('encoder') != 'residual_edge_gat':
            raise RuntimeError('Reward+2 provenance marker mismatched')
    else:
        if DST.exists():
            raise RuntimeError(f'Target already exists, refusing to overwrite: {DST}\nUse --resume if this is your Reward+2 run.')
        print('[PLAN] Clone EP205 to',DST,flush=True)
    print('[PLAN] Residual Edge-GAT | 20 AGVs | reward JobDone=2.0 | fresh from EP205 checkpoint (NOT V9)',flush=True)
    print('[PLAN] Run time:',args.hours,'hours | device:',args.device,'| original EP205 will be left unchanged',flush=True)
    command = [sys.executable,'-u',str(ROOT/'scripts'/'train_V8_jobdone2_quiet.py'),
       '--resume', '--from-previous-run',str(prior_run), '--from-config',str(cfg_v7),
       '--output',str(DST), '--model',str(model), '--fleetsize','20',
       '--device',args.device,'--hours',str(args.hours),'--max-additional-episodes','500',
       '--train-sim-seconds',str(meta['train_sim_seconds']), '--max-events',str(meta['max_events']),
       '--eval-seeds',*[str(i) for i in meta['eval_seeds']],
       '--eval-sim-seconds',str(meta['eval_sim_seconds']),
       '--gamma',str(meta['gamma']), '--eval-reserve-minutes','0', '--skip-final-eval']
    if args.dry_run:
        print(' '.join(command),flush=True)
        print('[DRY RUN] No files written.',flush=True)
        return
    if not args.resume:
        shutil.copytree(SRC, DST)
        newstate=read_json(state_path)
        newstate['checkpoint'] = str(DST / Path(newstate['checkpoint']).name)
        newstate['snapshots'] = [str(DST / Path(x).name) for x in newstate.get('snapshots', [])]
        missing = [x for x in [newstate['checkpoint'],*newstate['snapshots']] if not Path(x).is_file()]
        if missing:
            raise RuntimeError('Copied run has missing checkpoints: '+str(missing))
        state_path.write_text(json.dumps(newstate,ensure_ascii=False,indent=2),encoding='utf-8')
        (DST/'reward_ablation.json').write_text(json.dumps({
           'name':'V8_Residual_EdgeGAT_JobDone2_from_EP205', 'source_episode':205,
           'reward_part_completed':2.0, 'encoder':'residual_edge_gat',
           'note':'Reward override occurs in train_V8_jobdone2_quiet.py before importing the V8 trainer; evaluate with evaluate_V8_jobdone2.py',
           'source_model_sha256':sha256(model),
        },indent=2),encoding='utf-8')
    env = os.environ.copy()
    # 0 preserves original V8 PPO minibatch dynamics; override to 128 on CUDA OOM.
    env['MAPPO_MAX_MINIBATCH'] = str(args.max_minibatch)
    print('[CHECK] MAPPO_MAX_MINIBATCH='+str(args.max_minibatch)+' (0=original V8 PPO)',flush=True)
    print('[RUN] Starting 5h budget train. Stop with Ctrl+C; each committed episode remains saved.',flush=True)
    code = subprocess.call(command,cwd=str(ROOT),env=env)
    if code:
        raise SystemExit(code)
    result=read_json(state_path)
    print('[DONE] Last committed episode:',result['last_episode'],flush=True)
    print('[DONE] checkpoint:',result['checkpoint'],flush=True)
    print('[DONE] run folder:',DST,flush=True)

if __name__=='__main__':
    main()
