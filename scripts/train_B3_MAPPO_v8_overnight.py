"""Continue a trusted MAPPO V7 model for a real-clock budget, then select a demo checkpoint.

No changes to Edge-GAT or B2 Dispatching. Uses V7 PPO clipped Huber, KL guard,
on-policy trajectory audit. Longer simulation horizons expose completed jobs.
"""
from __future__ import annotations
import argparse
import json
import os
os.environ['MAPPO_DISPATCH_MASK_MODE'] = 'operation_first'
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
from _mappo_v8_utils import atomic_json, average_report, read_jsonl, check_resume_state, repair_metrics_on_resume

VERSION = 'B3_MAPPO_V8_OVERNIGHT'


def options(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--from-checkpoint', type=Path, default=ROOT/'mappo_runs/V7_brain30_seed43/checkpoint_ep_0030.pt')
    p.add_argument('--from-config', type=Path, default=ROOT/'mappo_runs/V7_brain30_seed43/run_config.json')
    p.add_argument('--from-previous-run', type=Path, default=None, help='Prior V8 run (8 or 15 AGVs). Transfer committed rolling checkpoint, preserve episode count; write a NEW output folder.')
    p.add_argument('--model', type=Path, default=None, help='Override AnyLogic ZIP only when identical topology')
    p.add_argument('--output', type=Path, default=ROOT/'mappo_runs/V8_overnight_fromV7_5h')
    p.add_argument('--hours', type=float, default=5)
    p.add_argument('--max-additional-episodes', type=int, default=500)
    p.add_argument('--train-sim-seconds', type=int, default=1800)
    p.add_argument('--max-events', type=int, default=None, help='Optional explicit ceiling on environment events per episode; default source run config.')
    p.add_argument('--fleetsize', type=int, default=None, help='Target AGV count. For 15->20 fine-tuning use --fleetsize 20 (max_fleetsize=20).')
    p.add_argument('--device', default='cpu', help='cpu or cuda:0. CUDA is strict (no silent CPU fallback).')
    p.add_argument('--cpu-threads', type=int, default=2, help='PyTorch CPU thread cap; AnyLogic/Java still uses CPU.')
    p.add_argument('--port-train', type=int, default=51880)
    p.add_argument('--port-eval', type=int, default=51881)
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[3201,3202])
    p.add_argument('--eval-sim-seconds', type=int, default=3600)
    p.add_argument('--eval-reserve-minutes', type=int, default=35)
    p.add_argument('--snapshot-every', type=int, default=10)
    p.add_argument('--gamma', type=float, default=None, help='Optional experimental change; default preserves V7 gamma=0.9')
    p.add_argument('--resume', action='store_true', help='Resume only from confirmed complete V8 episode')
    p.add_argument('--skip-final-eval', action='store_true', help='Keep checkpoints; evaluate manually later')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if not 0 < a.hours <= 72 or a.max_additional_episodes < 1 or a.train_sim_seconds < 1 or a.eval_sim_seconds < 1:
        p.error('Invalid hours/episodes/sim seconds')
    if not 0 <= a.eval_reserve_minutes < a.hours*60:
        p.error('Invalid evaluation reserve')
    if a.snapshot_every < 1 or not a.eval_seeds or len(set(a.eval_seeds)) != len(a.eval_seeds):
        p.error('Invalid snapshots/evaluation seeds')
    if a.gamma is not None and not 0 < a.gamma < 1:
        p.error('gamma must be strictly between 0 and 1')
    if a.port_eval == a.port_train: p.error('Use distinct train and eval ports')
    if a.max_events is not None and a.max_events < 1: p.error('--max-events must be positive')
    if a.device != 'cpu' and not (a.device.startswith('cuda:') and a.device[5:].isdigit()):
        p.error('--device must be cpu or cuda:N (e.g. cuda:0)')
    if not 1 <= a.cpu_threads <= 32:
        p.error('--cpu-threads must be 1..32')
    return a


def load_source(a):
    """Resolve transferred V8 checkpoint AND verify its originating fleet/graph.

    The model architecture is loaded from the trusted V7 config, while the
    transferred V8 checkpoint may have already adapted from 8 to 15 vehicles.
    """
    a.source_last_episode = None
    a.transfer_source_fleetsize = None
    prev_meta = None
    prev_cfg = None
    if a.from_previous_run is not None:
        previous_folder = a.from_previous_run.resolve()
        if previous_folder == a.output.resolve():
            raise ValueError('Transfer source and output folder must be different')
        state_path = previous_folder/'state.json'
        meta_path = previous_folder/'overnight_config.json'
        run_config_path = previous_folder/'run_config.json'
        for required in (state_path,meta_path,run_config_path):
            if not required.is_file():
                raise FileNotFoundError('Prior V8 run missing: '+str(required))
        prev_state = json.loads(state_path.read_text(encoding='utf-8-sig'))
        cp = Path(prev_state['checkpoint']).resolve()
        check_resume_state(prev_state,cp)
        if cp.parent != previous_folder:
            raise ValueError('Previous committed rolling checkpoint must reside in prior run folder')
        if cp.name not in ('rolling_slot_0.pt','rolling_slot_1.pt'):
            raise ValueError('Transfer must use a committed rolling checkpoint')
        prev_meta = json.loads(meta_path.read_text(encoding='utf-8-sig'))
        prev_cfg = json.loads(run_config_path.read_text(encoding='utf-8-sig'))
        previous_fleet = int(prev_meta['fleetsize'])
        if previous_fleet != int(prev_cfg['fleetsize']):
            raise ValueError('Previous V8 run has mismatched fleet counts in its metadata and eval config')
        if previous_fleet not in (8,15):
            raise ValueError('Only supported transfers: V8 8->15 or V8 15->20')
        if prev_meta.get('version') != VERSION:
            raise ValueError('Prior run is not a compatible V8 overnight run')
        a.from_checkpoint = cp
        a.source_last_episode = int(prev_state['last_episode'])
        a.transfer_source_fleetsize = previous_fleet
        print(f'[V8] Transfer checkpoint: {cp} from {previous_fleet} AGVs, committed episode {a.source_last_episode}',flush=True)

    if not a.from_checkpoint.is_file() or not a.from_config.is_file():
        raise FileNotFoundError('Checkpoint and V7 architecture config are required')
    cfg = json.loads(a.from_config.read_text(encoding='utf-8-sig'))
    if cfg.get('version') != 'B3_MAPPO_STABILITY_V7' or cfg.get('routing_encoder') != 'residual_edge_gat':
        raise ValueError('This runner continues only the V7 stable Edge-GAT architecture')
    if cfg.get('ppo_v7_mode') != 'stable':
        raise ValueError('Expected stable V7 PPO settings')
    if cfg.get('guide_start') != 1.5 or cfg.get('guide_final') != 1.5:
        raise ValueError('Dijkstra strength must remain 1.5')
    if cfg.get('graph_nodes') != 105 or cfg.get('graph_edges') != 328:
        raise ValueError('Unexpected graph node/edge count')

    if prev_meta is not None:
        if str(prev_meta.get('graph_fingerprint')) != str(cfg['graph_fingerprint']):
            raise ValueError('Transferred checkpoint graph fingerprint mismatches V7 architecture')
        if int(prev_meta.get('max_fleetsize',-1)) != int(cfg['max_fleetsize']):
            raise ValueError('Transferred checkpoint max fleet size is different')
        if float(prev_meta.get('guide_strength',-1)) != 1.5:
            raise ValueError('Transferred checkpoint has different guide strength')
        if abs(float(prev_meta.get('gamma',0)) - float(cfg['gamma'])) > 1e-8:
            raise ValueError('Transferred checkpoint gamma differs from base config')
        if int(prev_cfg.get('graph_nodes',-1)) != int(cfg['graph_nodes']) or int(prev_cfg.get('graph_edges',-1)) != int(cfg['graph_edges']):
            raise ValueError('Previous run evaluation graph size differs')

    model_path = a.model or Path(cfg['model_file'])
    if not model_path.is_file():
        raise FileNotFoundError('AnyLogic model ZIP missing: '+str(model_path))
    a.model = model_path
    a.fleetsize = int(cfg['fleetsize']) if a.fleetsize is None else int(a.fleetsize)
    a.transfer_source_fleetsize = int(cfg['fleetsize']) if a.transfer_source_fleetsize is None else a.transfer_source_fleetsize
    if not 1 <= a.fleetsize <= int(cfg['max_fleetsize']):
        raise ValueError(f'--fleetsize must be between 1 and {cfg["max_fleetsize"]}')
    if prev_meta is not None and a.fleetsize <= a.transfer_source_fleetsize:
        raise ValueError('Transfer to a different/higher fleet size, not same or lower; resume same fleet via --resume')
    a.max_events = int(cfg['max_events']) if a.max_events is None else int(a.max_events)
    return cfg


def train_args(a, cfg):
    import train_B3_MAPPO_v7 as v7
    argv = ['--model', str(a.model), '--port', str(a.port_train),
            '--episodes', str(a.max_additional_episodes),
            '--sim-seconds', str(a.train_sim_seconds),
            '--max-events', str(a.max_events), '--fleetsize', str(a.fleetsize),
            '--max-fleetsize', str(cfg['max_fleetsize']), '--device', str(a.device), '--seed', str(cfg['seed']),
            '--hidden', str(cfg['hidden']), '--graph-layers', str(cfg['graph_layers']),
            '--edge-heads', str(cfg['heads']), '--routing-guide-strength','1.5',
            '--guide-final','1.5', '--guide-decay-episodes','1',
            '--gamma', str(a.gamma if a.gamma is not None else cfg['gamma']),
            '--gae-lambda', str(cfg['gae_lambda']), '--clip', str(cfg['clip']),
            '--lr',str(cfg['lr']), '--epochs',str(cfg['epochs']),
            '--mini-batches',str(cfg['mini_batches']), '--ppo-v7-mode','stable',
            '--v7-value-loss',str(cfg['ppo_v7_options']['value_loss']),
            '--v7-value-clip', str(cfg['ppo_v7_options']['value_clip_epsilon']),
            '--v7-huber-delta',str(cfg['ppo_v7_options']['huber_delta']),
            '--v7-target-kl',str(cfg['ppo_v7_options']['target_kl']),
            '--v7-time-unit-seconds',str(cfg['ppo_v7_options'].get('time_unit_seconds',0)),
            '--parity-samples','64', '--min-candidate-map-rate','0.99']
    return v7, v7.args_parser().parse_args(argv)



def verify_requested_cuda(device_name):
    """Fail before a rollout if PyTorch cannot use the selected NVIDIA CUDA GPU."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            'CUDA is not available in this Python environment. Check NVIDIA GPU, '
            'driver and CUDA-enabled PyTorch first. No CPU fallback; checkpoint untouched.')
    idx = int(device_name.split(':',1)[1])
    if idx >= torch.cuda.device_count():
        raise RuntimeError(f'{device_name} not present; available CUDA GPUs={torch.cuda.device_count()}')
    torch.cuda.set_device(idx)
    print(f'[V8 CUDA] {device_name} | {torch.cuda.get_device_name(idx)}',flush=True)


def verify_model_device(model, selected, phase):
    """Prove Torch model parameters (routing/dispatch actor+critic) use selected device."""
    import torch
    wanted = torch.device(selected)
    configured = torch.device(model.device)
    if configured != wanted:
        raise RuntimeError(f'{phase}: model.device={configured} instead of {wanted}; refusing mislabeled GPU training')
    modules_found = []
    for role_name, role in model.roles.items():
        for comp in ('actor','critic'):
            net = getattr(role,comp,None)
            if not isinstance(net,torch.nn.Module):
                continue
            tensors = list(net.parameters()) + list(net.buffers())
            if not tensors:
                continue
            misplaced=[str(x.device) for x in tensors if x.device != wanted]
            if misplaced:
                raise RuntimeError(f'{phase}: {role_name}.{comp} has {len(misplaced)} tensors on {misplaced[0]}, wanted {wanted}')
            modules_found.append(f'{role_name}.{comp}')
        for key in ('actor_optimizer','critic_optimizer'):
            opt = getattr(role,key,None)
            if opt is None:
                continue
            for values in opt.state.values():
                for state_name, tensor in values.items():
                    # Adam 0-dim step counter is permitted on CPU by torch optimizer.
                    if (isinstance(tensor,torch.Tensor) and tensor.ndim > 0
                            and tensor.is_floating_point() and tensor.device != wanted):
                        raise RuntimeError(f'{phase}: optimizer {role_name}.{key}.{state_name} is on {tensor.device}, wanted {wanted}')
    if not modules_found:
        raise RuntimeError(f'{phase}: did not find Torch actor/critic modules')
    print(f'[V8 DEVICE CHECK] {phase}: {wanted}; {", ".join(modules_found)}',flush=True)

def save_model_atomic(model, path):
    tmp = Path(str(path)+'.tmp')
    model.save(str(tmp))
    os.replace(tmp, path)


def final_selection(a, out, state, candidates):
    """Evaluate using the unchanged V7 evaluator and known held-out seeds."""
    records = []
    for name, cp in candidates:
        cp = Path(cp)
        if not cp.is_file(): continue
        outfile = out/f'eval_{name}.jsonl'
        if outfile.exists():
            try:
                rows = read_jsonl(outfile)
                if len(rows) != len(a.eval_seeds) or [int(r['seed']) for r in rows] != a.eval_seeds:
                    raise ValueError('Incomplete evaluation; manual inspection required: '+str(outfile))
            except Exception as exc:
                print('[V8] Incomplete evaluation; skipped:',repr(exc), flush=True)
                continue
        else:
            cmd = [sys.executable, '-u', str(ROOT/'scripts/evaluate_B3_MAPPO_v7.py'),
                   '--checkpoint',str(cp),'--seeds',*(str(s) for s in a.eval_seeds),
                   '--sim-seconds',str(a.eval_sim_seconds),'--port',str(a.port_eval),
                   '--output',str(outfile),'--guide-strength','1.5', '--device', str(a.device)]
            print('[V8 EVAL]',name, ' '.join(cmd),flush=True)
            ret = subprocess.run(cmd, cwd=ROOT, check=False).returncode
            if ret != 0:
                print('[V8] Evaluation failed for',name,'exit=',ret,'; checkpoint retained',flush=True)
                continue
            rows = read_jsonl(outfile)
        if len(rows) != len(a.eval_seeds):
            print('[V8] Partial evaluation, skipping',name,flush=True); continue
        summary = average_report(rows)
        records.append(dict(candidate=name, checkpoint=str(cp.resolve()), **summary))
        print('[V8 EVAL]',name,summary,flush=True)
    atomic_json(out/'eval_rankings.json', records)
    if records:
        # Predeclared operational score: jobs - soft penalties for wrong-target/block/livelock.
        winner = max(records,key=lambda r:r['score'])
        dst = out/'best_mappo.pt'
        tmp = out/'best_mappo.pt.tmp'
        shutil.copy2(winner['checkpoint'],tmp)
        os.replace(tmp,dst)
        atomic_json(out/'best_info.json', winner)
        print('[V8] BEST:',winner['candidate'],'score',winner['score'],'checkpoint=',dst,flush=True)
    else:
        print('[V8] No fully validated evaluator result. Use latest or original V7; no best claimed.',flush=True)


def main(argv=None):
    a = options(argv)
    cfg = load_source(a)
    out = a.output.resolve()
    if a.dry_run:
        print('[V8 DRY RUN] Starting from',a.from_checkpoint.resolve())
        print('[V8 DRY RUN] Graph',cfg['graph_nodes'],cfg['graph_edges'],
              'Routing=Edge-GAT, Dispatch=B2 MLP, guide=1.5')
        print('[V8 DRY RUN] Budget',a.hours,'h. Sim seconds',a.train_sim_seconds,
              'Evaluation heldout',a.eval_seeds,'Output',out)
        print('[V8 DRY RUN] Transfer AGVs=',a.transfer_source_fleetsize,'->',a.fleetsize,'max=',cfg['max_fleetsize'], 'event cap=',a.max_events)
        print('[V8 DRY RUN] Initial episode:',a.source_last_episode if a.source_last_episode is not None else cfg['episodes'])
        print('[V8 DRY RUN] Requested device:',a.device,'CPU threads:',a.cpu_threads)
        if a.device.startswith('cuda:'):
            verify_requested_cuda(a.device)
        return
    if out.exists() and any(out.iterdir()) and not a.resume:
        raise FileExistsError('Output is nonempty. Use a new folder or --resume: '+str(out))
    # Limit Python compute threads. Does not change AnyLogic/Java CPU usage.
    os.environ['OMP_NUM_THREADS'] = str(a.cpu_threads)
    os.environ['MKL_NUM_THREADS'] = str(a.cpu_threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(a.cpu_threads)
    import numpy as np
    import torch
    torch.set_num_threads(a.cpu_threads)
    if a.device.startswith('cuda:'):
        verify_requested_cuda(a.device)
    print(f'[V8 DEVICE] requested={a.device}, torch_CPU_threads={torch.get_num_threads()}',flush=True)
    from thesis.mappo.mappo_stability_v7 import V7Options, enable_v7, audit_rollout_roles
    from thesis.mappo.gnn_audit_uoe_v2 import RoutingGraphAudit, verify_ppo_rollout_parity
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.routing_guidance_v121 import GuidedEventCollector
    v7, va = train_args(a,cfg)
    out.mkdir(parents=True,exist_ok=True)
    meta_file = out/'overnight_config.json'
    state_file = out/'state.json'
    initial_episode = int(a.source_last_episode if a.source_last_episode is not None else cfg['episodes'])
    expected = dict(version=VERSION,source_checkpoint=str(a.from_checkpoint.resolve()),
                    source_config=str(a.from_config.resolve()), model_file=str(a.model.resolve()),
                    seed=int(cfg['seed']), first_completed_episode=initial_episode,
                    train_sim_seconds=a.train_sim_seconds, max_events=a.max_events, graph_fingerprint=cfg['graph_fingerprint'],
                    gamma=va.gamma, guide_strength=1.5, fleetsize=a.fleetsize,
                    transfer_source_fleetsize=a.transfer_source_fleetsize,
                    max_fleetsize=int(cfg['max_fleetsize']), eval_seeds=a.eval_seeds,
                    eval_sim_seconds=a.eval_sim_seconds)
    if a.resume:
        if not meta_file.exists() or not state_file.exists():
            raise FileNotFoundError('--resume requires overnight_config.json + state.json')
        previous = json.loads(meta_file.read_text(encoding='utf-8'))
        if previous != expected:
            raise ValueError('Resume configuration changed: refusing to mix training runs')
        state = json.loads(state_file.read_text(encoding='utf-8'))
        check_resume_state(state,state['checkpoint'])
        repair_metrics_on_resume(out/'metrics.jsonl',state['last_episode'])
    else:
        atomic_json(meta_file,expected)
        state = dict(last_episode=initial_episode,checkpoint=str(a.from_checkpoint.resolve()),
                     snapshots=[], run_seconds=0.0)
        atomic_json(state_file,state)
        # Checkpoint format unchanged: V7 evaluator/desktop can restore V8.
        eval_config = dict(cfg)
        eval_config['episodes'] = int(initial_episode+a.max_additional_episodes+1)
        eval_config['model_file'] = str(a.model.resolve())
        eval_config['sim_seconds'] = a.train_sim_seconds
        eval_config['gamma'] = va.gamma
        eval_config['max_events'] = a.max_events
        eval_config['fleetsize'] = a.fleetsize
        eval_config['max_fleetsize'] = int(cfg['max_fleetsize'])
        eval_config['transfer_from_fleetsize'] = a.transfer_source_fleetsize
        eval_config['note'] = 'V8 fine-tuning: train/eval fleet size must equal this fleetsize' 
        (out/'run_config.json').write_text(json.dumps(eval_config,indent=2),encoding='utf-8')
    if not (out/'run_config.json').exists():
        raise FileNotFoundError('Missing evaluation run_config.json in V8 output')
    # Stage transfer-source checkpoint NEXT TO effective-fleet config.
    # Evaluator always reads run_config.json adjacent to checkpoint: evaluating original path
    # would accidentally use 8 AGVs, then rank it against 15-AGV candidates.
    staged_baseline = out/'baseline_source_evaluated_same_fleet.pt'
    if not staged_baseline.exists():
        tmp_baseline = out/'baseline_source_evaluated_same_fleet.pt.tmp'
        shutil.copy2(a.from_checkpoint, tmp_baseline)
        os.replace(tmp_baseline, staged_baseline)
    else:
        import hashlib
        def digest(path):
            h=hashlib.sha256()
            with Path(path).open('rb') as f:
                for part in iter(lambda: f.read(1024*1024),b''):
                    h.update(part)
            return h.digest()
        if digest(staged_baseline) != digest(a.from_checkpoint):
            raise RuntimeError('Source checkpoint changed; refusing staged baseline mismatch')
    effective_config = json.loads((out/'run_config.json').read_text(encoding='utf-8'))
    if int(effective_config['fleetsize']) != a.fleetsize:
        raise ValueError('Evaluator config fleetsize does not match training fleetsize')
    # Keep original V7 file in original directory. Never overwrite it.
    env = None
    budget_start = time.monotonic()
    budget_seconds = a.hours*3600
    reserve_seconds = 0 if a.skip_final_eval else a.eval_reserve_minutes*60
    try:
        env = v7.v1.baseline.build_env(va)
        model, graph, edge = v7.make_v4_model(env,va)
        verify_model_device(model,a.device,'model creation')
        if graph.fingerprint != cfg['graph_fingerprint']:
            raise RuntimeError('Source graph fingerprint differs from live AnyLogic model')
        opt = cfg['ppo_v7_options']
        enable_v7(model,V7Options(**opt))
        model.load(str(state['checkpoint']))
        verify_model_device(model,a.device,'checkpoint restored')
        print(f'[V8] Fine-tuning: source={a.transfer_source_fleetsize} AGVs -> target={a.fleetsize} AGVs; max={cfg["max_fleetsize"]}', flush=True)
        if abs(model.cfg.gamma-va.gamma)>1e-8:
            raise RuntimeError('Internal gamma mismatch')
        collector = GuidedEventCollector(model,env,stop_mask_mode=cfg['stop_mask_mode'],routing_guide_strength=1.5)
        auditor = RoutingGraphAudit(model.roles['routing'].actor,sample_limit=256)
        print('[V8] Restored',state['checkpoint'],'after episode',state['last_episode'],flush=True)
        last_duration = 0.0
        while state['last_episode'] < initial_episode+a.max_additional_episodes:
            remaining = budget_seconds-(time.monotonic()-budget_start)
            if remaining <= reserve_seconds+max(last_duration,0):
                print('[V8] Wall-clock budget reached between complete episodes.',flush=True)
                break
            ep = int(state['last_episode'])+1
            # Deliberate deterministic per-episode RNG seed. Environment seed follows V7.
            random.seed(int(cfg['seed']) + ep)
            np.random.seed(int(cfg['seed']) + ep)
            torch.manual_seed(int(cfg['seed']) + ep)
            auditor.reset()
            collector.guide.strength = 1.5
            ts = time.monotonic()
            trajectories, meta = collector.episode(seed=int(cfg['seed'])+ep-1,train=True,max_event_steps=a.max_events)
            live = GraphSpec.from_hive(env.agent_hive)
            if live.fingerprint != graph.fingerprint:
                raise RuntimeError('AnyLogic live graph changed during rollout')
            rep = auditor.report()
            if rep['candidate_map_rate'] < .99:
                raise RuntimeError('GNN mapping < 99%. Refuse policy update.')
            parity = verify_ppo_rollout_parity(model,trajectories,max_samples=64)
            role_audit = audit_rollout_roles(model,trajectories,max_per_role=64)
            updates = model.train(trajectories)
            # A/B rolling checkpoints. State written AFTER model & metrics; prior slot survives crash.
            slot = out/f'rolling_slot_{ep%2}.pt'
            save_model_atomic(model,slot)
            metric = dict(version=VERSION,episode=ep,environment_seed=int(cfg['seed'])+ep-1,
                          environment=meta,graph_mapping=rep,ppo_audit=parity,
                          all_roles_audit=role_audit,policy_updates=updates)
            with (out/'metrics.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(metric,default=str)+'\n')
                f.flush()
                os.fsync(f.fileno())
            state['checkpoint'] = str(slot.resolve())
            state['last_episode'] = ep
            state['run_seconds'] = round(state.get('run_seconds',0)+time.monotonic()-ts,2)
            if (ep-initial_episode)%a.snapshot_every == 0:
                snap = out/f'snapshot_ep_{ep:04d}.pt'
                shutil.copy2(slot,str(snap)+'.tmp')
                os.replace(str(snap)+'.tmp',snap)
                state['snapshots'].append(str(snap.resolve()))
            atomic_json(state_file,state)
            stats=meta.get('statistics',{})
            last_duration=time.monotonic()-ts
            print(f'[MAPPO-V8 EP {ep}] jobDone={stats.get("jobDone",0)} wrong={stats.get("wrongTarget",0)} '
                  f'block={stats.get("block",0)} livelock={stats.get("liveLock",0)} '
                  f'reward={meta.get("delivered_reward",0):.1f} '
                  f'KL={updates.get("routing",{}).get("approx_kl",0):.6f} '
                  f'elapsed={(time.monotonic()-budget_start)/60:.1f}min',flush=True)
    except KeyboardInterrupt:
        print('[V8] Interrupted. Last committed episode:',state['last_episode'],flush=True)
        return
    finally:
        if env is not None:
            env.close()
    # Copy latest AFTER environment is closed, so final_eval opens its own AnyLogic.
    latest = out/'latest_mappo.pt'
    shutil.copy2(state['checkpoint'],str(latest)+'.tmp')
    os.replace(str(latest)+'.tmp',latest)
    if a.skip_final_eval:
        print('[V8] Saved latest:',latest,'(evaluation disabled)',flush=True)
        return
    # Compare prior 15-AGV policy with new 20-AGV policies ON THE SAME 20-AGV FLEET.
    candidates=[(f'source_{a.transfer_source_fleetsize}agv_on_{a.fleetsize}agv',staged_baseline)]
    snapshots=state.get('snapshots',[])
    if snapshots:
        mid=Path(snapshots[len(snapshots)//2]); candidates.append((mid.stem,mid))
    candidates.append((f'latest_ep_{state["last_episode"]:04d}',latest))
    try:
        final_selection(a,out,state,candidates)
    except Exception as exc:
        print('[V8] Final evaluation error:',repr(exc),'Checkpoints preserved.',flush=True)
        raise
    print('[V8] Done. Check',out/'best_info.json','and',out/'best_mappo.pt',flush=True)

if __name__ == '__main__':
    main()
