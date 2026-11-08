"""Shared V8-compatible builder for V11. Never writes into source V8 run."""
from __future__ import annotations
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def atomic_json(path,data):
    path=Path(path); temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    os.replace(temp,path)


def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def locate_source_episode(folder, episode, manual=None):
    """Accept only an immutable EP-N snapshot or state-confirmed rolling checkpoint.

    An episode filename is NOT independently sufficient proof of model contents;
    strict V8 model.load and model/graph checks still occur before training.
    """
    folder = Path(folder).resolve()
    episode = int(episode)
    if episode < 1:
        raise ValueError('source episode must be a positive integer')
    suffix = f'{episode:04d}'
    allowed = (f'snapshot_ep_{suffix}.pt', f'checkpoint_ep_{suffix}.pt',
               f'checkpoint_ep_{suffix}.pth', f'checkpoint_ep{suffix}.pt')

    def state_matches(file):
        state_file = folder / 'state.json'
        if file.parent != folder or not state_file.is_file():
            return False
        s = read_json(state_file)
        return (int(s['last_episode']) == episode and
                file == Path(s['checkpoint']).resolve())

    if manual is not None:
        file = Path(manual).resolve()
        if not file.is_file():
            raise FileNotFoundError(file)
        if file.name in allowed or (file.name.startswith(('rolling_slot_', 'latest_mappo'))
                                    and state_matches(file)):
            return file
        raise RuntimeError(
            f'Cannot verify manually selected checkpoint for EP{episode}: {file}. '
            f'Use an immutable snapshot named {allowed[0]} or a rolling checkpoint '
            'whose source state.json points to it at the requested episode.')
    for filename in allowed:
        file = folder / filename
        if file.is_file():
            return file.resolve()
    state_file = folder / 'state.json'
    if state_file.is_file():
        s = read_json(state_file)
        if int(s['last_episode']) == episode:
            file = Path(s['checkpoint']).resolve()
            if file.is_file() and file.parent == folder:
                return file
    raise FileNotFoundError(
        f'No verified V8 EP{episode} checkpoint found under {folder}. '
        f'Search for {allowed[0]} or pass --source-checkpoint with its full path. '
        'Do NOT rename a checkpoint from a different episode.')


def validate_v8_cfg(cfg):
    if (int(cfg['fleetsize']),int(cfg['max_fleetsize']))!=(20,20):
        raise ValueError('Expected V8 20-AGV source config')
    if cfg.get('routing_encoder')!='residual_edge_gat' or cfg.get('dispatch_backbone')!='pooled':
        raise ValueError('Expected V8 Residual Edge-GAT routing + MLP Dispatch')
    if cfg.get('dispatch_mask_mode')!='operation_first':
        raise ValueError('V8 operation_first dispatch mask required')
    if float(cfg['final_used_guide_strength'])!=1.5:
        raise ValueError('Dijkstra guide must remain 1.5')
    if int(cfg['graph_nodes'])!=105 or int(cfg['graph_edges'])!=328:
        raise ValueError('Different graph geometry from V8 105 nodes / 328 edges')


def build_v11(cfg,source_checkpoint,device,port,sim_seconds,max_events,horizon,decay,
              v11_checkpoint=None, verify_parity=False):
    import torch
    import train_B3_MAPPO_v8_overnight as v8
    import train_mappo_long_B3_v121 as baseline
    from thesis.mappo.mappo_stability_v7 import V7Options,enable_v7
    from thesis.mappo.gnn_routing_B3_v121 import GraphSpec
    from thesis.mappo.congestion_edgegat_v11 import install_v11,synthetic_parity_probe
    validate_v8_cfg(cfg)
    # Must happen before AnyLogic environment creation.
    baseline.SIM_CONFIG['reward_part_completed']=2.0
    os.environ['MAPPO_DISPATCH_MASK_MODE']='operation_first'
    a=SimpleNamespace(model=Path(cfg['model_file']).resolve(),port_train=port,
        max_additional_episodes=10000,train_sim_seconds=sim_seconds,max_events=max_events,
        fleetsize=20,device=device,gamma=float(cfg['gamma']))
    if not a.model.is_file():
        raise FileNotFoundError('AnyLogic model ZIP missing: '+str(a.model))
    v7,args=v8.train_args(a,cfg)
    env=v7.v1.baseline.build_env(args)
    try:
        model,spec,_=v7.make_v4_model(env,args)
        if GraphSpec.from_hive(env.agent_hive).fingerprint != cfg['graph_fingerprint']:
            raise RuntimeError('AnyLogic graph mismatch source V8')
        enable_v7(model,V7Options(**cfg['ppo_v7_options']))
        model.load(str(source_checkpoint)) # strict 100% V8 before adding params
        if verify_parity:
            obs,old_logits=synthetic_parity_probe(model)
        installation=install_v11(model,horizon,decay)
        if verify_parity:
            with torch.no_grad():
                model.roles['routing'].actor.eval()
                newer=model.roles['routing'].actor.logits(obs)
                delta=(newer-old_logits).abs().max().item()
                if delta > 2e-5:
                    raise RuntimeError('V11 zero-initialization parity FAILED: '+str(delta))
                print('[V11 PARITY] max |logits V11-V8| =',delta,flush=True)
        if v11_checkpoint is not None:
            model.load(str(v11_checkpoint))
        return env,model,spec,installation
    except Exception:
        env.close()
        raise
