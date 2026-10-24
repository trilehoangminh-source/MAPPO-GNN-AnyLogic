"""Offline checks: previous 15 AGV transfer and honest cumulative training steps."""
from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import json
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'scripts'

def make_record(ep, steps, reward):
    return {'episode':ep,'environment':{'closed_transitions':steps,'delivered_reward':reward,
            'statistics':{'jobDone':1,'block':2,'targetReached':3,'wrongTarget':4,'liveLock':5}}}

def main():
    sys.path.insert(0,str(SCRIPTS))
    spec=importlib.util.spec_from_file_location('offline_runner_v8_20', SCRIPTS/'train_B3_MAPPO_v8_overnight.py')
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import plot_mappo_training_timesteps as plotmod
    with TemporaryDirectory() as t:
        root=Path(t)
        v7=root/'v7'; v7.mkdir()
        cfg={'version':'B3_MAPPO_STABILITY_V7','routing_encoder':'residual_edge_gat',
             'ppo_v7_mode':'stable','guide_start':1.5,'guide_final':1.5,'graph_nodes':105,
             'graph_edges':328,'graph_fingerprint':'graph-fingerprint-123',
             'model_file':str(root/'same_topology.zip'),'fleetsize':8,'max_fleetsize':20,
             'gamma':.9,'max_events':9000}
        (root/'same_topology.zip').write_bytes(b'placeholder model for dryrun')
        config_path=v7/'run_config.json'; config_path.write_text(json.dumps(cfg), encoding='utf-8')
        (v7/'checkpoint_ep_0030.pt').write_bytes(b'placeholder checkpoint')
        prior=root/'V8_15'; prior.mkdir()
        (prior/'rolling_slot_0.pt').write_bytes(b'committed checkpoint from ep144')
        state={'checkpoint':str((prior/'rolling_slot_0.pt').resolve()),'last_episode':144}
        (prior/'state.json').write_text(json.dumps(state), encoding='utf-8')
        prev_meta={'version':mod.VERSION,'fleetsize':15,'max_fleetsize':20,
                   'graph_fingerprint':cfg['graph_fingerprint'],'guide_strength':1.5,'gamma':.9}
        (prior/'overnight_config.json').write_text(json.dumps(prev_meta), encoding='utf-8')
        (prior/'run_config.json').write_text(json.dumps({**cfg,'fleetsize':15}), encoding='utf-8')
        out=root/'V8_20'
        args=mod.options(['--from-checkpoint',str(v7/'checkpoint_ep_0030.pt'),
                          '--from-config',str(config_path), '--from-previous-run',str(prior),
                          '--fleetsize','20','--output',str(out), '--hours','5', '--dry-run'])
        c=mod.load_source(args)
        assert args.source_last_episode==144
        assert args.transfer_source_fleetsize==15
        assert args.fleetsize==20 and args.max_events==9000
        assert args.from_checkpoint.name=='rolling_slot_0.pt'
        bad=mod.options(['--from-checkpoint',str(v7/'checkpoint_ep_0030.pt'),
                          '--from-config',str(config_path), '--from-previous-run',str(prior),
                          '--fleetsize','15','--output',str(out),'--dry-run'])
        try: mod.load_source(bad)
        except ValueError as exc: assert 'higher' in str(exc)
        else: raise AssertionError('Same fleet must be refused for transfer')
        # No RL imports / Java launched on dry run
        p=subprocess.run([sys.executable,str(SCRIPTS/'train_B3_MAPPO_v8_overnight.py'),
            '--from-checkpoint',str(v7/'checkpoint_ep_0030.pt'),
            '--from-config',str(config_path),'--from-previous-run',str(prior),
            '--fleetsize','20','--output',str(out),'--dry-run'],
            capture_output=True,text=True)
        assert p.returncode==0, p.stderr
        assert 'Transfer AGVs= 15 -> 20' in p.stdout, p.stdout
        folder15=root/'fleet15'; folder15.mkdir()
        folder20=root/'fleet20'; folder20.mkdir()
        (folder15/'run_config.json').write_text('{"fleetsize":15}',encoding='utf-8')
        (folder20/'run_config.json').write_text('{"fleetsize":20}',encoding='utf-8')
        (folder15/'metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in (
            make_record(143,123,-50),make_record(144,456,20)))+'\n',encoding='utf-8')
        (folder20/'metrics.jsonl').write_text(json.dumps(make_record(145,789,-100))+'\n',encoding='utf-8')
        phases=plotmod.load_training_phases([folder15,folder20])
        assert phases[0][2][-1]['cumulative_transitions']==579
        assert phases[1][2][-1]['cumulative_transitions']==1368
        assert phases[1][2][0]['episode']==145
        plotmod.plot(phases,root/'figures'/'Figure_TrainingTimesteps_6panels',7)
        assert (root/'figures/Figure_TrainingTimesteps_6panels.png').is_file()
        assert (root/'figures/training_timesteps_data.csv').is_file()
    print('V8 20 AGV + TRAINING TIMESTEPS OFFLINE: PASS')

if __name__=='__main__':
    main()
