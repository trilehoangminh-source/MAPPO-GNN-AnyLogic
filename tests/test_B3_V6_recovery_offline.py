"""Offline integration test: create isolated fake V6 project and recover 4/5 rows."""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1]
V6_SCRIPTS = (SOURCE / 'scripts') if (SOURCE / 'scripts' / 'run_B3_V6_CD_replication.py').is_file() else (SOURCE.parent / 'B3_V6_CD_Replication_patch' / 'scripts')

def write_jsonl(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in data), encoding='utf-8')

def run_test():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root/'scripts').mkdir()
        for name in ('run_B3_V6_CD_replication.py', 'v6_cd_common.py'):
            shutil.copy2(V6_SCRIPTS/name, root/'scripts'/name)
        shutil.copy2(SOURCE/'scripts'/'recover_B3_V6_partial_eval.py', root/'scripts'/'recover_B3_V6_partial_eval.py')
        model = root/'envs'/'Matrix_critic_fresh.zip'
        model.parent.mkdir()
        model.write_bytes(b'model')
        folder = root/'mappo_runs'/'test'
        folder.mkdir(parents=True)
        manifest = dict(protocol='V6_CD_EdgeGAT_constant_guide_v1',train_seeds=[143],eval_seeds=[2001,2002,2003,2004,2005], train_episodes=30,train_sim_seconds=720,eval_sim_seconds=3600,max_events=20000,hidden=32,graph_layers=2,heads=2,epochs=5,model_file=str(model),conditions=[dict(code='C',encoder='residual_edge_gat',guide=1.5),dict(code='D',encoder='residual_edge_gat',guide=0.5)])
        (folder/'experiment_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False), encoding='utf-8')
        runfolder = folder/'C_edge_gat_guide1p5'/'train_seed143'
        runfolder.mkdir(parents=True)
        cfg = dict(version='B3_EDGE_GAT_MAPPO_V4',routing_encoder='residual_edge_gat',dispatch_backbone='pooled',dispatch_mask_mode='operation_first', seed=143, episodes=30,sim_seconds=720,max_events=20000,hidden=32,graph_layers=2,heads=2,epochs=5,mini_batches=1,stop_mask_mode='forbid_when_move',guide_start=1.5,guide_final=1.5,final_used_guide_strength=1.5,gamma=0.9,gae_lambda=0.95,clip=0.1,lr=0.0001,model_file=str(model))
        (runfolder/'run_config.json').write_text(json.dumps(cfg, ensure_ascii=False), encoding='utf-8')
        ckpt = runfolder/'checkpoint_ep_0030.pt'
        ckpt.write_bytes(b'z'*2048)
        write_jsonl(runfolder/'metrics.jsonl', [dict(episode=i, environment_seed=143+i-1) for i in range(1,31)])
        def row(seed):
            return dict(version='B3_EDGE_GAT_MAPPO_V4',evaluation=True,seed=seed,checkpoint=str(ckpt.resolve()),description='Mô phỏng sản xuất – thử nghiệm Unicode',environment=dict(reached_terminal=True,dijkstra_strength_eval=1.5))
        outfile=runfolder/'evaluation_3600.jsonl'
        write_jsonl(outfile, [row(i) for i in (2001,2002,2003,2004)])
        original=outfile.read_bytes()
        # Fake evaluator writes a single JSON line for requested missing seeds.
        (root/'scripts'/'evaluate_B3_edgeGAT_MAPPO_v4.py').write_text('''import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument("--checkpoint");p.add_argument("--seeds",nargs="+",type=int);p.add_argument("--sim-seconds");p.add_argument("--port");p.add_argument("--output");a=p.parse_args()
for seed in a.seeds:
 with Path(a.output).open("a",encoding="utf-8") as f:f.write(json.dumps(dict(version="B3_EDGE_GAT_MAPPO_V4",evaluation=True,seed=seed,checkpoint=a.checkpoint,environment=dict(reached_terminal=True,dijkstra_strength_eval=1.5)))+"\\n")
''', encoding='utf-8')
        script = root/'scripts'/'recover_B3_V6_partial_eval.py'
        command=[sys.executable,str(script),'--folder',str(folder),'--condition','C','--train-seed','143','--port','61881']
        env=os.environ.copy()
        env['PYTHONIOENCODING']='utf-8'
        dry=subprocess.run(command+['--dry-run'],capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
        assert dry.returncode == 0,dry.stdout+dry.stderr
        assert outfile.read_bytes()==original
        done=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
        assert done.returncode==0,done.stdout+done.stderr
        actual=[json.loads(x) for x in outfile.read_text(encoding='utf-8').splitlines()]
        assert [r['seed'] for r in actual] == manifest['eval_seeds']
        backups=list((folder/'recovery_logs').glob('*partial_backup*'))
        assert len(backups)==1 and backups[0].read_bytes()==original
        idem=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',env=env)
        assert idem.returncode==0 and 'Already complete' in idem.stdout,idem.stdout+idem.stderr
        assert len(list((folder/'recovery_logs').glob('*partial_backup*')))==1
        print('PASS: dry-run retains input; missing seed only; validated merge; backup; idempotence')

if __name__ == '__main__':
    run_test()
