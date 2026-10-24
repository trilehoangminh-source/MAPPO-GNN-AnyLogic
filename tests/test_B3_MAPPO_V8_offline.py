"""Checks V8 safe utilities and dry-run without starting AnyLogic."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from _mappo_v8_utils import average_report,atomic_json,check_resume_state,repair_metrics_on_resume

def run():
    from tempfile import TemporaryDirectory
    with TemporaryDirectory() as temp:
        folder=Path(temp)
        checkpoint=folder/'test.pt'; checkpoint.write_bytes(b'fake')
        assert check_resume_state({'last_episode':31},checkpoint)
        try:
            check_resume_state({'last_episode':30},folder/'missing')
        except FileNotFoundError:
            pass
        else:
            raise AssertionError('Missing checkpoint not rejected')
        state=folder/'state.json'; atomic_json(state,{'checkpoint':str(checkpoint),'last_episode':31})
        assert json.loads(state.read_text(encoding='utf-8'))['last_episode']==31
    with TemporaryDirectory() as temp:
        f=Path(temp)/'metrics.jsonl'
        f.write_text('\n'.join(json.dumps({'episode':i}) for i in (31,32,33))+'\n',encoding='utf-8')
        assert repair_metrics_on_resume(f,32)==2
        assert [x['episode'] for x in json.loads('['+f.read_text(encoding='utf-8').replace('\n',',').rstrip(',')+']')]==[31,32]
    def row(job,wrong,block,live):
        return {'environment':{'statistics':{'jobDone':job,'wrongTarget':wrong,'block':block,'liveLock':live}}}
    good=average_report([row(8,1,40,20),row(10,0,30,15)])
    bad=average_report([row(7,110,30,10),row(8,100,20,10)])
    assert good['score'] > bad['score']
    assert good['jobDone']==9
    print('MAPPO V8 OFFLINE: PASS')

if __name__=='__main__': run()
