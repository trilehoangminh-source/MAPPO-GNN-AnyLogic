"""Offline tests: explicit guided vs raw logits, masks, and read-only wrapper."""
from __future__ import annotations
import sys, tempfile, types
from pathlib import Path
import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'scripts'))
from thesis.mappo.routing_influence_uoe_v3_2 import RoutingInfluenceAudit
from analyze_B3_UOE_v3_2_influence import summarize


class StubActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = nn.Parameter(torch.tensor([0.0, 0.2, 1.0, -0.5, 0.0]))
    def logits(self, obs):
        return self.theta.unsqueeze(0)


class StubModel:
    def __init__(self):
        self.device = torch.device('cpu')
        self.roles = {'routing': types.SimpleNamespace(actor=StubActor())}
        self.roles['routing'].actor.eval()
    def act(self, role, obs, state, greedy=False):
        if role!='routing':
            return 0, -0.5, 0.0
        base = self.roles['routing'].actor.logits(None)
        mask = torch.as_tensor(obs['action_mask'])[None, :] > 0.5
        prior = torch.as_tensor(obs['routing_guide_bias'])[None, :]
        logits = base.masked_fill(~mask, -1e9) + prior
        logp = torch.log_softmax(logits, -1)
        a = int(logits.argmax(-1).item())
        return a, float(logp[0,a].detach().item()), 0.0


def test_influence():
    # raw choice=2, guided choice=1, stop masked. Exactly 2 AGVs nearby.
    model = StubModel()
    o = np.zeros((3,20), dtype=np.float32)
    o[:,0]=1
    o[0,4:6]=(0,0); o[1,4:6]=(0.5,0); o[2,4:6]=(10.,0)
    obs={'agvs':o, 'stat':np.zeros((2,3),np.float32),
         'action_mask':np.array([0,1,1,0,0],np.float32),
         'routing_guide_bias':np.array([0,1.5,0,0,0],np.float32)}
    s={'ego':7}
    expected=model.act('routing',obs,s,greedy=True)
    assert expected[0]==1
    audit=RoutingInfluenceAudit()
    # supply same interface as as_actor() without importing user's full project
    mod=types.ModuleType('thesis.mappo.algorithm')
    mod.as_actor=lambda o, dev: {k:torch.as_tensor(o[k],dtype=torch.float32,device=dev)
          for k in ('agvs','stat','action_mask','routing_guide_bias')}
    mod.batchify=lambda x: {k:v.unsqueeze(0) for k,v in x.items()}
    old=sys.modules.get('thesis.mappo.algorithm')
    import thesis.mappo as package
    old_attr=getattr(package,'algorithm',None)
    sys.modules['thesis.mappo.algorithm']=mod
    package.algorithm=mod
    try:
        with tempfile.TemporaryDirectory() as t:
            f=Path(t)/'influence_seed1001.jsonl'
            with audit.attach(model,f):
                assert model.act('routing',obs,s,greedy=True)==expected
                assert model.act('dispatch_front',obs,s)==(0,-0.5,0.0)
            assert model.act('routing',obs,s,greedy=True)==expected
            result=summarize(f)
            assert result['decisions']==1
            assert result['raw_top1_on_guide']==0
            assert result['guided_top1_on_guide']==1
            assert result['guidance_changes_top1']==1
            assert result['nearby_1m']==1
            assert result['all_seven_nearby_6m']==0
            assert result['logprob_parity_error_max']<1e-4
    finally:
        if old is None:
            sys.modules.pop('thesis.mappo.algorithm',None)
        else:
            sys.modules['thesis.mappo.algorithm']=old
        if old_attr is None:
            delattr(package,'algorithm')
        else:
            package.algorithm=old_attr


if __name__=='__main__':
    test_influence()
    print('PASS V3.2 influence audit: logits reconstruction, greedy decision, masks, non-invasive wrapper, distance proxy')
