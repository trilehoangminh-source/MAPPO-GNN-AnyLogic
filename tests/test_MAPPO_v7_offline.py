"""Offline validation: PPO, clipped critic, KL stop, GAE, role isolation.

Run: python tests/test_MAPPO_v7_offline.py
No AnyLogic, Alpyne or Java required.
"""
from __future__ import annotations
import importlib.util
import math
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
modfile = ROOT / 'thesis' / 'mappo' / 'mappo_stability_v7.py'
spec = importlib.util.spec_from_file_location('v7_stability_unit', modfile)
v7 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v7
spec.loader.exec_module(v7)


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_gae_and_value_clip():
    traj = [[dict(role='routing', reward=1., value=2., next_value=3., done=False,
                  elapsed_seconds=10.),
             dict(role='routing', reward=4., value=3., next_value=100., done=True,
                  elapsed_seconds=10.)]]
    g = v7.make_gae_records(traj, gamma=.9, lam=.95)
    a0 = g['routing'][0]['advantage']
    a1 = g['routing'][1]['advantage']
    check(abs(a0 - (4. - 3.)) < 1e-7, 'terminal must not bootstrap')
    check(abs(a1 - (1 + .9*3 - 2 + .9*.95*1)) < 1e-7, 'GAE recursion wrong')
    timed = v7.make_gae_records(traj, gamma=.9, lam=.95, time_unit_seconds=20.)
    expected = 1 + (.9 ** .5)*3 - 2 + (.9 ** .5)*.95*1
    check(abs(timed['routing'][1]['advantage']-expected) < 1e-7, 'SMDP GAE wrong')
    # A separate alias lifetime (trajectory) MUST NOT inherit later GAE.
    split = v7.make_gae_records([traj[0][:1], traj[0][1:]], gamma=.9, lam=.95)
    check(abs(split['routing'][0]['advantage'] - (1 + .9*3 - 2)) < 1e-7,
          'alias boundary incorrectly spliced')

    o = v7.V7Options(value_loss='clipped_mse', value_clip_epsilon=.2)
    prediction = torch.tensor([.5], requires_grad=True)
    loss = v7.critic_loss(prediction, torch.tensor([0.]), torch.tensor([2.]), o)
    check(abs(loss.item() - 3.24) < 1e-5, 'Clipped loss must take max(unclipped,clipped)')
    loss.backward()
    check(abs(prediction.grad.item()) < 1e-7, 'Value-clipped branch should be constant outside clip')
    huber = v7.critic_loss(torch.tensor([1.]), torch.tensor([.8]), torch.tensor([0.]),
                           v7.V7Options(value_loss='clipped_huber', value_clip_epsilon=.2))
    check(math.isfinite(huber.item()), 'Huber loss not finite')
    for kwargs in (dict(target_kl=-1),dict(huber_delta=0),dict(value_clip_epsilon=0)):
        try:
            v7.V7Options(**kwargs)
            raise AssertionError('Invalid options allowed')
        except ValueError:
            pass


# The original algorithm.py is NOT bundled into this patch. For the standalone
# test only, supply a compatibility shim for its documented converter interface;
# the real project uses thesis.mappo.algorithm unchanged.
class TinyValueNorm:
    def __init__(self):
        self.mean = 0.
        self.scale = 1.
        self.updates = 0
    def update(self, arr):
        self.mean = float(np.mean(arr)); self.scale = max(float(np.std(arr)),1.)
        self.updates += 1
    def normalize(self, arr):
        return (arr-self.mean)/self.scale


class TinyActor(nn.Module):
    def __init__(self, num_actions=5):
        super().__init__()
        self.action_dim = num_actions
        self.layer = nn.Linear(2, num_actions)
        self.bn = nn.BatchNorm1d(2)
        with torch.no_grad():
            self.layer.weight.zero_()
            self.layer.bias.zero_()
    def dist(self, ob):
        x = self.bn(ob['agvs'])
        logits = self.layer(x)
        if 'routing_guide_bias' in ob:
            logits = logits + ob['routing_guide_bias']
        logits = logits.masked_fill(ob['action_mask'] <= 0.5, -1e8)
        return Categorical(logits=logits)


class TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3,1)
    def forward(self, state):
        x = torch.cat((state['agvs'],state['ego'].unsqueeze(-1).float()),dim=-1)
        return self.linear(x).squeeze(-1)


def tiny_model(cfg):
    roles = {}
    for role in ('routing','dispatch_front','dispatch_rear'):
        actor = TinyActor(5 if role=='routing' else 3)
        critic = TinyCritic()
        roles[role] = SimpleNamespace(
            actor=actor, critic=critic,
            actor_optimizer=torch.optim.Adam(actor.parameters(),lr=1e-3),
            critic_optimizer=torch.optim.Adam(critic.parameters(),lr=1e-3),
            valuenorm=TinyValueNorm(),
        )
    model=SimpleNamespace(roles=roles,cfg=cfg,device=torch.device('cpu'))
    return model


def make_transitions(model, n=6, seed=1):
    torch.manual_seed(seed)
    routes=[]
    for role in model.roles:
        steps=[]
        actor=model.roles[role].actor
        actor.eval()
        for i in range(n):
            d=5 if role=='routing' else 3
            ob={'agvs':np.array([float(i%3),float((i+2)%4)],dtype=np.float32),
                'stat':np.zeros(2,dtype=np.float32),
                'action_mask':np.ones(d,dtype=np.float32)}
            if role=='routing':
                ob['routing_guide_bias']=np.array([0., 0.5,0.,0.,0.],dtype=np.float32)
            act=1 if d==5 else 2
            with torch.no_grad():
                t_ob={k:torch.tensor(v[None,:]) for k,v in ob.items()}
                lp=actor.dist(t_ob).log_prob(torch.tensor([act])).item()
            steps.append(dict(role=role,observation=ob,
                              state={'agvs':np.array([.2,.4],dtype=np.float32),
                                     'stat':np.array([0.,0.],dtype=np.float32),'ego':1},
                              action=act,logp=float(lp),value=0.,
                              reward=(i%4)*.2,next_value=0.,done=i==n-1,
                              elapsed_seconds=10.))
        routes.append(steps)
    return routes


def test_optimizer():
    # Save any already imported real algorithm module; restore after test.
    pkg = sys.modules.get('thesis.mappo')
    actual = sys.modules.get('thesis.mappo.algorithm')
    if 'thesis' not in sys.modules:
        sys.modules['thesis']=ModuleType('thesis');sys.modules['thesis'].__path__=[]
    if pkg is None:
        sys.modules['thesis.mappo']=ModuleType('thesis.mappo')
        sys.modules['thesis.mappo'].__path__=[]
    shim = ModuleType('thesis.mappo.algorithm')
    def as_actor(ob, device):
        return {k: torch.tensor(v,device=device,dtype=torch.float32) for k,v in ob.items()}
    def as_global(s,device):
        return dict(agvs=torch.tensor(s['agvs'],device=device,dtype=torch.float32),
                    stat=torch.tensor(s['stat'],device=device,dtype=torch.float32),
                    ego=torch.tensor(s['ego'],device=device,dtype=torch.long))
    def stack_dict(items,fn,device):
        converted=[fn(s,device) for s in items]
        return {k:torch.stack([a[k] for a in converted]) for k in converted[0]}
    def batchify(d):
        return {k:x.unsqueeze(0) for k,x in d.items()}
    shim.as_actor=as_actor; shim.as_global=as_global
    shim.stack_dict=stack_dict;shim.batchify=batchify
    sys.modules['thesis.mappo.algorithm']=shim
    setattr(sys.modules['thesis.mappo'], 'algorithm', shim)
    try:
        cfg=SimpleNamespace(gamma=.9,gae_lambda=.95, normalize_advantage=True,
                            value_normalization=True, mini_batches=2,epochs=3,
                            clip_epsilon=.1,entropy_coef=.01,value_coef=.5,max_grad_norm=.5)
        m=tiny_model(cfg)
        rollout=make_transitions(m)
        parity=v7.audit_rollout_roles(m,rollout,max_per_role=5)
        check(set(parity)==set(m.roles), 'Audit did not include Dispatching roles')
        check(all(x['passed'] for x in parity.values()), 'Parity audit failed')
        before={k: {n:v.detach().clone() for n,v in role.actor.state_dict().items()}
                for k,role in m.roles.items()}
        v7.enable_v7(m,v7.V7Options(target_kl=.03))
        result=m.train(rollout)
        check(len(result)==3, 'All 3 roles must be trained')
        for key,role in m.roles.items():
            r=result[key]
            check(r['transitions']==6 and r['optimizer_steps']>0, 'No role update')
            check(role.valuenorm.updates==1, 'ValueNorm should update once/role/rollout')
            check('clip_fraction' in r and r['approx_kl']>=0, 'Missing MAPPO diagnostics')
            change=sum((v.detach()-before[key][n]).abs().sum().item()
                       for n,v in role.actor.state_dict().items() if v.is_floating_point())
            check(change>0, 'Actor did not update: '+key)
        # Ensure KL early-stop prevents an off-policy optimizer step.
        m2=tiny_model(cfg)
        old=make_transitions(m2)
        for segment in old:
            for s in segment:
                s['logp'] = float(s['logp']) - 8.0
        v7.enable_v7(m2,v7.V7Options(target_kl=.001))
        rep=m2.train(old)
        check(all(r['early_stopped'] and r['optimizer_steps']==0 for r in rep.values()),
              'KL should guard all optimizer steps')
        # Check that basic checkpoint structure remains serializable.
        import io
        buff=io.BytesIO()
        torch.save({'roles': {role:{'actor':p.actor.state_dict(),
                         'critic':p.critic.state_dict(),
                         'valuenorm':{'mean':p.valuenorm.mean}}
                         for role,p in m.roles.items()}}, buff)
        check(buff.getbuffer().nbytes>100, 'Checkpoint incompatible')
    finally:
        if actual is not None:
            sys.modules['thesis.mappo.algorithm']=actual
            setattr(sys.modules['thesis.mappo'],'algorithm',actual)
        else:
            sys.modules.pop('thesis.mappo.algorithm',None)
        if pkg is None:
            sys.modules.pop('thesis.mappo',None)

if __name__=='__main__':
    test_gae_and_value_clip()
    test_optimizer()
    print('[MAPPO V7 OFFLINE] PASS: GAE/aliases, time discount, clipped critic, '
          'all-role parity, actor updates, value norm, KL guard and checkpoint schema')
