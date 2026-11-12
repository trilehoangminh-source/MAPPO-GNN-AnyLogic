"""CPU-only numerical tests for full-rollout vs split microbatch gradients."""
from __future__ import annotations

import copy
import importlib.util
import sys
import types
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Make a minimal stub with the same mathematical contracts as the installed V7,
# allowing this patch to be tested standalone without AnyLogic / PettingZoo.
import math
from collections import defaultdict

def make_gae_records(trajectories, gamma, lam, time_unit_seconds=0.):
    groups = defaultdict(list)
    for path in trajectories:
        gae = 0.
        for step in reversed(path):
            gamma_t=gamma if time_unit_seconds==0 else gamma**(step['elapsed_seconds']/time_unit_seconds)
            done = bool(step['done'])
            nxt=0. if done else float(step['next_value'])
            delta=float(step['reward'])+gamma_t*nxt-float(step['value'])
            gae=delta+gamma_t*lam*(0. if done else gae)
            rec=dict(step,advantage=gae,return_target=gae+float(step['value']))
            groups[step['role']].append(rec)
    return groups

def critic_loss(p,old,y,opts):
    err = torch.nn.functional.smooth_l1_loss if opts.value_loss=='clipped_huber' else None
    if err:
        fn=lambda x:err(x,torch.zeros_like(x),beta=opts.huber_delta,reduction='none')
    else:
        fn=lambda x:x.square()
    original=fn(p-y)
    if opts.value_loss=='mse':return original.mean()
    clipped=old+(p-old).clamp(-opts.value_clip_epsilon,opts.value_clip_epsilon)
    return torch.maximum(original,fn(clipped-y)).mean()

algo=types.ModuleType('thesis.mappo.algorithm')
algo.as_actor=lambda x,dev:{k:torch.as_tensor(v,device=dev,dtype=torch.float32) for k,v in x.items()}
algo.as_global=lambda x,dev:{k:torch.as_tensor(v,device=dev,dtype=torch.float32) for k,v in x.items()}
algo.stack_dict=lambda seq,fn,dev:{k:torch.stack([fn(x,dev)[k] for x in seq]) for k in seq[0]}

thesis=types.ModuleType('thesis');thesis.__path__=[]
mappo=types.ModuleType('thesis.mappo');mappo.__path__=[]
v7=types.ModuleType('thesis.mappo.mappo_stability_v7')
v7.make_gae_records=make_gae_records
v7.critic_loss=critic_loss
mappo.algorithm=algo
mappo.mappo_stability_v7=v7
sys.modules.update({'thesis':thesis,'thesis.mappo':mappo,
 'thesis.mappo.algorithm':algo,'thesis.mappo.mappo_stability_v7':v7})

spec=importlib.util.spec_from_file_location('accum',ROOT/'thesis/mappo/v15_fullrollout_accum.py')
accum=importlib.util.module_from_spec(spec)
spec.loader.exec_module(accum)

class ToyActor(torch.nn.Module):
    uses_action_mask=True
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Linear(3,3)
    def dist(self,obs):
        return torch.distributions.Categorical(logits=self.net(obs['x']))

class ToyCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net=torch.nn.Linear(2,1)
    def forward(self,gs):
        return self.net(gs['x']).squeeze(-1)

class ToyValueNorm:
    def update(self,*a):pass
    def normalize(self,x):return x

class TestFullRollout(unittest.TestCase):
    def make_model(self):
        actor=ToyActor(); critic=ToyCritic()
        role=types.SimpleNamespace(actor=actor,critic=critic,
            actor_optimizer=torch.optim.Adam(actor.parameters(),lr=.003),
            critic_optimizer=torch.optim.Adam(critic.parameters(),lr=.003),
            valuenorm=ToyValueNorm())
        cfg=types.SimpleNamespace(mini_batches=1,gamma=.99,gae_lambda=.95,
            normalize_advantage=True,value_normalization=False,epochs=1,
            clip_epsilon=.2,entropy_coef=.01,value_coef=.5,max_grad_norm=.5)
        opts=types.SimpleNamespace(value_loss='clipped_huber',value_clip_epsilon=.2,
            huber_delta=1.,target_kl=.03,time_unit_seconds=0.)
        return types.SimpleNamespace(roles={'routing':role},cfg=cfg,device='cpu',
            _v7_options=opts,_v7_stability_installed=True)

    def make_data(self, model, n=11):
        rng=np.random.default_rng(1234)
        path=[]
        for i in range(n):
            obs={'x':rng.normal(size=3).astype(np.float32),
                 'action_mask':np.ones(3,dtype=np.float32)}
            st={'x':rng.normal(size=2).astype(np.float32)}
            with torch.no_grad():
                dist=model.roles['routing'].actor.dist({'x':torch.tensor(obs['x'])[None,:]})
                action=int(dist.sample().item())
                logp=float(dist.log_prob(torch.tensor([action])))
                old_v=float(model.roles['routing'].critic({'x':torch.tensor(st['x'])[None,:]}))
            path.append({'role':'routing','observation':obs,'state':st,
                'action':action,'logp':logp,'value':old_v,
                'reward':float(rng.normal()),'done':True,'next_value':0.})
        return [[rec] for rec in path]

    def test_uneven_microbatch_matches_exact_full_rollout(self):
        torch.manual_seed(27)
        model_full=self.make_model()
        data=self.make_data(model_full,11)
        model_split=self.make_model()
        model_split.roles['routing'].actor.load_state_dict(model_full.roles['routing'].actor.state_dict())
        model_split.roles['routing'].critic.load_state_dict(model_full.roles['routing'].critic.state_dict())
        accum.enable_fullrollout(model_full,microbatch_size=11)
        accum.enable_fullrollout(model_split,microbatch_size=3)
        torch.manual_seed(19)
        whole=model_full.train(data)['routing']
        torch.manual_seed(19)
        split=model_split.train(data)['routing']
        for key in ('actor_loss','value_loss','approx_kl','entropy','actor_grad_norm','critic_grad_norm'):
            self.assertAlmostEqual(whole[key],split[key],places=5,msg=key)
        self.assertEqual(whole['optimizer_steps'],1)
        self.assertEqual(split['optimizer_steps'],1)
        self.assertEqual(split['microbatches_processed'],4)
        for group in ['actor','critic']:
            for pa,pb in zip(getattr(model_full.roles['routing'],group).parameters(),
                             getattr(model_split.roles['routing'],group).parameters()):
                self.assertTrue(torch.allclose(pa,pb,atol=1e-6,rtol=1e-6),group)

    def test_one_step_per_epoch(self):
        torch.manual_seed(36)
        model=self.make_model(); data=self.make_data(model,17)
        model.cfg.epochs=5
        accum.enable_fullrollout(model, microbatch_size=4)
        report=model.train(data)['routing']
        self.assertEqual(report['effective_minibatches_per_epoch'],1)
        self.assertEqual(report['optimizer_steps'],5)
        self.assertEqual(report['microbatches_processed'],25)

    def test_kl_aborts_entire_optimizer_step(self):
        torch.manual_seed(37)
        model=self.make_model(); data=self.make_data(model,11)
        for path in data:path[0]['logp']-=2.
        weights=[p.detach().clone() for p in model.roles['routing'].actor.parameters()]
        accum.enable_fullrollout(model,microbatch_size=3)
        result=model.train(data)['routing']
        self.assertEqual(result['optimizer_steps'],0)
        self.assertTrue(result['early_stopped'])
        for old,p in zip(weights,model.roles['routing'].actor.parameters()):
            self.assertTrue(torch.equal(old,p))

    def test_requires_v7_and_one_blueprint_minibatch(self):
        m=self.make_model();m.cfg.mini_batches=2
        with self.assertRaisesRegex(ValueError,'mini_batches=1'):
            accum.enable_fullrollout(m)
        m.cfg.mini_batches=1;m._v7_stability_installed=False
        with self.assertRaisesRegex(RuntimeError,'V7 stability'):
            accum.enable_fullrollout(m)

if __name__=='__main__':unittest.main()
