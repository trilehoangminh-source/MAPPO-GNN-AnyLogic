"""Standalone CPU tests; no AnyLogic or Java engine required.

Run: python -m unittest discover -s tests -p 'test_v14_deep_mlp_offline.py' -v
"""
from __future__ import annotations
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from thesis.mappo.v14_deep_candidate_mlp import (
    CandidateAwareDispatchActor, DeepCentralizedCritic, install_v14)
import train_v14_deep_mlp_scratch as runner


class OfflineV14Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(23)
        torch.set_num_threads(2)

    def data(self,b=3,a=20,n=27,f=20):
        agvs=torch.zeros(b,a,f)
        stations=torch.zeros(b,n,f)
        agvs[:,:min(a,7),:]=torch.randn(b,min(a,7),f)
        stations[:,:12,:]=torch.randn(b,12,f)
        mask=torch.zeros(b,n)
        mask[:,[1,7,15,26]]=1
        return ({'agvs':agvs,'stat':stations,'action_mask':mask},
                {'agvs':agvs,'stat':stations,'ego':torch.tensor([0,1,2],dtype=torch.long)[:b]})

    def test_dispatch_forward_gradient_and_exact_dims(self):
        obs,_=self.data()
        actor=CandidateAwareDispatchActor(20,27)
        out=actor.logits(obs)
        self.assertEqual(tuple(out.shape),(3,27))
        self.assertTrue(torch.isfinite(out).all())
        out.sum().backward()
        for name in ('ego_encoder.0.weight','other_encoder.0.weight',
                     'station_encoder.0.weight','station_identity.weight','scorer.0.weight'):
            g=dict(actor.named_parameters())[name].grad
            self.assertIsNotNone(g,name)
            self.assertGreater(float(g.norm()),0,name)
        self.assertEqual(len([m for m in actor.scorer if isinstance(m,torch.nn.Linear)]),4)

    def test_invalid_actions_remain_forbidden(self):
        obs,_=self.data()
        actor=CandidateAwareDispatchActor(20,27)
        dummy=types.ModuleType('thesis.mappo.networks')
        def masked_logits(logits,mask):
            if logits.shape!=mask.shape or not (mask>.5).any(dim=1).all():
                raise ValueError('Bad action mask')
            return logits.masked_fill(mask<=.5,torch.finfo(logits.dtype).min/2)
        dummy.masked_logits=masked_logits
        with patch.dict(sys.modules,{'thesis.mappo.networks':dummy}):
            d=actor.dist(obs)
        chosen=d.probs.argmax(dim=-1)
        self.assertTrue(bool((obs['action_mask'].gather(1,chosen[:,None])>.5).all()))
        self.assertTrue(bool((d.probs[obs['action_mask']<=.5]<1e-7).all()))
        with self.assertRaises(ValueError):
            actor.logits({**obs,'action_mask':torch.zeros_like(obs['action_mask'])})

    def test_dispatch_can_detect_individual_station_difference(self):
        obs,_=self.data(b=1)
        actor=CandidateAwareDispatchActor(20,27).eval()
        with torch.no_grad():
            original=actor.logits(obs).clone()
            changed=obs['stat'].clone()
            changed[:,7,:] += 9.0
            new=actor.logits({**obs,'stat':changed})
        self.assertGreater(float((new-original).abs().max()),1e-7)

    def test_dispatch_is_invariant_to_order_of_other_agvs(self):
        obs,_=self.data(b=2)
        actor=CandidateAwareDispatchActor(20,27).eval()
        shuffled=obs['agvs'].clone()
        shuffled[:,1:]=torch.flip(shuffled[:,1:],dims=(1,))
        with torch.no_grad():
            old=actor.logits(obs)
            new=actor.logits({**obs,'agvs':shuffled})
        self.assertTrue(torch.allclose(old,new,atol=1e-5,rtol=0))

    def test_dispatch_handles_only_one_agv_and_legal_zero_station_row(self):
        obs,_=self.data(b=1,a=1)
        actor=CandidateAwareDispatchActor(20,27)
        with torch.no_grad():
            output=actor.logits(obs)
        self.assertEqual(tuple(output.shape),(1,27))
        self.assertTrue(torch.isfinite(output).all())

    def test_critic_forward_backprop_ego_identity(self):
        _,gs=self.data()
        critic=DeepCentralizedCritic(20)
        y=critic(gs)
        self.assertEqual(tuple(y.shape),(3,))
        y.sum().backward()
        for name in ('agv_encoder.0.weight','station_encoder.0.weight','value_head.0.weight'):
            grad=dict(critic.named_parameters())[name].grad
            self.assertIsNotNone(grad)
            self.assertGreater(float(grad.norm()),0)
        self.assertEqual([int(m.out_features) for m in critic.value_head
                          if isinstance(m,torch.nn.Linear)], [128,128,64,1])

    def test_critic_uses_physical_ego_index(self):
        _,gs=self.data()
        critic=DeepCentralizedCritic(20).eval()
        with torch.no_grad():
            old=critic(gs)
            gs['ego']=torch.tensor([2,3,4],dtype=torch.long)
            new=critic(gs)
        self.assertGreater(float((new-old).abs().max()),1e-6)
        gs['ego']=torch.tensor([200,1,2],dtype=torch.long)
        with self.assertRaises(ValueError):critic(gs)

    def test_installer_replaces_2_dispatch_actors_and_3_critics(self):
        alg=types.ModuleType('thesis.mappo.algorithm')
        class RunningValueNorm:
            def __init__(self):self.count=1
        alg.RunningValueNorm=RunningValueNorm
        class Role:
            def __init__(self,name):
                self.actor=torch.nn.Linear(8,8)
                self.actor.action_dim=5 if name=='routing' else 27
                self.critic=torch.nn.Linear(8,1)
                self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=1e-4)
                self.critic_optimizer=torch.optim.Adam(self.critic.parameters(),lr=1e-4)
                self.valuenorm='old'
        roles={name:Role(name) for name in ('routing','dispatch_front','dispatch_rear')}
        routing_actor=roles['routing'].actor
        original_routing_optim=roles['routing'].actor_optimizer
        model=types.SimpleNamespace(roles=roles,device=torch.device('cpu'),
                     cfg=types.SimpleNamespace(actor_lr=1e-4,critic_lr=1e-4))
        with patch.dict(sys.modules,{'thesis.mappo.algorithm':alg}):
            info=install_v14(model,20,27)
        self.assertEqual(set(info),set(roles))
        self.assertIs(model.roles['routing'].actor,routing_actor)
        self.assertIs(model.roles['routing'].actor_optimizer,original_routing_optim)
        for name,r in roles.items():
            self.assertIsInstance(r.critic,DeepCentralizedCritic)
            self.assertIsInstance(r.valuenorm,RunningValueNorm)
            if name!='routing':self.assertIsInstance(r.actor,CandidateAwareDispatchActor)

    def test_state_dict_strict_round_trip(self):
        original=CandidateAwareDispatchActor(20,27)
        new=CandidateAwareDispatchActor(20,27)
        new.load_state_dict(original.state_dict(),strict=True)
        for k,v in original.state_dict().items():
            self.assertTrue(torch.equal(v,new.state_dict()[k]),k)

    def test_defaults_scratch_gat4_3600(self):
        a=runner.parse_args([])
        self.assertEqual((a.gat_layers,a.gat_heads,a.dispatch_width,a.embedding),
                         (4,2,128,64))
        self.assertEqual((a.sim_seconds,a.max_minibatch,a.max_episodes),(3600,16,1100))
        self.assertFalse(a.resume)
        self.assertNotIn('V12_FULL_NO_DIJKSTRA_FROM_V11',str(runner.DEFAULT_OUT))
        self.assertIn('V14',str(runner.DEFAULT_OUT))

    def test_commit_log_preflight_and_expected_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp).resolve()
            log=root/'metrics.jsonl'
            log.write_text(''.join(json.dumps({'episode':i})+'\n' for i in (1,2,3)))
            runner.metrics_preflight(log,2)
            self.assertEqual(len(log.read_text().splitlines()),2)
            (root/'rolling_slot_0.pt').touch()
            ep,checkpoint=runner.check_state({'last_episode':2,
                        'checkpoint':str(root/'rolling_slot_0.pt')},root)
            self.assertEqual(ep,2)
            self.assertEqual(checkpoint.name,'rolling_slot_0.pt')
            with self.assertRaises(RuntimeError):
                runner.check_state({'last_episode':3,
                        'checkpoint':str(root/'rolling_slot_0.pt')},root)

    def test_dry_run_uses_immutable_blueprint_without_engine(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            model=root/'test_sim.zip';model.write_bytes(b'fake-dry-run-model')
            configdir=root/'config';configdir.mkdir()
            cfg={'model_file':str(model),'fleetsize':20,'max_fleetsize':20,
                 'routing_encoder':'residual_edge_gat','dispatch_mask_mode':'operation_first',
                 'graph_fingerprint':'testhash','max_events':1000,'seed':43,
                 'ppo_v7_options':{},'stop_mask_mode':'forbid_when_move'}
            (configdir/'run_config.json').write_text(json.dumps(cfg))
            runner.main(['--config-run',str(configdir),'--output',str(root/'newrun'),
                        '--dry-run','--device','cpu'])
            self.assertFalse((root/'newrun').exists())


if __name__=='__main__':unittest.main()
