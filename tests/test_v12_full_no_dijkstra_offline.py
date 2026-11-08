"""Offline functional checks for V12 full removal; no AnyLogic engine needed."""
import importlib.util
import sys
import unittest
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from thesis.mappo.congestion_edgegat_v12_clean import (ObservationOnlyTraffic,
                                                       CongestionResidualEdgeGAT,
                                                       TRAFFIC_NAMES)


class TestObservedTraffic(unittest.TestCase):
    def setUp(self):
        self.xy = torch.tensor([[0.,0.],[1.,0.],[2.,0.]])
        self.arc = torch.tensor([[0,1,0],[1,0,1],[0,1,0]])
        self.feat = ObservationOnlyTraffic(self.xy,self.arc)
        self.agvs = torch.zeros((1,3,16))
        # First moving vehicle on 0->1; next=1, target=2.
        self.agvs[0,0,0]=1; self.agvs[0,0,1]=1
        self.agvs[0,0,2:4]=self.xy[0];self.agvs[0,0,4:6]=self.xy[1];self.agvs[0,0,6:8]=self.xy[2]
        # Second vehicle waiting at node1, with no assigned edge.
        self.agvs[0,1,0]=1
        self.agvs[0,1,2:4]=self.xy[1]; self.agvs[0,1,4:6]=self.xy[1]
        self.agvs[0,1,6:8]=self.xy[0]
        # third is padded.

    def test_no_path_table_or_shortest_path_buffers(self):
        self.assertFalse(hasattr(self.feat,'next_hop'))
        self.assertEqual(set(self.feat.state_dict()),{'node_xy','arc_mask'})
        self.assertNotIn('predicted_flow',TRAFFIC_NAMES)

    def test_current_edge_occupancy_and_local_waiting(self):
        traffic=self.feat({'agvs':self.agvs})
        self.assertEqual(tuple(traffic.shape),(1,3,3,6))
        self.assertAlmostEqual(float(traffic[0,0,1,0]),1/3)
        self.assertAlmostEqual(float(traffic[0,0,1,1]),1/3)
        self.assertAlmostEqual(float(traffic[0,1,2,2]),1/3)
        self.assertAlmostEqual(float(traffic[0,0,1,3]),1/3)
        self.assertAlmostEqual(float(traffic[0,0,1,4]),-1/3)
        self.assertAlmostEqual(float(traffic[0,0,2].abs().sum()),0)

    def test_features_independent_of_target_and_unobserved_future(self):
        a=self.feat({'agvs':self.agvs})
        changed=self.agvs.clone()
        changed[:,:,6:8]=987654.0  # target coordinates irrelevant
        b=self.feat({'agvs':changed})
        self.assertTrue(torch.equal(a,b))

    def test_no_path_estimates_from_target_only(self):
        changed=self.agvs.clone()
        changed[0,0,4:6]=self.xy[0]  # no next waypoint; target still node2
        traffic=self.feat({'agvs':changed})
        self.assertEqual(float(traffic[0,:,:,1].sum()),0.)

    def test_trainable_layers_six_compatible_channels_and_grad(self):
        class Pre(nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden=8;self.heads=2;self.head_dim=4
                self.q=nn.Linear(8,8);self.k=nn.Linear(8,8);self.v=nn.Linear(8,8)
                self.edge_score=nn.Linear(3,2);self.edge_value=nn.Linear(3,8)
                self.output=nn.Linear(8,8);self.norm=nn.LayerNorm(8)
        old=CongestionResidualEdgeGAT(Pre())
        with torch.no_grad():
            old.traffic_score.weight.fill_(0.02)
            old.traffic_value.weight.fill_(0.03)
        self.assertEqual(old.traffic_score.weight.shape,(2,6))
        self.assertEqual(old.traffic_value.weight.shape,(8,6))
        self.assertTrue(old.traffic_score.weight.requires_grad)
        self.assertTrue(old.traffic_value.weight.requires_grad)


if __name__ == '__main__': unittest.main()

class TestStrictV11Migration(unittest.TestCase):
    def test_old_next_hop_removed_no_trainable_weight_or_adam_reset(self):
        import tempfile
        import types
        from types import SimpleNamespace
        # v11_common is already installed in the user's project; use a tiny
        # import stub to isolate this test from the AnyLogic runtime.
        stub = sys.modules.get('v11_common')
        if stub is None:
            mock = types.ModuleType('v11_common')
            mock.validate_v8_cfg = lambda _: None
            sys.modules['v11_common'] = mock
        try:
            import v12_clean_common
            def make_actor(old):
                actor=nn.Module()
                actor.layer=nn.Linear(4,2)
                actor.v11_traffic=nn.Module()
                actor.v11_traffic.register_buffer('node_xy',torch.zeros(3,2))
                actor.v11_traffic.register_buffer('arc_mask',torch.zeros(3,3,dtype=torch.bool))
                if old:
                    actor.v11_traffic.register_buffer('next_hop',torch.arange(9).reshape(3,3))
                return actor
            def make_model(old):
                roles={}
                for name in ('routing','dispatch_front','dispatch_rear'):
                    actor=make_actor(old) if name=='routing' else nn.Linear(4,2)
                    critic=nn.Linear(4,1)
                    aopt=torch.optim.Adam(actor.parameters(),lr=1e-3)
                    copt=torch.optim.Adam(critic.parameters(),lr=1e-3)
                    value=nn.LayerNorm(1)
                    # Populate optimizer state so the test checks preservation.
                    for p in actor.parameters(): p.grad=torch.ones_like(p)
                    for p in critic.parameters(): p.grad=torch.ones_like(p)
                    aopt.step(); copt.step()
                    roles[name]=SimpleNamespace(actor=actor,critic=critic,
                        actor_optimizer=aopt,critic_optimizer=copt,valuenorm=value)
                return SimpleNamespace(roles=roles,device='cpu')
            old=make_model(True)
            new=make_model(False)
            data={'roles':{name:{
                'actor':r.actor.state_dict(),'critic':r.critic.state_dict(),
                'actor_optimizer':r.actor_optimizer.state_dict(),
                'critic_optimizer':r.critic_optimizer.state_dict(),
                'valuenorm':r.valuenorm.state_dict()} for name,r in old.roles.items()}}
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'source_v11.pt'; torch.save(data,p)
                info=v12_clean_common.load_clean_checkpoint(new,p,source_v11=True)
            self.assertEqual(info['removed_buffers'],1)
            self.assertNotIn('v11_traffic.next_hop',new.roles['routing'].actor.state_dict())
            for name, r in new.roles.items():
                old_r=old.roles[name]
                for key, val in r.actor.state_dict().items():
                    self.assertTrue(torch.equal(val,old_r.actor.state_dict()[key]))
                adam_new=r.actor_optimizer.state_dict()['state']
                adam_old=old_r.actor_optimizer.state_dict()['state']
                self.assertEqual(set(adam_new),set(adam_old))
                for pid in adam_new:
                    for key,val in adam_new[pid].items():
                        self.assertTrue(torch.equal(val,adam_old[pid][key]))
        finally:
            if stub is None: sys.modules.pop('v11_common',None)
