"""Offline fast tests; no AnyLogic required. Run inside user's existing repo."""
import unittest
import torch
from torch import nn
from thesis.mappo.congestion_edgegat_v11 import (ObservationTrafficFeatures,
               install_v11, synthetic_parity_probe, TRAFFIC_NAMES)
from thesis.mappo.edge_gat_routing_v4 import EdgeAwareLayer, ResidualEdgeGAT, compute_static_edge_features

class TestTrafficV11(unittest.TestCase):
    def make_graph(self):
        # 4-node fully-connected directed toy graph; nonzero coordinates.
        xy=torch.tensor([[.1,.1],[.2,.1],[.3,.1],[.4,.1]])
        adjacency=torch.ones((4,4))-torch.eye(4)
        return xy,adjacency/3.

    def test_traffic_occupancy_flow_queue(self):
        xy,adj=self.make_graph()
        ext=ObservationTrafficFeatures(xy,adj,horizon=3,decay=.75)
        obs=torch.zeros((1,4,18))
        # AGV0: traveling 0->1, target 3; AGV1: waiting at 1, target 3.
        obs[0,:2,0]=1
        obs[0,0,1]=1
        obs[0,0,2:4]=xy[0];obs[0,0,4:6]=xy[1];obs[0,0,6:8]=xy[3]
        obs[0,1,2:4]=xy[1];obs[0,1,4:6]=xy[1];obs[0,1,6:8]=xy[3]
        feat=ext({'agvs':obs})
        self.assertEqual(tuple(feat.shape),(1,4,4,6))
        self.assertGreater(feat[0,0,1,0].item(),0)
        self.assertGreater(feat[0,1,3,1].item(),0)
        self.assertGreater(feat[0,1,3,2].item(),0)
        self.assertTrue(torch.isfinite(feat).all())
        self.assertTrue((feat[:,torch.eye(4,dtype=torch.bool)]==0).all())

    def test_traffic_backprop_and_old_weights_preserved(self):
        xy,adj=self.make_graph()
        from types import SimpleNamespace
        from thesis.mappo.networks import EntityEncoder
        class Actor(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer('node_xy',xy)
                self.register_buffer('outgoing_average',adj)
                self.entity_encoder=EntityEncoder(18,hidden=8)
                edges=compute_static_edge_features(xy,adj)
                self.layers=nn.ModuleList([EdgeAwareLayer(8,2,edges)])
                self.node_in=nn.Linear(2,8)
            def logits(self, obs):
                nodes=torch.tanh(self.node_in(self.node_xy)).unsqueeze(0).expand(obs['agvs'].shape[0],-1,-1)
                for layer in self.layers:
                    nodes=layer(nodes,self.outgoing_average)
                return nodes.sum(dim=1)[:,:5]
        actor=Actor()
        base_parms={n:p.clone() for n,p in actor.named_parameters()}
        a=torch.zeros((1,4,18))
        a[0,:2,0]=1
        a[0,0,1]=1
        a[0,0,2:4]=xy[0];a[0,0,4:6]=xy[1];a[0,0,6:8]=xy[3]
        a[0,1,2:4]=xy[1];a[0,1,4:6]=xy[1];a[0,1,6:8]=xy[2]
        obs={'agvs':a,'stat':torch.zeros((1,1,18)),'action_mask':torch.ones((1,5))}
        old=actor.logits(obs).detach()
        optim=torch.optim.Adam(actor.parameters(),lr=1e-3)
        fake=SimpleNamespace(roles={'routing':SimpleNamespace(actor=actor,actor_optimizer=optim)})
        info=install_v11(fake,horizon=3)
        self.assertEqual(info['new_parameter_tensors'],2)
        self.assertEqual(len(optim.param_groups),2)
        new=actor.logits(obs)
        self.assertLess((new-old).abs().max().item(),1e-6)
        for name,oldparam in base_parms.items():
            self.assertTrue(torch.equal(dict(actor.named_parameters())[name],oldparam),name)
        # Zero-weight branch initially has non-zero gradient so it can learn.
        (-new.square().mean()).backward()
        for name in ('traffic_score','traffic_value'):
            grad=getattr(actor.layers[0].attention,name).weight.grad
            self.assertIsNotNone(grad,name)
            self.assertTrue(torch.isfinite(grad).all(),name)

class TestEpisodeLocator(unittest.TestCase):
    def test_immutable_ep304(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from scripts.v11_common import locate_source_episode
        with TemporaryDirectory() as t:
            root=Path(t)
            chk=root/'snapshot_ep_0304.pt'
            chk.write_bytes(b'test dummy checkpoint name')
            self.assertEqual(locate_source_episode(root,304),chk.resolve())
            with self.assertRaises((FileNotFoundError,RuntimeError)):
                locate_source_episode(root,300)
            with self.assertRaises(RuntimeError):
                locate_source_episode(root,300,chk)

if __name__=='__main__': unittest.main()
