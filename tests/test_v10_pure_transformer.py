"""Offline CPU tests, no AnyLogic required. Run from thesis root."""
import sys
import types
import unittest
import torch
from torch import nn

try:
    import thesis.mappo.networks
except ModuleNotFoundError:
    mod = types.ModuleType('thesis.mappo.networks')
    def masked_logits(logits, mask):
        if not (mask > 0.5).any(-1).all():
            raise ValueError('No valid action')
        return logits.masked_fill(mask <= 0.5, torch.finfo(logits.dtype).min/2)
    mod.masked_logits = masked_logits
    sys.modules['thesis.mappo.networks'] = mod

from thesis.mappo.v10_pure_transformer import (
    PureStationTransformerActor, PureTransformerCentralCritic,
)


class TestPureTransformer(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)
        self.obs = {
            'agvs':torch.randn(3,20,10),
            'stat':torch.randn(3,27,10),
            'action_mask':torch.zeros(3,27),
        }
        self.obs['agvs'][:,8:] = 0
        self.obs['stat'][:,14:] = 0
        self.obs['action_mask'][:,[0,2,4,7,11,15]] = 1
        self.obs['action_mask'][1,:] = 0
        self.obs['action_mask'][1,24] = 1  # legal despite missing station features
        self.gs = {
            'agvs': self.obs['agvs'].clone(), 'stat':self.obs['stat'].clone(),
            'ego':torch.tensor([0,3,5],dtype=torch.long),
        }

    def test_pure_actor_mask_and_random_initial_policy(self):
        actor = PureStationTransformerActor(10,27,32,2,1).eval()
        prob = actor.dist(self.obs).probs
        self.assertEqual(tuple(prob.shape),(3,27))
        self.assertTrue(torch.isfinite(prob).all())
        self.assertTrue(torch.all(prob[self.obs['action_mask']==0]==0))
        self.assertAlmostEqual(float(prob[1,24].detach()),1,places=6)
        self.assertTrue(torch.allclose(prob.sum(-1),torch.ones(3),atol=1e-6))
        self.assertFalse(any('base.' in k or '.policy.' in k for k in actor.state_dict()))

    def test_pure_critic_shape(self):
        critic=PureTransformerCentralCritic(10,27).eval()
        value=critic(self.gs)
        self.assertEqual(tuple(value.shape),(3,))
        self.assertTrue(torch.isfinite(value).all())
        self.assertFalse(any('base.' in k for k in critic.state_dict()))

    def test_gradients_nonzero_and_checkpoint_roundtrip(self):
        actor=PureStationTransformerActor(10,27)
        critic=PureTransformerCentralCritic(10,27)
        choices=torch.tensor([2,24,7])
        loss=(-actor.dist(self.obs).log_prob(choices).mean()
              +critic(self.gs).square().mean())
        loss.backward()
        self.assertGreater(actor.encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum().item(),0)
        self.assertGreater(actor.pair_scorer[-1].weight.grad.abs().sum().item(),0)
        self.assertGreater(critic.encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum().item(),0)
        opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=1e-3)
        opt.step()
        actor2=PureStationTransformerActor(10,27)
        actor2.load_state_dict(actor.state_dict(),strict=True)
        actor.eval();actor2.eval()
        torch.testing.assert_close(actor.dist(self.obs).probs,actor2.dist(self.obs).probs)

    def test_actor_other_agv_permutation_equivariance(self):
        actor=PureStationTransformerActor(10,27).eval()
        obs2={k:v.clone() for k,v in self.obs.items()}
        ordering=[0]+list(range(7,0,-1))+list(range(8,20))
        obs2['agvs']=obs2['agvs'][:,ordering]
        with torch.no_grad():
            p=actor.dist(self.obs).probs
            p2=actor.dist(obs2).probs
        torch.testing.assert_close(p,p2,atol=2e-6,rtol=2e-5)

    def test_all_actions_invalid_rejected(self):
        actor=PureStationTransformerActor(10,27)
        bad={k:v.clone() for k,v in self.obs.items()}
        bad['action_mask'][0]=0
        with self.assertRaises(ValueError):
            actor.dist(bad)


if __name__=='__main__':
    unittest.main()
