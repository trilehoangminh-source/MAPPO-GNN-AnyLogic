"""CPU offline tests. No AnyLogic or Alpyne server needed."""
import unittest
import sys
import types
import torch
from torch import nn
from torch.distributions import Categorical

# Offline unit tests can run without the full AnyLogic-dependent V8 repository.
# On the user computer the actual thesis.mappo.networks.masked_logits is used.
try:
    import thesis.mappo.networks
except ModuleNotFoundError:
    mod = types.ModuleType("thesis.mappo.networks")
    def masked_logits(logits, mask):
        valid = mask > 0.5
        if (~valid.any(-1)).any():
            raise ValueError("No valid actions")
        return logits.masked_fill(~valid, torch.finfo(logits.dtype).min / 2)
    mod.masked_logits = masked_logits
    sys.modules["thesis.mappo.networks"] = mod
from thesis.mappo.v10_transformer import (
    ResidualStationTransformerActor, ResidualTransformerCentralCritic,
)

class FakeBase(nn.Module):
    def __init__(self,f=6,stations=27):
        super().__init__()
        self.encoder=nn.Module()
        self.encoder.main=nn.Sequential(nn.Linear(f,32),nn.Tanh())
        self.policy=nn.Linear(32,stations)
    def dist(self,obs):
        logits=self.policy(self.encoder.main(obs['agvs'][:,0,:]))
        masked=logits.masked_fill(obs['action_mask']<=0,torch.finfo(logits.dtype).min/2)
        return Categorical(logits=masked)

class AdapterFriendlyBase(FakeBase):
    def __init__(self,f=6,stations=27):
        super().__init__(f,stations)
        class Ego(nn.Module):
            def __init__(self,net):
                super().__init__();self.main=net
            def forward(self,obs): return self.main(obs['agvs'][:,0,:])
        self.encoder=Ego(self.encoder.main)

class FakeCritic(nn.Module):
    def __init__(self,f=6):
        super().__init__();self.linear=nn.Linear(f,1)
    def forward(self,state):
        idx=state['ego']
        return self.linear(state['agvs'][torch.arange(len(idx)),idx]).squeeze(-1)

class TestV10(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(11)
        self.obs={
            'agvs': torch.randn(3,20,6),
            'stat': torch.randn(3,27,6),
            'action_mask': torch.zeros(3,27),
        }
        self.obs['agvs'][:,3:,:]=0
        self.obs['stat'][:,12:,:]=0
        self.obs['action_mask'][:,[0,2,4,6]]=1
        self.obs['action_mask'][1]=0
        self.obs['action_mask'][1,5]=1  # One allowed action, even if station features zero.
        self.obs['stat'][:,5,:]=0
        self.gs={'agvs':self.obs['agvs'],'stat':self.obs['stat'],
                 'ego':torch.tensor([0,1,2])}

    def test_actor_exact_initial_parity(self):
        base=AdapterFriendlyBase()
        old=base.dist(self.obs).probs.detach().clone()
        new=ResidualStationTransformerActor(base,6,27,32,2,1)
        new.eval()
        now=new.dist(self.obs)
        torch.testing.assert_close(now.probs,old,atol=1e-7,rtol=1e-6)
        self.assertTrue(torch.all(now.probs[self.obs['action_mask']==0]==0))
        self.assertEqual(now.probs[1,5].item(),1.)
        self.assertTrue(torch.isfinite(now.entropy()).all())

    def test_critic_exact_initial_parity(self):
        base=FakeCritic()
        old=base(self.gs).detach().clone()
        new=ResidualTransformerCentralCritic(base,6,32,2,1)
        new.eval()
        torch.testing.assert_close(new(self.gs),old,atol=1e-7,rtol=1e-6)

    def test_gradient_and_save_load(self):
        act=ResidualStationTransformerActor(AdapterFriendlyBase(),6,27)
        critic=ResidualTransformerCentralCritic(FakeCritic(),6)
        loss=-act.dist(self.obs).log_prob(torch.tensor([2,5,6])).mean()+critic(self.gs).square().mean()
        loss.backward()
        self.assertIsNotNone(act.station_head[-1].weight.grad)
        self.assertGreater(act.station_head[-1].weight.grad.abs().sum().item(),0)
        self.assertIsNotNone(critic.value_head[-1].weight.grad)
        opt=torch.optim.Adam(list(act.parameters())+list(critic.parameters()),lr=1e-3)
        opt.step()
        other=ResidualStationTransformerActor(AdapterFriendlyBase(),6,27)
        other.load_state_dict(act.state_dict(),strict=True)
        torch.testing.assert_close(act.dist(self.obs).probs,other.dist(self.obs).probs)

if __name__=='__main__': unittest.main()
