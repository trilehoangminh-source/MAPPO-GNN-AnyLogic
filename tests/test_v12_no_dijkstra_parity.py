"""Offline regression tests for V12 no-shortest-path PPO parity. No AnyLogic."""
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from v12_clean_audit import verify_v12_ppo_rollout_parity


class FakeActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0.5, 1., -0.2, 0.8, 0.1]))

    def dist(self, obs):
        batch = obs['action_mask'].shape[0]
        scores = self.logits.unsqueeze(0).expand(batch, -1)
        scores = scores.masked_fill(obs['action_mask'] <= .5, -1.e9)
        return Categorical(logits=scores)


def sample(model, action=1):
    obs = dict(agvs=torch.zeros(2, 16).numpy(),
               stat=torch.zeros(2, 16).numpy(),
               action_mask=torch.tensor([0., 1., 1., 1., 0.]).numpy())
    logits = model.roles['routing'].actor.dist({k:torch.as_tensor(v).unsqueeze(0)
                                               for k,v in obs.items()})
    logp = float(logits.log_prob(torch.tensor([action])).item())
    return dict(role='routing', observation=obs, action=action, logp=logp)


class Tests(unittest.TestCase):
    def setUp(self):
        self.actor = FakeActor()
        self.model = SimpleNamespace(device='cpu', roles={'routing':SimpleNamespace(actor=self.actor)})

    def test_valid_no_bias_parity(self):
        record = verify_v12_ppo_rollout_parity(self.model, [[sample(self.model)]])
        self.assertTrue(record['passed'])
        self.assertEqual(record['compared'], 1)
        self.assertFalse(record['dijkstra_guidance_required'])

    def test_old_guidance_not_silently_allowed(self):
        entry = sample(self.model)
        entry['observation']['routing_guide_bias'] = torch.zeros(5).numpy()
        with self.assertRaisesRegex(RuntimeError, 'legacy Dijkstra bias'):
            verify_v12_ppo_rollout_parity(self.model, [[entry]])

    def test_illegal_action_rejected(self):
        entry = sample(self.model)
        entry['action'] = 0  # mask is zero at action 0
        with self.assertRaisesRegex(RuntimeError, 'logprob mismatch'):
            verify_v12_ppo_rollout_parity(self.model, [[entry]])

    def test_corrupt_logged_logp_rejected(self):
        entry = sample(self.model)
        entry['logp'] += 0.2
        with self.assertRaisesRegex(RuntimeError, 'logprob mismatch'):
            verify_v12_ppo_rollout_parity(self.model, [[entry]])

    def test_no_routing_samples_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'no Routing samples'):
            verify_v12_ppo_rollout_parity(self.model, [[dict(role='dispatch_front')]])

    def test_actor_mode_restored(self):
        self.actor.train(True)
        verify_v12_ppo_rollout_parity(self.model, [[sample(self.model)]])
        self.assertTrue(self.actor.training)
        self.actor.train(False)
        verify_v12_ppo_rollout_parity(self.model, [[sample(self.model)]])
        self.assertFalse(self.actor.training)

if __name__ == '__main__':
    unittest.main()
