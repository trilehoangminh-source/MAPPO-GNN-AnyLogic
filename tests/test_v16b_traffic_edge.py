"""Run offline: python -m pytest -q tests/test_v16b_traffic_edge.py

Requires a V15 project tree, PyTorch, pytest. No AnyLogic runtime needed.
"""
import types
import torch

from thesis.mappo.v15_tobias_arch import TobiasGraphRoutingActor
from thesis.mappo.v16b_traffic_edge import TrafficRoutingActor


# Nodes form directed triangle: 0->1, 1->2, 2->0; plus physical reverse 1->0.
# GraphSpec xy are normalized, with dummy node appended by original V15.
GRAPH = types.SimpleNamespace(xy=((.1,.1),(.4,.1),(.4,.4)),
                              arcs=((0,1),(1,0),(1,2),(2,0)),
                              fingerprint='V16B_TEST_TOPOLOGY')


def observation(batch=2, all_zero_other=False):
    agvs = torch.zeros((batch,20,34))
    agvs[:,0,0] = 1.
    agvs[:,0,2:4] = torch.tensor(GRAPH.xy[0])
    agvs[:,0,4:6] = torch.tensor(GRAPH.xy[1])
    agvs[:,0,6:8] = torch.tensor(GRAPH.xy[2])
    agvs[:,0,8:16] = torch.tensor([*GRAPH.xy[1],*GRAPH.xy[2],*GRAPH.xy[0],*GRAPH.xy[1]])
    agvs[:,0,18] = 1.
    # One OTHER moving AGV 0->1, one OTHER moving AGV 1->0
    if not all_zero_other:
        for index, src, dst in [(1,0,1),(2,1,0)]:
            agvs[:,index,0] = 1.
            agvs[:,index,1] = 1.
            agvs[:,index,2:4] = torch.tensor(GRAPH.xy[src])
            agvs[:,index,4:6] = torch.tensor(GRAPH.xy[dst])
            agvs[:,index,6:8] = torch.tensor(GRAPH.xy[dst])
            agvs[:,index,8:16] = agvs[:,0,8:16]
            agvs[:,index,18] = 1.
    stat = torch.zeros((batch,1,34))
    stat[:,:,0:2] = torch.tensor(GRAPH.xy[1])
    return {'agvs':agvs,'stat':stat,'action_mask':torch.ones(batch,5)}


def test_zero_edge_weights_exactly_v15():
    torch.manual_seed(23)
    before = TobiasGraphRoutingActor(GRAPH)
    after = TrafficRoutingActor(GRAPH)
    missing, extra = after.load_state_dict(before.state_dict(),strict=False)
    assert set(missing) == {'convs.0.edge_lin.weight','convs.1.edge_lin.weight'}
    assert not extra
    with torch.no_grad():
        for variant in (False,True):
            obs = observation(all_zero_other=variant)
            got = after.logits(obs)
            expected = before.logits(obs)
            torch.testing.assert_close(got,expected,atol=2e-6,rtol=2e-6)


def test_directed_features_and_dummy_ignored():
    actor = TrafficRoutingActor(GRAPH)
    obs = observation()
    attr = actor.traffic_features(obs)
    assert attr.shape == (2,actor.edge_index.shape[1],4)
    assert torch.isfinite(attr).all()
    src, dst = actor.edge_index[1],actor.edge_index[0]
    idx01 = ((src==0) & (dst==1)).nonzero().item()
    idx10 = ((src==1) & (dst==0)).nonzero().item()
    torch.testing.assert_close(attr[:,idx01,1], torch.full((2,),1/20))
    torch.testing.assert_close(attr[:,idx01,2], torch.full((2,),1/20))
    torch.testing.assert_close(attr[:,idx10,1], torch.full((2,),1/20))
    assert attr[:,idx01,0].min() > 0
    # next-node pressure 1 each for node 0 and node 1
    torch.testing.assert_close(attr[:,idx01,3],torch.full((2,),1/20))
    self_edge = src==dst
    assert torch.equal(attr[:,self_edge,:],torch.zeros_like(attr[:,self_edge,:]))
    empty = actor.traffic_features(observation(all_zero_other=True))
    assert torch.all(empty[...,1:]==0)


def test_traffic_attention_backprop_and_no_mask():
    actor = TrafficRoutingActor(GRAPH)
    obs = observation()
    logits = actor.logits(obs)
    loss = logits.square().sum()
    loss.backward()
    for conv in actor.convs:
        assert conv.edge_lin.weight.grad is not None
        assert torch.isfinite(conv.edge_lin.weight.grad).all()
    assert actor.uses_action_mask is False
    dist = actor.dist({**obs,'action_mask':torch.tensor([[1,0,0,0,0]]).expand(2,-1).float()})
    assert (dist.probs[:,1:] > 0).all() # intentionally UNMASKED routing


def test_padding_or_other_agent_order_do_not_change_features():
    actor = TrafficRoutingActor(GRAPH)
    obs = observation()
    canonical = actor.traffic_features(obs)
    reordered = {k:v.clone() for k,v in obs.items()}
    reordered['agvs'][:,[1,2,3]] = obs['agvs'][:,[3,1,2]]
    torch.testing.assert_close(canonical,actor.traffic_features(reordered))


def test_nonzero_edge_weights_make_policy_traffic_sensitive():
    actor = TrafficRoutingActor(GRAPH)
    with torch.no_grad():
        for conv in actor.convs:
            conv.edge_lin.weight.normal_(0., .4)
    obs_traffic = observation(all_zero_other=False)
    obs_empty = observation(all_zero_other=True)
    with torch.no_grad():
        logits_traffic = actor.logits(obs_traffic)
        logits_empty = actor.logits(obs_empty)
    # Node features themselves also differ: this simply verifies finite outputs
    # when edge term is enabled; attention-specific test below holds nodes fixed.
    assert torch.isfinite(logits_traffic).all()
    assert torch.isfinite(logits_empty).all()
    x = torch.randn((2,4,32))
    a = actor.traffic_features(obs_traffic)
    z = actor.traffic_features(obs_empty)
    o1 = actor.convs[0](x,actor.edge_index,a)
    o2 = actor.convs[0](x,actor.edge_index,z)
    assert not torch.allclose(o1,o2,atol=1e-8,rtol=1e-8)


def test_v15_warmstart_weights_and_optimizers(tmp_path):
    from thesis.mappo.v16b_traffic_edge import transfer_v15_checkpoint
    from types import SimpleNamespace
    from torch import nn

    class Role:
        def __init__(self, actor):
            self.actor = actor
            self.critic = nn.Linear(2,1)
            self.actor_optimizer = torch.optim.Adam(actor.parameters(),lr=1e-4)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr=1e-4)
            self.valuenorm = SimpleNamespace(
                state_dict=lambda: {'count': 1.}, load_state_dict=lambda x: None)

    v15_roles = {'routing':Role(TobiasGraphRoutingActor(GRAPH)),
                 'dispatch_front':Role(nn.Linear(2,3)),
                 'dispatch_rear':Role(nn.Linear(2,3))}
    checkpoint = tmp_path / 'v15.pt'
    torch.save({'roles':{k:{'actor':v.actor.state_dict(),
                           'critic':v.critic.state_dict(),
                           'actor_optimizer':v.actor_optimizer.state_dict(),
                           'critic_optimizer':v.critic_optimizer.state_dict(),
                           'valuenorm':v.valuenorm.state_dict()}
                         for k,v in v15_roles.items()}},checkpoint)
    model = SimpleNamespace(
        device=torch.device('cpu'), cfg=SimpleNamespace(actor_lr=1e-4),
        roles={'routing':Role(TrafficRoutingActor(GRAPH)),
               'dispatch_front':Role(nn.Linear(2,3)),
               'dispatch_rear':Role(nn.Linear(2,3))})
    mode = transfer_v15_checkpoint(model,checkpoint)
    assert mode == 'v15_compatible_weights_routing_actor_adam_reset'
    assert len(model.roles['routing'].actor_optimizer.state) == 0
    assert all(torch.count_nonzero(c.edge_lin.weight) == 0
               for c in model.roles['routing'].actor.convs)
    old = v15_roles['routing'].actor.state_dict()
    new = model.roles['routing'].actor.state_dict()
    for k,v in old.items():
        torch.testing.assert_close(v,new[k])
