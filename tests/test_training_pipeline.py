"""Offline training pipeline checks; no AnyLogic/Java/Ray/PyG needed."""
from __future__ import annotations
import importlib.util
import sys
import types
import unittest
import json
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def load(path,module):
    spec=importlib.util.spec_from_file_location(module,ROOT/path)
    m=importlib.util.module_from_spec(spec)
    sys.modules[module]=m
    spec.loader.exec_module(m)
    return m

stage=load(Path('scripts/stage_policy.py'),'stage_policy')
arch=load(Path('thesis/mappo/model_architecture.py'),'model_architecture')
builder=load(Path('scripts/model_builder.py'),'model_builder')

class Space:
    def __init__(self,shape):self.shape=shape

class Graph:
    xy=((.1,.1),(.2,.1),(.2,.2))
    arcs=((0,1),(1,2),(2,0))
    fingerprint='offline-fingerprint'


def observation(batch=2):
    agvs=torch.zeros((batch,20,19))
    stat=torch.zeros((batch,27,19))
    agvs[:,0,2:16]=torch.tensor([.1,.1,.1,.1,.2,.2,.2,.1,.1,.1,.2,.2,.1,.1])
    agvs[:,0,1]=1
    agvs[:,0,18]=.8
    stat[:,0,:2]=torch.tensor([.2,.1])
    return {'agvs':agvs,'stat':stat,'action_mask':torch.ones((batch,5))}

class TestTrainingPipeline(unittest.TestCase):
    def setUp(self):torch.manual_seed(2)

    def test_routing_tobias_two_gatv2_four_heads(self):
        net=arch.TobiasGraphRoutingActor(Graph())
        self.assertEqual(len(net.convs),2)
        self.assertEqual([x.heads for x in net.convs],[4,4])
        o=observation()
        o['action_mask'][:,3]=0
        p=net.dist(o).probs
        self.assertEqual(tuple(p.shape),(2,5))
        self.assertTrue(torch.isfinite(p).all())
        self.assertTrue(torch.all(p[:,3]>0))
        loss=net.logits(o).sum();loss.backward()
        self.assertTrue(all(torch.isfinite(g).all() for g in
            (x.grad for x in net.parameters()) if g is not None))

    def test_dispatch_linfe_branches_and_27_actions(self):
        s={'agvs':Space((20,19)),'stat':Space((27,19))}
        net=arch.TobiasLinearDispatchActor(s,27)
        self.assertEqual(len([x for x in net.main_ff if isinstance(x,torch.nn.Linear)]),4)
        self.assertEqual(len([x for x in net.other_ff if isinstance(x,torch.nn.Linear)]),4)
        self.assertEqual(len([x for x in net.station_ff if isinstance(x,torch.nn.Linear)]),4)
        obs=observation();obs['action_mask']=torch.ones((2,27))
        obs['action_mask'][:,13]=0
        p=net.dist(obs).probs
        self.assertEqual(tuple(p.shape),(2,27))
        self.assertEqual(float(p[:,13].detach().abs().max()),0)
        net.logits(obs).sum().backward()
        self.assertIsNotNone(net.action_net[0].weight.grad)

    def test_critic_global_state_20_padded(self):
        net=arch.V15CentralCritic(19)
        obs=observation()
        gs={'agvs':obs['agvs'],'stat':obs['stat'],'ego':torch.tensor([0,0])}
        result=net(gs)
        self.assertEqual(tuple(result.shape),(2,))
        result.sum().backward()
        self.assertIsNotNone(net.agv[0].weight.grad)

    def test_untrained_dispatch_is_mask_safe_random(self):
        rng=np.random.default_rng(5)
        obs={'action_mask':np.array([0,1,0,0,1],np.float32)}
        self.assertTrue(all(stage.masked_random_dispatch(obs,rng) in (1,4) for _ in range(100)))

    def test_no_geometric_router_for_dispatch_pretraining(self):
        self.assertFalse(hasattr(stage,'masked_geometric_routing'))
        self.assertIn('dispatch8',stage.PHASES)
        self.assertTrue('CollisionFreeRouting' in (ROOT/'scripts/model_builder.py').read_text(encoding='utf-8'))

    def test_dispatch8_refuses_routing_decisions_inside_ppo(self):
        fake=types.SimpleNamespace(roles={r:object() for r in ('routing','dispatch_front','dispatch_rear')},
                                   cfg=types.SimpleNamespace(two_fleets=True))
        adapter=stage.StagePolicyAdapter(fake,'dispatch8',seed=19)
        with self.assertRaisesRegex(RuntimeError,'CollisionFreeRouting'):
            adapter.act('routing',{'action_mask':np.array([1,1,0,0,0])},None)

    def test_native_router_confined_to_dispatch8(self):
        builder_text=(ROOT/'scripts/model_builder.py').read_text(encoding='utf-8')
        self.assertIn("use_conventional_routing = (phase == 'dispatch8')", builder_text)
        self.assertIn('pseudo_routing=use_conventional_routing', builder_text)
        self.assertIn('CollisionFreeRoutingHive', builder_text)
        runner_text=(ROOT/'scripts/train.py').read_text(encoding='utf-8')
        self.assertIn('phase=a.phase',runner_text)

    def test_phase_role_filters(self):
        def record(role):return {'role':role}
        seg=[[record('routing')],[record('dispatch_front')],[record('dispatch_rear')]]
        for phase,(_fleet,roles) in stage.PHASES.items():
            got={s[0]['role'] for s in stage.trained_trajectories(seg,phase)}
            self.assertEqual(got,set(roles))
        self.assertEqual([stage.PHASES[x][0] for x in stage.PHASES], [8,8,8,12,16,20])

    def test_sparse_gat_matches_dense_reference(self):
        g=arch.TobiasGATv2(embedding=4,heads=2)
        sample=torch.randn(3,5,4,requires_grad=True)
        pairs=torch.tensor([[0,1,1,2,2,3,3,4,4,0,1,2,3,4],
                            [0,0,1,1,2,2,3,3,4,4,0,1,2,3]],dtype=torch.long)
        result=g(sample,pairs)
        b,n,d=sample.shape;h=2
        left=g.linear_dst(sample).view(b,n,h,d)
        right=g.linear_src(sample).view(b,n,h,d)
        dense=torch.zeros_like(result)
        for rec in range(n):
            senders=pairs[1][pairs[0]==rec]
            scoring= torch.nn.functional.leaky_relu(left[:,rec,None]+right[:,senders],.2)
            scoring=(scoring*g.att[None,None]).sum(-1).softmax(1)
            dense[:,rec]=(scoring.unsqueeze(-1)*right[:,senders]).sum(1).mean(1)+g.bias
        self.assertTrue(torch.allclose(result,dense,atol=1e-5,rtol=1e-5))

    def test_stage_adapter_does_not_evaluate_inactive_policy(self):
        class Fake:
            def __init__(self):
                self.roles={r:object() for r in ('routing','dispatch_front','dispatch_rear')}
                self.cfg=types.SimpleNamespace(two_fleets=True)
                self.called=[]
            def act(self,role,obs,state,greedy=False):
                self.called.append(role)
                return 1,-.1,5.
            def value(self,role,state):return 5.
            def role_for(self,*args):return 'routing'
        m=Fake()
        adapter=stage.StagePolicyAdapter(m,'routing8',1)
        obs={'action_mask':np.array([1,1,1],np.float32)}
        self.assertEqual(adapter.act('dispatch_front',obs,None)[1:],(0.,0.))
        self.assertEqual(m.called,[])
        self.assertEqual(adapter.act('routing',obs,None)[0],1)
        self.assertEqual(m.called,['routing'])

    def test_builder_selects_native_router_only_in_dispatch8(self):
        from unittest.mock import patch
        created=[]
        class CollisionFreeRoutingHive: pass
        class NormalHive: pass
        class CollisionFreeRouting: pass
        class Shape:
            def __init__(self,shape): self.shape=shape
        class Matrix:
            def __init__(self,**kw):
                self.pseudo_routing=kw['pseudo_routing']
                self.agent_hive=(CollisionFreeRoutingHive() if self.pseudo_routing else NormalHive())
                self.agent_behaviors={str(i):CollisionFreeRouting() for i in range(kw['fleetsize'])}
                self.agents=[str(1000+i) for i in range(kw['fleetsize'])]
                self.agent_selection='1000'
                self.closed=False
                created.append(self)
            def reset(self,seed): pass
            def observation_space(self,selected):
                return {'agvs':Shape((20,19)),'stat':Shape((27,19))}
            def get_mappo_state(self):
                return {'agvs':np.zeros((20,19)),'stat':np.zeros((27,19))}
            def close(self): self.closed=True
        class MAPPOConfig:
            def __init__(self,**kw): self.kw=kw
        class MAPPO:
            def __init__(self,spaces,feat,cfg,device):
                self.roles={'routing':types.SimpleNamespace(actor=object()),
                            'dispatch_front':types.SimpleNamespace(actor=object()),
                            'dispatch_rear':types.SimpleNamespace(actor=object())}
        class GraphSpec:
            @staticmethod
            def from_hive(hive):
                return types.SimpleNamespace(fingerprint='offline-fingerprint')
        class V7Options:
            def __init__(self,**kwargs): pass
        def module(name,**attrs):
            m=types.ModuleType(name)
            m.__dict__.update(attrs)
            return m
        fakes={
            'thesis.envs.matrix': module('thesis.envs.matrix', Matrix=Matrix),
            'thesis.envs.behaviors': module('thesis.envs.behaviors',
                                                 CollisionFreeRouting=CollisionFreeRouting,
                                                 CollisionFreeRoutingHive=CollisionFreeRoutingHive),
            'thesis.mappo.algorithm': module('thesis.mappo.algorithm', MAPPO=MAPPO, MAPPOConfig=MAPPOConfig),
            'thesis.mappo.ppo_stability':module('thesis.mappo.ppo_stability',
                                                   enable_v7=lambda *x: None,V7Options=V7Options),
            'thesis.mappo.routing_graph':module('thesis.mappo.routing_graph', GraphSpec=GraphSpec),
            'thesis.mappo.model_architecture':module('thesis.mappo.model_architecture',
                                                install_base_architecture=lambda *x: {'mock':True},
                                                install_traffic_routing=lambda *x: {'mock':True},
                                                transfer_checkpoint=lambda *x: 'mock'),
        }
        # These are injected into the builder's local imports only; no AnyLogic needed.
        with patch.dict(sys.modules, fakes):
            base=dict(max_fleetsize=20,gamma=.9,gae_lambda=.95,clip=.1,lr=.0001,
                      epochs=5,mini_batches=1,ppo_v7_options={},
                      graph_fingerprint='offline-fingerprint')
            env,_,_,info=builder.build_model(base,model_path=__file__,fleet_size=8,port=51880,
                                            device='cpu',seed=1,max_events=1000,phase='dispatch8')
            self.assertTrue(env.pseudo_routing)
            self.assertTrue(info['native_conventional_routing'])
            self.assertEqual(info['routing_controller'],'tobias_collision_free_reservation')
            env.close()
            for name,fleet in [('routing8',8),('joint8',8),('joint12',12),('joint16',16),('joint20',20)]:
                env,_,_,info=builder.build_model(base,model_path=__file__,fleet_size=fleet,
                      port=51880,device='cpu',seed=1,max_events=1000,phase=name)
                self.assertFalse(env.pseudo_routing)
                self.assertFalse(info['native_conventional_routing'])
                self.assertEqual(info['routing_controller'],'traffic_edge_gatv2')
                env.close()
        self.assertEqual(len(created),6)

    def test_frozen_role_weight_audit_detects_mutation(self):
        sys.path.insert(0,str(ROOT/'scripts'))
        import train as runner
        class SmallRole:
            def __init__(self):
                self.actor=torch.nn.Linear(3,2)
                self.critic=torch.nn.Linear(3,1)
                self.valuenorm=types.SimpleNamespace(state_dict=lambda:{'mean':0.,'std':1.})
        model=types.SimpleNamespace(roles={'routing':SmallRole(),
                                           'dispatch_front':SmallRole(),
                                           'dispatch_rear':SmallRole()})
        old=runner._frozen_role_hashes(model,('routing',))
        self.assertEqual(set(old),{'dispatch_front','dispatch_rear'})
        with torch.no_grad():
            model.roles['dispatch_front'].actor.weight[0,0]+=1.
        new=runner._frozen_role_hashes(model,('routing',))
        self.assertNotEqual(old,new)

    def test_source_reward_no_V14_shaping(self):
        rewards=builder.TOBIAS_JOINT_REWARDS
        self.assertEqual(rewards['reward_removed_for_block'],-20)
        self.assertEqual(rewards['reward_pass'],.2)
        self.assertEqual(rewards['reward_duration'],-.5)
        self.assertEqual(rewards['block_timeout'],60)
        self.assertNotIn('reward_part_completed',rewards)
        self.assertEqual(rewards['routing_interval'],2)

    def test_tobias_original_dispatch_mask_selected(self):
        self.assertEqual(builder.TOBIAS_DISPATCH_MASK_MODE,'legacy')
        baseline=(ROOT/'thesis/envs/behaviors.py').read_text(encoding='utf-8')
        self.assertIn('possible_stations.extend(self.part.proc_stat[proc])',baseline)
        self.assertIn('possible_station_nodes = [self.part.stat_node[stat] for stat in possible_stations] + self.part.stat_node["puffer"]',baseline)
        for p in ('scripts/train.py',
                  'scripts/evaluate.py'):
            content=(ROOT/p).read_text(encoding='utf-8')
            self.assertIn("os.environ['MAPPO_DISPATCH_MASK_MODE']=TOBIAS_DISPATCH_MASK_MODE",content)
            self.assertNotIn("os.environ['MAPPO_DISPATCH_MASK_MODE']='operation_first'",content)

    def test_joint16_checkpoint_option(self):
        runner=load(Path('scripts/train.py'),'train')
        cfg=runner.parse_args(['--phase','joint16','--max-episodes','1'])
        self.assertIsNone(cfg.init_checkpoint)
        self.assertIn('TRAFFIC_EDGE_MESSAGE_GA512',str(cfg.output_root))
        self.assertRaises(SystemExit,runner.parse_args,
                          ['--phase','joint8','--init-checkpoint','missing.pt'])

    def test_joint20_checkpoint_initialization_and_resume_are_distinct(self):
        from tempfile import TemporaryDirectory
        runner=load(Path('scripts/train.py'),'train')
        with TemporaryDirectory() as folder:
            checkpoint=Path(folder)/'snapshot_ep_0490.pt'
            checkpoint.write_bytes(b'parser fixture')
            args=['--phase','joint20','--port','0',
                  '--init-checkpoint',str(checkpoint)]
            cfg=runner.parse_args(args)
            self.assertEqual(cfg.init_checkpoint,checkpoint)
            self.assertEqual(stage.PHASES[cfg.phase][0],20)
            self.assertFalse(cfg.resume)
            self.assertRaises(SystemExit,runner.parse_args,args+['--resume'])
            self.assertRaises(SystemExit,runner.parse_args,
                              ['--phase','joint12','--init-checkpoint',str(checkpoint)])
            resumed=runner.parse_args(['--phase','joint20','--resume','--port','0'])
            self.assertTrue(resumed.resume)
            self.assertIsNone(resumed.init_checkpoint)

    def test_training_rejects_ports_outside_tcp_range(self):
        runner=load(Path('scripts/train.py'),'train')
        for port in ('-1','65536'):
            with self.subTest(port=port):
                self.assertRaises(SystemExit,runner.parse_args,
                                  ['--phase','joint20','--port',port])

    def test_source_does_not_import_guidance(self):
        content=(ROOT/'scripts/model_builder.py').read_text(encoding='utf-8')
        self.assertNotIn('from thesis.mappo.routing_guidance import',content)
        self.assertNotIn('from thesis.mappo.v14_deep_candidate_mlp import',content)
        self.assertNotIn('install_v12(',content)
        self.assertNotIn('masked_geometric_routing', (ROOT/'scripts/stage_policy.py').read_text(encoding='utf-8'))

if __name__=='__main__':unittest.main()
