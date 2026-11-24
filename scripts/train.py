"""Train the traffic-aware MAPPO model with full-rollout gradient accumulation.

No change to thesis/mappo/algorithm.py, ppo_stability.py, collector, or Java.
No Dijkstra guidance in the learned Routing GAT, no reward shaping,
no V14 deep GAT or candidate scoring. Dispatch-only uses native Tobias
CollisionFreeRouting with time-window reservations (not GAT guidance).

Phase order: routing8 -> dispatch8 -> joint8 -> joint12 -> joint16 -> joint20
Each phase starts at its own EP0. Every next phase inherits full role weights,
critics, ValueNorm and Adam optimizers from the preceding phase checkpoint.
"""
#===================== MODULE 00 - IMPORT, ĐƯỜNG DẪN VÀ HẰNG SỐ =====================
# Mục đích: Nạp thư viện, định vị thư mục thesis và liệt kê phase training.
# Giải thích: Các đường dẫn module sẽ được băm SHA256 để kiểm tra tính nguyên vẹn.

from __future__ import annotations
import argparse
import hashlib
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from stage_policy import PHASES, StagePolicyAdapter, trained_trajectories
from thesis.mappo.fullrollout_optimizer import enable_fullrollout
from thesis.mappo.model_architecture import EDGE_FEATURE_NAMES

VERSION='V17B_EDGE7_MESSAGE_GATV2_FULLROLLOUT_GA512_V1'
PHASE_ORDER=tuple(PHASES)
DEFAULT_OUT=ROOT/'mappo_runs'/'TRAFFIC_EDGE_MESSAGE_GA512'
MODEL_MODULE=ROOT/'thesis'/'mappo'/'model_architecture.py'
BUILD_MODULE=ROOT/'scripts'/'model_builder.py'
STAGE_MODULE=ROOT/'scripts'/'stage_policy.py'
OPTIMIZER_MODULE=ROOT/'thesis'/'mappo'/'fullrollout_optimizer.py'



#================= KẾT THÚC MODULE 00 - IMPORT, ĐƯỜNG DẪN VÀ HẰNG SỐ =================




#======================= MODULE 01 - TÍNH SHA256 CỦA MỘT FILE =======================
# Mục đích: Đọc file theo khối và tính mã băm kiểm tra thay đổi nội dung.
# Giải thích: Tương tự tính CRC/checksum nhưng dùng SHA256.

def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for c in iter(lambda:stream.read(1024*1024),b''):
            h.update(c)
    return h.hexdigest()

#=================== KẾT THÚC MODULE 01 - TÍNH SHA256 CỦA MỘT FILE ===================






#=========================== MODULE 02 - ĐỌC DỮ LIỆU JSON ===========================
# Mục đích: Đọc text UTF-8 và chuyển thành Python dict/list.
# Giải thích: Ví dụ state["last_episode"] tương tự truy cập trường trạng thái.

def _read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))

#======================= KẾT THÚC MODULE 02 - ĐỌC DỮ LIỆU JSON =======================






#=========================== MODULE 03 - GHI JSON AN TOÀN ===========================
# Mục đích: Ghi file tạm .tmp rồi thay thế file chính.
# Giải thích: Hạn chế file state.json bị hỏng khi chương trình gián đoạn.

def _atomic_json(path,data):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    os.replace(tmp,p)

#======================= KẾT THÚC MODULE 03 - GHI JSON AN TOÀN =======================






#======================== MODULE 04 - CHỌN BLUEPRINT CẤU HÌNH ========================
# Mục đích: Ưu tiên --blueprint; nếu không có, thử các đường dẫn dự án cũ.
# Giải thích: Trả về đường dẫn config chứa siêu tham số và AnyLogic model.

def _pick_blueprint(provided):
    if provided is not None: return Path(provided).resolve()
    options=[
        ROOT/'mappo_runs'/'V14_GAT4_MLP_3600S_MB128_WAIT_SWING_REWARD'/'run_config_blueprint.json',
        ROOT/'mappo_runs'/'V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK'/'run_config_blueprint.json',
        ROOT/'mappo_runs'/'V14_GAT4_MLP_3600S_MB128_STOP_UNLOCK'/'run_config.json',
    ]
    for p in options:
        if p.is_file():return p.resolve()
    raise FileNotFoundError('Missing topology/AnyLogic blueprint. Specify --blueprint PATH_TO_V14/run_config_blueprint.json')

#=================== KẾT THÚC MODULE 04 - CHỌN BLUEPRINT CẤU HÌNH ===================






#=================== MODULE 05 - ĐỌC THAM SỐ TRAINING TỪ TERMINAL ===================
# Mục đích: argparse phân tích phase, số episode, checkpoint, resume, device và seed.
# Giải thích: Đồng thời kiểm tra tổ hợp tham số không hợp lệ.

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=PHASE_ORDER,required=True)
    p.add_argument('--max-episodes',type=int,default=1,help='Max completed EP in THIS phase, beginning from 0')
    p.add_argument('--blueprint',type=Path,default=None)
    p.add_argument('--model',type=Path,default=None,help='Existing original AnyLogic simulator .zip; overrides blueprint model_file')
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUT)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--port',type=int,default=51880)
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--max-events',type=int,default=None)
    p.add_argument('--microbatch-size',dest='max_minibatch',type=int,default=512)
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--snapshot-every',type=int,default=10)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--init-only',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--joint-routing-ep',type=int,default=None,help='For fresh joint8 only: routing8 snapshot episode')
    p.add_argument('--joint-dispatch-ep',type=int,default=None,help='For fresh joint8 only: dispatch8 snapshot episode')
    p.add_argument('--init-checkpoint',type=Path,default=None,
                   help='For fresh joint16 EP0: old joint16/initial_ep_0000.pt (or compatible checkpoint)')
    a=p.parse_args(argv)
    if a.max_episodes<0 or a.max_minibatch<1 or a.cpu_threads<1 or a.snapshot_every<1 or not 1<=a.port<=65535:
        p.error('Invalid episode/minibatch/threads/snapshot/port')
    if a.resume and a.init_only:p.error('--resume and --init-only are mutually exclusive')
    if a.init_checkpoint is not None:
        if a.phase!='joint16':p.error('--init-checkpoint works only for joint16')
        if a.resume:p.error('Do not combine --init-checkpoint and --resume')
        if not a.init_checkpoint.is_file():p.error('Missing --init-checkpoint file')
    chosen=(a.joint_routing_ep is not None,a.joint_dispatch_ep is not None)
    if chosen[0]!=chosen[1]:p.error('Provide BOTH --joint-routing-ep and --joint-dispatch-ep')
    if any(chosen):
        if a.init_checkpoint is not None:p.error('Cannot mix --init-checkpoint and joint8 role-wise sources')
        if a.phase!='joint8':p.error('Role-wise episode selection is supported for joint8 only')
        if a.resume:p.error('On --resume omit --joint-routing-ep and --joint-dispatch-ep')
        if a.joint_routing_ep<1 or a.joint_dispatch_ep<1:
            p.error('Source snapshot episode must be positive')
    return a

#=============== KẾT THÚC MODULE 05 - ĐỌC THAM SỐ TRAINING TỪ TERMINAL ===============






#=================== MODULE 06 - KIỂM TRA HỆ THỐNG TRƯỚC KHI TRAIN ===================
# Mục đích: Kiểm tra blueprint, simulator ZIP, giới hạn 20 slot AGV và cấu hình PPO.
# Giải thích: Tính fingerprint nguồn để khóa điều kiện thí nghiệm.

def preflight(a):
    blueprint=_pick_blueprint(a.blueprint)
    if not blueprint.exists():raise FileNotFoundError(blueprint)
    cfg=_read(blueprint)
    keys=('max_fleetsize','hidden','model_file','seed','gamma','gae_lambda','clip','lr',
          'epochs','mini_batches','ppo_v7_options')
    missing=[x for x in keys if x not in cfg]
    if missing:raise ValueError('Blueprint missing: '+repr(missing))
    if int(cfg['max_fleetsize'])!=20:
        raise ValueError('Must reuse V8/V14 blueprint with max_fleetsize=20')
    from model_builder import TOBIAS_JOINT_REWARDS
    assert 'reward_part_completed' not in TOBIAS_JOINT_REWARDS
    if a.seed is None:a.seed=int(cfg['seed'])
    if a.max_events is None:a.max_events=int(cfg.get('max_events',100000))
    if a.max_events<1:raise ValueError('max-events must be positive')
    if int(cfg['mini_batches']) != 1:
        raise ValueError('Full-rollout requires blueprint mini_batches=1')
    model=Path(a.model or cfg['model_file']).resolve()
    if not model.is_file():
        raise FileNotFoundError('AnyLogic model .zip missing: '+str(model)+'; use --model PATH')
    fingerprints={
        'architecture':_digest(MODEL_MODULE),
        'environment_builder':_digest(BUILD_MODULE),
        'staged_policy':_digest(STAGE_MODULE),
        'full_rollout_optimizer':_digest(OPTIMIZER_MODULE),
        'training_runner':_digest(Path(__file__).resolve()),
        'ppo_algorithm':_digest(ROOT/'thesis'/'mappo'/'algorithm.py'),
        'ppo_stability':_digest(ROOT/'thesis'/'mappo'/'ppo_stability.py'),
        'event_collector':_digest(ROOT/'thesis'/'mappo'/'collector.py'),
        'matrix_environment':_digest(ROOT/'thesis'/'envs'/'matrix.py'),
        'behaviors':_digest(ROOT/'thesis'/'envs'/'behaviors.py'),
        'alpyne_bridge':_digest(ROOT/'thesis'/'envs'/'alpyne_bridge.py'),
        'reward_config':hashlib.sha256(json.dumps(TOBIAS_JOINT_REWARDS,sort_keys=True).encode()).hexdigest(),
    }
    p=PHASE_ORDER.index(a.phase)
    out=a.output_root.resolve()/a.phase
    return cfg,blueprint,model,fingerprints,out,p

#============== KẾT THÚC MODULE 06 - KIỂM TRA HỆ THỐNG TRƯỚC KHI TRAIN ==============






#================= MODULE 07 - KIỂM TRA CHECKPOINT TRONG STATE.JSON =================
# Mục đích: Xác minh file checkpoint nằm đúng phase và có đúng SHA256.
# Giải thích: Chống resume nhầm hoặc đọc checkpoint bị thay đổi.

def _checkpoint(state,out):
    path=Path(state['checkpoint']).resolve()
    if not path.is_file() or path.parent!=Path(out).resolve():
        raise RuntimeError('Missing checkpoint in own phase directory: '+str(path))
    if _digest(path)!=state.get('checkpoint_sha256'):
        raise RuntimeError('Checkpoint hash mismatch; refusing corrupt resume/transfer')
    return path

#============= KẾT THÚC MODULE 07 - KIỂM TRA CHECKPOINT TRONG STATE.JSON =============






#======================== MODULE 08 - TẠO METADATA THÍ NGHIỆM ========================
# Mục đích: Ghi phase, số xe, code hash, model, seed, reward và thiết bị.
# Giải thích: Dùng khi resume/đánh giá để tái lập điều kiện chạy.

def _meta(a,cfg,blueprint,model,fingerprints,phaseidx):
    return dict(version=VERSION,phase=a.phase,phase_index=phaseidx,
        fleet_size=PHASES[a.phase][0],active_roles=list(PHASES[a.phase][1]),
        sim_seconds=3600,base_seed=a.seed,max_events=a.max_events,device=a.device,
        max_minibatch=a.max_minibatch,cpu_threads=a.cpu_threads,
        snapshot_every=a.snapshot_every,blueprint_sha256=_digest(blueprint),
        blueprint_file=str(blueprint),anylogic_sha256=_digest(model),
        model_file=str(model),fingerprints=fingerprints,
        architecture='7-edge conditioned attention+message GATv2 2x4/32 (no mask) + LinFE/CTDE',
        edge_features=list(EDGE_FEATURE_NAMES),
        edge_attention_conditioned=True,
        edge_message_conditioned=True,
        edge_dimension=len(EDGE_FEATURE_NAMES),
        stationary_occupancy_is_proxy=True,
        traffic_observation_only=True,
        routing_action_mask_applied=False,
        ppo_update_mode='full_rollout_gradient_accumulation',
        ppo_microbatch_size=a.max_minibatch,
        original_tobias_dispatch='random_when_training_routing',
        dispatch_pretraining_router='Tobias native CollisionFreeRouting (time-window reservation)',
        reward_source='Tobias training_disp_rout.ipynb joint configuration',
        reward_part_completed='not overridden (commented in original notebook)',
        ppo_core='unchanged MAPPO + existing V7 stability optimizer',
        phase_train_episode_index_starts_at=0,
        dispatch_mask_mode='tobias_original_union')

#=================== KẾT THÚC MODULE 08 - TẠO METADATA THÍ NGHIỆM ===================






#===================== MODULE 09 - SO SÁNH TÍNH KẾ THỪA CẤU HÌNH =====================
# Mục đích: Đối chiếu fingerprint, blueprint, seed và tham số giữa hai phase.
# Giải thích: Nếu khác, từ chối tiếp tục để tránh trộn hai thí nghiệm.

def _verify_lineage(old,current):
    comparable=('version','blueprint_sha256','anylogic_sha256','fingerprints',
                'model_file','max_events','base_seed','device','max_minibatch',
                'cpu_threads','snapshot_every','sim_seconds')
    diff={k:(old.get(k),current.get(k)) for k in comparable if old.get(k)!=current.get(k)}
    if diff:raise RuntimeError('Cross-stage source/config mismatch: '+str(diff))

#================ KẾT THÚC MODULE 09 - SO SÁNH TÍNH KẾ THỪA CẤU HÌNH ================







#=============== MODULE 10 - BẢO ĐẢM ROLE ĐÓNG BĂNG KHÔNG BỊ THAY ĐỔI ===============
# Mục đích: Băm trạng thái Actor, Critic và ValueNorm của role không train.
# Giải thích: Kiểm tra trước và sau PPO update.

def _frozen_role_hashes(model, trained_roles):
    """Read-only audit: verify pretraining does not change frozen-role weights."""
    allowed=set(trained_roles)
    output={}
    for name,role in model.roles.items():
        if name in allowed: continue
        h=hashlib.sha256()
        for component in ('actor','critic'):
            for key,value in sorted(getattr(role,component).state_dict().items()):
                h.update(component.encode());h.update(key.encode())
                h.update(value.detach().contiguous().cpu().numpy().tobytes())
        h.update(json.dumps(role.valuenorm.state_dict(),sort_keys=True).encode())
        output[name]=h.hexdigest()
    return output

#=========== KẾT THÚC MODULE 10 - BẢO ĐẢM ROLE ĐÓNG BĂNG KHÔNG BỊ THAY ĐỔI ===========






#==================== MODULE 11 - CHỌN CHECKPOINT ROLE CHO JOINT8 ====================
# Mục đích: Đọc hai snapshot routing8/dispatch8 khi có tùy chọn opt-in.
# Giải thích: Kiểm tra source, SHA256 và đăng ký snapshot.

def _joint_snapshot_sources(a, expected, run_root):
    """Validate TWO committed source snapshots before touching joint8 outputs.

    This is opt-in. Default staged inheritance and downstream joint12+ phases
    remain untouched. Snapshots are always read-only.
    """
    if a.joint_routing_ep is None:
        return None
    result={}
    for phase,episode in (('routing8',a.joint_routing_ep),
                          ('dispatch8',a.joint_dispatch_ep)):
        folder=Path(run_root)/phase
        meta=_read(folder/'experiment.json')
        if meta.get('phase')!=phase or int(meta.get('fleet_size',-1))!=8:
            raise RuntimeError('Joint8 source phase/fleet mismatch: '+phase)
        _verify_lineage(meta,expected)
        stage_state=_read(folder/'state.json')
        if int(stage_state.get('last_episode',0))<episode:
            raise RuntimeError(f'{phase} has not completed EP{episode}')
        snapshot=(folder/f'snapshot_ep_{episode:04d}.pt').resolve()
        if not snapshot.is_file():
            raise FileNotFoundError('Missing joint8 source snapshot: '+str(snapshot))
        registered={str(Path(p).resolve()) for p in stage_state.get('snapshots',[])}
        if str(snapshot) not in registered:
            raise RuntimeError('Snapshot is not registered in source state.json: '+str(snapshot))
        result[phase]=dict(episode=episode,checkpoint=str(snapshot),
                           checkpoint_sha256=_digest(snapshot))
    return result

#=============== KẾT THÚC MODULE 11 - CHỌN CHECKPOINT ROLE CHO JOINT8 ===============






#================= MODULE 12 - CHUYỂN TRỌNG SỐ CÁC ROLE SANG JOINT8 =================
# Mục đích: Nạp Actor, Critic, optimizer và ValueNorm từ hai checkpoint.
# Giải thích: Đảm bảo role được khởi tạo đúng theo giai đoạn tiền huấn luyện.

def _load_joint_roles(model, sources):
    """Transfer each role's actor, critic, ValueNorm and Adam state intact.

    routing8 EP200 -> routing; dispatch8 EP100 -> front/rear.
    Fails closed on incompatible checkpoints, without touching source files.
    """
    import torch
    loaded={}
    for phase,entry in sources.items():
        checkpoint=Path(entry['checkpoint'])
        if _digest(checkpoint)!=entry['checkpoint_sha256']:
            raise RuntimeError('Joint8 source checkpoint changed: '+str(checkpoint))
        loaded[phase]=torch.load(str(checkpoint),map_location='cpu',weights_only=True)
        if set(loaded[phase].get('roles',{}))!=set(model.roles):
            raise RuntimeError('Unexpected role set in '+phase+' source')
    if loaded['routing8'].get('config')!=loaded['dispatch8'].get('config'):
        raise RuntimeError('Routing8 and Dispatch8 PPO configs differ')
    mapping={'routing':'routing8','dispatch_front':'dispatch8','dispatch_rear':'dispatch8'}
    for role_name,phase in mapping.items():
        saved=loaded[phase]['roles'][role_name]
        net=model.roles[role_name]
        net.actor.load_state_dict(saved['actor'],strict=True)
        net.critic.load_state_dict(saved['critic'],strict=True)
        net.actor_optimizer.load_state_dict(saved['actor_optimizer'])
        net.critic_optimizer.load_state_dict(saved['critic_optimizer'])
        net.valuenorm.load_state_dict(saved['valuenorm'])
    print('[V15 JOINT8 ROLE TRANSFER] routing from routing8 EP',
          sources['routing8']['episode'],'dispatch front/rear from dispatch8 EP',
          sources['dispatch8']['episode'],flush=True)

#============= KẾT THÚC MODULE 12 - CHUYỂN TRỌNG SỐ CÁC ROLE SANG JOINT8 =============






#=============== MODULE 13 - HÀM ĐIỀU PHỐI TOÀN BỘ QUÁ TRÌNH TRAINING ===============
# Mục đích: Quản lý cấu hình, tạo AnyLogic/MAPPO, thu episode, gọi PPO, lưu checkpoint.
# Giải thích: Chức năng tương tự main() của một chương trình C điều khiển toàn hệ thống.

def run(argv=None):

    #================== PHẦN RUN-01 - ĐỌC THAM SỐ VÀ KIỂM TRA TRƯỚC RUN ==================
    # Chuyển chuỗi lệnh vào biến a, xác thực cấu hình và fingerprint.

    a=parse_args(argv)
    cfg,blueprint,simulator,fingerprints,out,index=preflight(a)
    expected=_meta(a,cfg,blueprint,simulator,fingerprints,index)
    joint_sources=_joint_snapshot_sources(a,expected,out.parent)
    if a.init_checkpoint is not None:
        src=Path(a.init_checkpoint).resolve()
        expected['init_checkpoint']=str(src)
        expected['init_checkpoint_sha256']=_digest(src)

    #=========================== PHẦN RUN-02 - CHẾ ĐỘ DRY RUN ===========================
    # In thông tin, không khởi tạo AnyLogic, không train.

    if a.dry_run:
        print('[TRAIN DRY RUN OK] phase=',a.phase,'fleet=',PHASES[a.phase][0],
              'roles=',PHASES[a.phase][1],'sim=3600 seconds',flush=True)
        print('[TRAIN DRY RUN OK] GAT Dijkstra guidance=OFF; dispatch8 native Tobias CollisionFreeRouting=ON; stop mask=original')
        print('[TRAIN DRY RUN OK] AnyLogic ZIP=',simulator)
        print('[TRAIN DRY RUN OK] blueprint=',blueprint)
        if joint_sources:
            print('[TRAIN DRY RUN OK] joint8 role-wise sources:',joint_sources,flush=True)
        if a.init_checkpoint is not None:
            print('[TRAIN DRY RUN OK] joint16 EP0 source:',expected['init_checkpoint'],
                  'sha256=',expected['init_checkpoint_sha256'],flush=True)
        print('[TRAIN DRY RUN OK] Dispatch mask=Tobias original union',flush=True)
        print('[TRAIN DRY RUN OK] no environment launched and no output modified')
        return

    #======================= PHẦN RUN-03 - RESUME HAY KHỞI TẠO EP0 =======================
    # Resume kiểm tra state/metrics; run mới chuẩn bị checkpoint nguồn.

    if a.resume:
        actual=_read(out/'experiment.json')
        _verify_lineage(actual,expected)
        if actual['phase']!=a.phase or actual['fleet_size']!=PHASES[a.phase][0]:
            raise RuntimeError('Resume stage mismatch')
        state=_read(out/'state.json')
        ck=_checkpoint(state,out)
        if not (out/'metrics.jsonl').exists():raise FileNotFoundError('Missing metrics.jsonl')
        ids=[int(json.loads(s)['episode']) for s in (out/'metrics.jsonl').read_text(encoding='utf8').splitlines() if s.strip()]
        if ids!=list(range(1,int(state['last_episode'])+1)):
            raise RuntimeError('Metrics/committed checkpoint mismatch, inspect before resuming')
    else:
        if out.exists() and any(out.iterdir()):
            raise FileExistsError('Output not empty: '+str(out)+'; use --resume')
        ck=None
        if a.init_checkpoint is not None:
            ck=Path(a.init_checkpoint).resolve()
        elif index > 0 and a.phase != 'dispatch8':
            prev=out.parent/PHASE_ORDER[index-1]
            previous=_read(prev/'experiment.json')
            _verify_lineage(previous,expected)
            state_previous=_read(prev/'state.json')
            if int(state_previous['last_episode'])<1:
                raise RuntimeError('Previous phase must complete at least one episode')
            ck=_checkpoint(state_previous,prev)
        if joint_sources:
            ck=None  # All THREE roles will load from their requested snapshots.
        state=dict(last_episode=0,checkpoint=None,checkpoint_sha256=None,
                   run_seconds=0.0,snapshots=[],source_checkpoint=str(ck) if ck else None)

    #======================= PHẦN RUN-04 - CỐ ĐỊNH SEED / THIẾT BỊ =======================
    # Chuẩn bị PyTorch, NumPy, CUDA và random seed.

    import numpy as np
    import torch
    from model_builder import build_model, TOBIAS_DISPATCH_MASK_MODE
    from thesis.mappo.collector import EventCollector
    from thesis.mappo.routing_graph import GraphSpec
    from thesis.mappo.ppo_stability import audit_rollout_roles
    os.environ['MAPPO_MAX_MINIBATCH']=str(a.max_minibatch)
    os.environ['MAPPO_DISPATCH_MASK_MODE']=TOBIAS_DISPATCH_MASK_MODE
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name]=str(a.cpu_threads)
    torch.set_num_threads(a.cpu_threads)
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; use --device cpu only for offline smoke')
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    logging.getLogger('alpyne').setLevel(logging.ERROR)
    env=None
    try:

        #=================== PHẦN RUN-05 - KHỞI TẠO ANYLOGIC VÀ MẠNG MAPPO ===================
        # Builder cung cấp environment, ba role và đồ thị giao thông.

        env,model,graph,arch=build_model(cfg,model_path=simulator,
            fleet_size=PHASES[a.phase][0],port=a.port,device=a.device,seed=a.seed,
            max_events=a.max_events,sim_seconds=3600,checkpoint=ck,phase=a.phase)
        try:
            arch.update(enable_fullrollout(model, microbatch_size=a.max_minibatch))
        except Exception:
            env.close()
            env = None
            raise
        if a.init_checkpoint is not None:
            from dataclasses import asdict
            loaded_cfg=torch.load(str(ck),map_location='cpu',weights_only=True).get('config')
            if loaded_cfg!=asdict(model.cfg):
                raise RuntimeError('Initial checkpoint MAPPOConfig differs from current configuration')
        if joint_sources:
            _load_joint_roles(model,joint_sources)

        #======================= PHẦN RUN-06 - GHI TRẠNG THÁI KHỞI TẠO =======================
        # Tạo EP0, initial checkpoint, experiment.json, metrics.jsonl.

        if not a.resume:
            out.mkdir(parents=True,exist_ok=True)
            shutil.copy2(blueprint,out/'run_config_blueprint.json')
            initial=out/'initial_ep_0000.pt'
            model.save(str(initial))
            state['checkpoint']=str(initial.resolve())
            state['checkpoint_sha256']=_digest(initial)
            expected['architecture_actual']=arch
            expected['source_checkpoint']=str(ck) if ck else None
            expected['source_checkpoint_sha256']=_digest(ck) if ck else None
            if joint_sources:
                expected['joint_role_sources']=joint_sources
                state['joint_role_sources']=joint_sources
            _atomic_json(out/'experiment.json',expected)
            (out/'metrics.jsonl').touch(exist_ok=False)
            _atomic_json(out/'state.json',state)
            print('[TRAIN PHASE INITIALIZED]',a.phase,'EP0','src=',ck,flush=True)
        if a.init_only:
            print('[TRAIN INIT ONLY] No training conducted');return
        # No guidance, no GNN wrapper, no collector/algorithm source edits.

        #================== PHẦN RUN-07 - KẾT NỐI POLICY VÀ EVENT COLLECTOR ==================
        # Adapter chọn controller theo role; collector thu trajectories.

        adapter=StagePolicyAdapter(model,a.phase,a.seed)
        collector=EventCollector(adapter,env,stop_mask_mode='baseline')
        phase_number=index

        #========================== PHẦN RUN-08 - VÒNG LẶP EPISODE ==========================
        # Lặp lại cho đến khi đạt tổng số EP đã yêu cầu.

        while state['last_episode']<a.max_episodes:
            ep=int(state['last_episode'])+1
            roll_seed=int(a.seed)+phase_number*100000+ep-1
            random.seed(roll_seed);np.random.seed(roll_seed);torch.manual_seed(roll_seed)
            adapter.rng=np.random.default_rng(roll_seed)
            t0=time.monotonic()

            #=================== PHẦN RUN-09 - CHẠY MÔ PHỎNG VÀ THU TRAJECTORY ===================
            # Mỗi episode được chạy hết trước khi PPO update.

            paths,meta=collector.episode(seed=roll_seed,train=True,max_event_steps=a.max_events)
            if meta['end_reason']=='max_event_steps':
                raise RuntimeError('Max-events truncated rollout. No PPO update.')
            if GraphSpec.from_hive(env.agent_hive).fingerprint!=graph.fingerprint:
                raise RuntimeError('Hive graph changed within training')

            #========================= PHẦN RUN-10 - LỌC ROLE CẦN TRAIN =========================
            # Chỉ dùng các transition thuộc role active.

            selected=trained_trajectories(paths,a.phase)
            observed={segment[0]['role'] for segment in selected if segment}
            if not observed.intersection(PHASES[a.phase][1]):
                raise RuntimeError('No trainable transitions collected')

            #========================== PHẦN RUN-11 - KIỂM TRA LOG-PROB ==========================
            # Từ chối update nếu log-prob lúc rollout không khớp mạng.

            parity=audit_rollout_roles(model,selected,max_per_role=48)
            if any(isinstance(x,dict) and x.get('passed') is False for x in parity.values()):
                raise RuntimeError('PPO rollout logprob mismatch; refusing update')
            # Reuse V7 stable PPO, unchanged: only selected role trajectories.
            frozen_before=_frozen_role_hashes(model,PHASES[a.phase][1])

            #=================== PHẦN RUN-12 - CẬP NHẬT ACTOR/CRITIC BẰNG PPO ===================
            # Đây mới là nơi gọi GAE, PPO loss, backward và Adam.

            updates=model.train(selected)
            if _frozen_role_hashes(model,PHASES[a.phase][1])!=frozen_before:
                raise RuntimeError('A frozen role changed during PPO; aborting commit')
            if set(updates)-set(PHASES[a.phase][1]):
                raise RuntimeError('Unexpected frozen role optimizer updated')

            #====================== PHẦN RUN-13 - LƯU CHECKPOINT SAU UPDATE ======================
            # Luân phiên hai slot, tránh ghi đè snapshot cũ.

            slot=out/f'rolling_slot_{ep%2}.pt'
            tmp=out/f'rolling_slot_{ep%2}.pt.tmp'
            model.save(str(tmp));os.replace(tmp,slot)
            elapsed=time.monotonic()-t0

            #======================= PHẦN RUN-14 - LƯU METRICS TỪ EPISODE =======================
            # Ghi JSONL, cập nhật state.json và snapshot định kỳ.

            row=dict(version=VERSION,phase=a.phase,episode=ep,
                     fleet_size=PHASES[a.phase][0],sim_seconds=3600,
                     environment_seed=roll_seed,environment=meta,
                     ppo_audit=parity,policy_updates=updates,architectures=arch,
                     train_roles=sorted(PHASES[a.phase][1]),
                     non_train_actions=('random_dispatch' if a.phase=='routing8' else
                                       'tobias_native_collision_free_routing' if a.phase=='dispatch8' else 'none'),
                     reward='Tobias joint notebook, no shaping',
                     gat_dijkstra_guidance=False,
                     native_conventional_routing=(a.phase=='dispatch8'),
                     dispatch_mask_mode=TOBIAS_DISPATCH_MASK_MODE,
                     elapsed_real_seconds=elapsed)
            with (out/'metrics.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(row,ensure_ascii=False,default=str)+'\n')
                f.flush();os.fsync(f.fileno())
            state['last_episode']=ep
            state['checkpoint']=str(slot.resolve())
            state['checkpoint_sha256']=_digest(slot)
            state['run_seconds']=round(float(state['run_seconds'])+elapsed,2)
            if ep%a.snapshot_every==0:
                snapshot=out/f'snapshot_ep_{ep:04d}.pt'
                partial=out/f'snapshot_ep_{ep:04d}.pt.tmp'
                shutil.copy2(slot,partial);os.replace(partial,snapshot)
                state['snapshots'].append(str(snapshot.resolve()))
            _atomic_json(out/'state.json',state)
            stats=meta.get('statistics',{})
            print(f'[TRAIN {a.phase.upper()} EP{ep}] '
                  f'AGV={PHASES[a.phase][0]} JobDone={stats.get("jobDone",0)} '
                  f'Block={stats.get("block",0)} LiveLock={stats.get("liveLock",0)} '
                  f'Reward={meta.get("delivered_reward",0.):.2f} '
                  f'Updated={sorted(updates)} Elapsed={elapsed:.1f}s',flush=True)
    except KeyboardInterrupt:
        print('[TRAIN INTERRUPTED] Last committed EP',state['last_episode'],flush=True)
    finally:
        if env is not None:env.close()
    print('[TRAIN PHASE DONE]',a.phase,'committed EP',state['last_episode'],'output',out,flush=True)

#=========== KẾT THÚC MODULE 13 - HÀM ĐIỀU PHỐI TOÀN BỘ QUÁ TRÌNH TRAINING ===========






#========================= MODULE 14 - ĐIỂM VÀO TRAINER V15 =========================
# Mục đích: Chỉ gọi run() khi file được chạy trực tiếp.
# Giải thích: V16-B dùng lại hàm này thông qua wrapper, nên log vẫn có nhãn V15.

if __name__=='__main__':run()

#===================== KẾT THÚC MODULE 14 - ĐIỂM VÀO TRAINER V15 =====================




