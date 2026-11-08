V12 FULL NO DIJKSTRA - FIX PPO PARITY AUDIT (01-11-2026)

Loi cu: RuntimeError: Routing sample missing Dijkstra guidance in PPO buffer
Nguyen nhan: runner V12 goi gnn_audit_uoe_v2.verify_ppo_rollout_parity cua V11,
trong khi V12 da bo routing_guide_bias. Day la loi audit, truoc model.train.

Patch: 2 file Python + 1 file test; KHONG sua V8/V11, KHONG them lai Dijkstra.
- scripts/train_v12_full_no_dijkstra.py: goi verify_v12_ppo_rollout_parity
- scripts/v12_clean_audit.py: audit doc lap V12, check logp, mask, bias vang mat
- tests/test_v12_no_dijkstra_parity.py: 6 offline tests

Sau khi giai nen patch vao C:\MARL_AnyLogic\MAPPO_NEW\thesis:

cd C:\MARL_AnyLogic\MAPPO_NEW\thesis
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p "test_v12_no_dijkstra_parity.py" -v
.\.venv311\Scripts\python.exe -m unittest discover -s tests -p "test_v12_full_no_dijkstra_offline.py" -v

Neu V12 da tao thu muc run va state.json o EP529 thi PHAI --resume:
.\.venv311\Scripts\python.exe -u .\scripts\train_v12_full_no_dijkstra.py --device cuda:0 --max-minibatch 64 --max-episodes 1000 --resume

Khong xoa state.json, checkpoint hay metrics. Khong chay lai lan dau neu da co run.
PPO parity safety van duoc kiem tra; V12 se tu choi buffer neu co routing_guide_bias.

Neu console hien [V12 EP530] ... sau PPO thi EP530 da duoc cap nhat va commit.
