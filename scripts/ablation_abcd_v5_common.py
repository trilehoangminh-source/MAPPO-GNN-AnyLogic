"""Shared definitions for controlled 2x2 Routing ablation. No torch/AnyLogic imports."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Arm:
    code: str
    encoder: str
    guide: float
    train_script: str
    eval_script: str


ARMS = (
    Arm('A', 'gat', 1.5, 'train_B3_GNN_UOE_v2.py', 'evaluate_B3_GNN_UOE_v2.py'),
    Arm('B', 'gat', 0.5, 'train_B3_GNN_UOE_v2.py', 'evaluate_B3_GNN_UOE_v2.py'),
    Arm('C', 'edge_gat', 1.5, 'train_B3_edgeGAT_MAPPO_v4.py', 'evaluate_B3_edgeGAT_MAPPO_v4.py'),
    Arm('D', 'edge_gat', 0.5, 'train_B3_edgeGAT_MAPPO_v4.py', 'evaluate_B3_edgeGAT_MAPPO_v4.py'),
)

METRICS = ('jobDone', 'targetReached', 'wrongTarget', 'block', 'liveLock', 'delivered_reward')


def run_dir(root: Path, arm: Arm, seed: int) -> Path:
    return root / f'{arm.code}_{arm.encoder}_guide{str(arm.guide).replace(".", "p")}' / f'train_seed{seed}'


def assert_distinct_seeds(train_seeds: list[int], eval_seeds: list[int], episodes: int) -> None:
    if episodes <= 0 or not train_seeds or not eval_seeds:
        raise ValueError('Training episodes and seed lists must not be empty')
    if len(train_seeds) != len(set(train_seeds)) or len(eval_seeds) != len(set(eval_seeds)):
        raise ValueError('Seed lists must contain distinct values')
    used = {seed + i for seed in train_seeds for i in range(episodes)}
    overlapping = sorted(used.intersection(eval_seeds))
    if overlapping:
        raise ValueError(f'Evaluation seeds overlap with training episodes: {overlapping}')


def train_command(py: str, root: Path, model: Path, arm: Arm, seed: int,
                  episodes: int, sim_seconds: int, max_events: int, port: int,
                  hidden: int, layers: int, heads: int, output: Path, *, epochs: int = 5) -> list[str]:
    cmd = [py, '-u', str(root / 'scripts' / arm.train_script),
           '--model', str(model), '--port', str(port), '--episodes', str(episodes),
           '--sim-seconds', str(sim_seconds), '--max-events', str(max_events),
           '--seed', str(seed), '--output', str(output),
           '--device', 'cpu', '--hidden', str(hidden), '--graph-layers', str(layers),
           '--routing-guide-strength', str(arm.guide),
           '--gamma', '0.9', '--gae-lambda', '0.95', '--clip', '0.1',
           '--lr', '0.0001', '--epochs', str(epochs), '--mini-batches', '1',
           '--diagnostic-samples', '1024', '--parity-samples', '64',
           '--min-candidate-map-rate', '0.99']
    if arm.encoder == 'gat':
        cmd += ['--routing-encoder', 'gat', '--gat-heads', str(heads)]
    else:
        cmd += ['--edge-heads', str(heads), '--guide-final', str(arm.guide),
                '--guide-decay-episodes', '2']
    return cmd


def eval_command(py: str, project_root: Path, arm: Arm, ckpt: Path,
                 seeds: list[int], seconds: int, port: int, out: Path) -> list[str]:
    return [py, '-u', str(project_root / 'scripts' / arm.eval_script),
            '--checkpoint', str(ckpt), '--seeds', *[str(s) for s in seeds],
            '--sim-seconds', str(seconds), '--port', str(port), '--output', str(out)]
