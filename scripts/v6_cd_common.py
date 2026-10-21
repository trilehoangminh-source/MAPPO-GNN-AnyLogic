"""V6 C/D comparison; standard-library only; Edge-GAT Routing, MLP Dispatching."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

METRICS = ('jobDone', 'targetReached', 'wrongTarget', 'block', 'liveLock', 'delivered_reward')


@dataclass(frozen=True)
class Condition:
    code: str
    guide: float


CONDITIONS = (Condition('C', 1.5), Condition('D', 0.5))


def seed_validation(train: list[int], evaluation: list[int], episodes: int) -> None:
    if episodes < 1 or not train or not evaluation:
        raise ValueError('Positive episodes and nonempty train/eval seeds required')
    if len(train) != len(set(train)) or len(evaluation) != len(set(evaluation)):
        raise ValueError('Duplicate train/eval seeds are not allowed')
    # Each train invocation calls AnyLogic with environment_seed = train_seed + episode index.
    used_train_episode_seeds = {s + i for s in train for i in range(episodes)}
    overlap = sorted(used_train_episode_seeds.intersection(evaluation))
    if overlap:
        raise ValueError(f'Evaluation seeds leak into training: {overlap}')


def condition_dir(root: Path, condition: Condition, seed: int) -> Path:
    return root / f'{condition.code}_edge_gat_guide{str(condition.guide).replace(".","p")}' / f'train_seed{seed}'


def training_cmd(py: str, project: Path, model: Path, condition: Condition,
                 seed: int, episode_count: int, train_seconds: int, max_events: int,
                 port: int, hidden: int, layers: int, heads: int, epochs: int,
                 output: Path) -> list[str]:
    g = str(condition.guide)
    return [py, '-u', str(project / 'scripts' / 'train_B3_edgeGAT_MAPPO_v4.py'),
            '--model', str(model), '--port', str(port),
            '--episodes', str(episode_count), '--sim-seconds', str(train_seconds),
            '--max-events', str(max_events), '--seed', str(seed), '--output', str(output),
            '--device', 'cpu', '--hidden', str(hidden), '--graph-layers', str(layers),
            '--routing-guide-strength', g, '--edge-heads', str(heads),
            '--guide-final', g, '--guide-decay-episodes', '2',
            '--gamma', '0.9', '--gae-lambda', '0.95', '--clip', '0.1',
            '--lr', '0.0001', '--epochs', str(epochs), '--mini-batches', '1',
            '--diagnostic-samples', '1024', '--parity-samples', '64',
            '--min-candidate-map-rate', '0.99']


def evaluation_cmd(py: str, project: Path, checkpoint: Path, eval_seeds: list[int],
                   eval_seconds: int, port: int, outfile: Path) -> list[str]:
    return [py, '-u', str(project / 'scripts' / 'evaluate_B3_edgeGAT_MAPPO_v4.py'),
            '--checkpoint', str(checkpoint), '--seeds', *map(str, eval_seeds),
            '--sim-seconds', str(eval_seconds), '--port', str(port), '--output', str(outfile)]
