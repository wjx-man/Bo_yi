from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from einstein_chess.training import generate_self_play_dataset


DEFAULT_OUTPUT = Path("artifacts") / "data" / "self_play.npz"


def main() -> None:
    args = _parse_args()
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    dataset = generate_self_play_dataset(
        num_games=args.games,
        agent_kind=args.agent,
        simulations=args.simulations,
        exploration=args.exploration,
        c_puct=args.c_puct,
        max_rollout_steps=args.max_rollout_steps,
        max_turns=args.max_turns,
        checkpoint_path=args.checkpoint,
        device=args.device,
        layout=args.layout,
        rng=random.Random(args.seed),
        output_path=output_path,
    )
    elapsed = time.perf_counter() - start

    print(f"saved: {output_path}")
    print(f"agent: {args.agent}")
    print(f"layout: {args.layout}")
    print(f"games: {args.games}")
    print(f"samples: {dataset.states.shape[0]}")
    print(f"states: {dataset.states.shape}")
    print(f"policies: {dataset.policies.shape}")
    print(f"values: {dataset.values.shape}")
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"samples_per_second: {dataset.states.shape[0] / max(elapsed, 1e-9):.2f}")
    print(f"value_distribution: {_format_counts(dataset.values)}")
    print(f"winner_distribution: {_format_counts(dataset.winners)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 爱恩斯坦棋 MCTS self-play data as an NPZ dataset."
    )
    parser.add_argument(
        "--agent",
        choices=("mcts", "neural-mcts"),
        default="mcts",
        help="Self-play agent type.",
    )
    parser.add_argument(
        "--layout",
        choices=("default", "random"),
        default="default",
        help="Opening layout mode for each self-play game.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=10,
        help="Number of self-play games to generate.",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="MCTS simulations per move.",
    )
    parser.add_argument(
        "--exploration",
        type=float,
        default=1.4,
        help="UCT exploration coefficient.",
    )
    parser.add_argument(
        "--c-puct",
        type=float,
        default=1.5,
        help="PUCT exploration coefficient for neural-mcts.",
    )
    parser.add_argument(
        "--max-rollout-steps",
        type=int,
        default=160,
        help="Maximum rollout steps per MCTS simulation.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=300,
        help="Maximum turns per self-play game.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible data generation.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Policy-value checkpoint for neural-mcts self-play.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for neural-mcts inference.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output NPZ path.",
    )
    return parser.parse_args()


def _format_counts(values: np.ndarray) -> dict[float, int]:
    unique_values, counts = np.unique(values, return_counts=True)
    return {
        float(value): int(count)
        for value, count in zip(unique_values, counts)
    }


if __name__ == "__main__":
    main()
