from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from einstein_chess.agents import (
    FullNeuralMCTSAgent,
    MCTSAgent,
    NeuralMCTSAgent,
    PolicyValueAgent,
)
from einstein_chess.engine import PlayerColor
from einstein_chess.match import MatchRunner
from einstein_chess.players import PlayerAgent, RandomAIPlayer


def main() -> None:
    args = _parse_args()
    red_wins = 0
    blue_wins = 0
    draws = 0
    reasons: dict[str, int] = {}
    total_steps = 0
    game_rows: list[dict[str, int | str | float]] = []

    for game_index in range(args.games):
        rng = random.Random(args.seed + game_index)
        players = {
            PlayerColor.RED: _build_agent(
                kind=args.red,
                color=PlayerColor.RED,
                args=args,
                rng=random.Random(rng.randrange(2**31)),
            ),
            PlayerColor.BLUE: _build_agent(
                kind=args.blue,
                color=PlayerColor.BLUE,
                args=args,
                rng=random.Random(rng.randrange(2**31)),
            ),
        }
        result = MatchRunner(
            players,
            rng=random.Random(rng.randrange(2**31)),
            charge_layout_time=False,
            layout_mode="random" if args.layout == "random" else "agent",
            max_turns=args.max_turns,
        ).play()
        step_count = len(result.steps)
        total_steps += step_count
        reasons[result.reason] = reasons.get(result.reason, 0) + 1
        if result.winner is PlayerColor.RED:
            red_wins += 1
        elif result.winner is PlayerColor.BLUE:
            blue_wins += 1
        else:
            draws += 1
        game_rows.append(
            {
                "game_index": game_index,
                "red": args.red,
                "blue": args.blue,
                "layout": args.layout,
                "winner": "none" if result.winner is None else result.winner.value,
                "reason": result.reason,
                "steps": step_count,
                "red_seconds_left": result.red_seconds_left,
                "blue_seconds_left": result.blue_seconds_left,
            }
        )

    total_games = max(args.games, 1)
    summary = {
        "games": args.games,
        "red": args.red,
        "blue": args.blue,
        "layout": args.layout,
        "red_wins": red_wins,
        "blue_wins": blue_wins,
        "draws_or_unfinished": draws,
        "red_win_rate": red_wins / total_games,
        "blue_win_rate": blue_wins / total_games,
        "average_steps": total_steps / total_games,
        "reasons": reasons,
    }
    print(f"games: {args.games}")
    print(f"red: {args.red}")
    print(f"blue: {args.blue}")
    print(f"layout: {args.layout}")
    print(f"red_wins: {red_wins}")
    print(f"blue_wins: {blue_wins}")
    print(f"draws_or_unfinished: {draws}")
    print(f"red_win_rate: {red_wins / total_games:.4f}")
    print(f"blue_win_rate: {blue_wins / total_games:.4f}")
    print(f"average_steps: {total_steps / total_games:.2f}")
    print(f"reasons: {reasons}")
    if args.output is not None:
        _write_outputs(args.output, summary, game_rows)
        print(f"saved_eval_summary: {args.output}")
        print(f"saved_eval_games: {_games_output_path(args.output)}")


def _build_agent(
    kind: str,
    color: PlayerColor,
    args: argparse.Namespace,
    rng: random.Random,
) -> PlayerAgent:
    if kind == "random":
        return RandomAIPlayer(name=f"{color.value}_random", rng=rng)
    if kind == "mcts":
        return MCTSAgent(
            name=f"{color.value}_mcts",
            simulations=args.mcts_simulations,
            max_rollout_steps=args.mcts_rollout_steps,
            rng=rng,
        )
    if kind == "policy":
        checkpoint = args.red_checkpoint if color is PlayerColor.RED else args.blue_checkpoint
        if checkpoint is None:
            checkpoint = args.checkpoint
        if checkpoint is None:
            raise ValueError("PolicyValueAgent requires --checkpoint or side-specific checkpoint.")
        return PolicyValueAgent(
            checkpoint_path=checkpoint,
            name=f"{color.value}_policy",
            device=args.device,
        )
    if kind == "neural-mcts":
        checkpoint = args.red_checkpoint if color is PlayerColor.RED else args.blue_checkpoint
        if checkpoint is None:
            checkpoint = args.checkpoint
        if checkpoint is None:
            raise ValueError("NeuralMCTSAgent requires --checkpoint or side-specific checkpoint.")
        return NeuralMCTSAgent(
            checkpoint_path=checkpoint,
            name=f"{color.value}_neural_mcts",
            simulations=args.neural_mcts_simulations,
            c_puct=args.c_puct,
            device=args.device,
            rng=rng,
        )
    if kind == "full-neural-mcts":
        checkpoint = args.red_checkpoint if color is PlayerColor.RED else args.blue_checkpoint
        if checkpoint is None:
            checkpoint = args.checkpoint
        if checkpoint is None:
            raise ValueError("FullNeuralMCTSAgent requires --checkpoint or side-specific checkpoint.")
        return FullNeuralMCTSAgent(
            checkpoint_path=checkpoint,
            name=f"{color.value}_full_neural_mcts",
            simulations=args.full_neural_mcts_simulations,
            c_puct=args.c_puct,
            max_depth=args.full_neural_mcts_depth,
            chance_mode=args.chance_mode,
            device=args.device,
            rng=rng,
        )
    raise ValueError(f"Unknown agent kind: {kind}")


def _write_outputs(
    output_path: Path,
    summary: dict[str, object],
    game_rows: list[dict[str, int | str | float]],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    games_output_path = _games_output_path(output_path)
    with games_output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(game_rows[0].keys()))
        writer.writeheader()
        writer.writerows(game_rows)


def _games_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_games{output_path.suffix}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate 爱恩斯坦棋 agents by running complete matches."
    )
    parser.add_argument("--games", type=int, default=50)
    agent_choices = ("random", "mcts", "policy", "neural-mcts", "full-neural-mcts")
    parser.add_argument("--red", choices=agent_choices, default="policy")
    parser.add_argument("--blue", choices=agent_choices, default="random")
    parser.add_argument(
        "--layout",
        choices=("default", "random"),
        default="default",
        help="Opening layout mode. default lets each agent choose its layout; random shuffles both start zones.",
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--red-checkpoint", type=Path, default=None)
    parser.add_argument("--blue-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mcts-simulations", type=int, default=50)
    parser.add_argument("--mcts-rollout-steps", type=int, default=120)
    parser.add_argument("--neural-mcts-simulations", type=int, default=80)
    parser.add_argument("--full-neural-mcts-simulations", type=int, default=80)
    parser.add_argument("--full-neural-mcts-depth", type=int, default=12)
    parser.add_argument("--chance-mode", choices=("sample", "enumerate"), default="sample")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
