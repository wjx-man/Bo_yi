from __future__ import annotations

import argparse
from pathlib import Path
import random

from einstein_chess.agents import MCTSAgent, NeuralMCTSAgent, PolicyValueAgent
from einstein_chess.engine import PlayerColor
from einstein_chess.players import HumanPlayer, PlayerAgent, RandomAIPlayer
from einstein_chess.ui import EinsteinChessApp


def main() -> None:
    args = _parse_args_from()
    players = {
        PlayerColor.RED: _build_player(PlayerColor.RED, args),
        PlayerColor.BLUE: _build_player(PlayerColor.BLUE, args),
    }
    app = EinsteinChessApp(players=players)
    app.mainloop()


def _build_player(color: PlayerColor, args: argparse.Namespace) -> PlayerAgent:
    kind = args.red if color is PlayerColor.RED else args.blue
    name = color.label
    rng = random.Random(args.seed + (0 if color is PlayerColor.RED else 1))

    if kind == "human":
        return HumanPlayer(name)
    if kind == "random":
        return RandomAIPlayer(name=f"{name} RandomAI", rng=rng)
    if kind == "mcts":
        return MCTSAgent(
            name=f"{name} MCTS",
            simulations=args.mcts_simulations,
            max_rollout_steps=args.mcts_rollout_steps,
            rng=rng,
        )
    if kind == "policy":
        checkpoint = _checkpoint_for(color, args)
        return PolicyValueAgent(
            checkpoint_path=checkpoint,
            name=f"{name} Policy",
            device=args.device,
        )
    if kind == "neural-mcts":
        checkpoint = _checkpoint_for(color, args)
        return NeuralMCTSAgent(
            checkpoint_path=checkpoint,
            name=f"{name} NeuralMCTS",
            simulations=args.neural_mcts_simulations,
            c_puct=args.c_puct,
            device=args.device,
            rng=rng,
        )
    raise ValueError(f"Unsupported player kind: {kind}")


def _checkpoint_for(color: PlayerColor, args: argparse.Namespace) -> Path:
    checkpoint = args.red_checkpoint if color is PlayerColor.RED else args.blue_checkpoint
    if checkpoint is None:
        checkpoint = args.checkpoint
    if checkpoint is None:
        raise ValueError(
            f"{color.label} uses a neural agent, so --checkpoint or side-specific checkpoint is required."
        )
    return checkpoint


def _parse_args_from(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch 爱恩斯坦棋 GUI.")
    player_choices = ("human", "random", "mcts", "policy", "neural-mcts")
    parser.add_argument("--red", choices=player_choices, default="human")
    parser.add_argument("--blue", choices=player_choices, default="human")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--red-checkpoint", type=Path, default=None)
    parser.add_argument("--blue-checkpoint", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--mcts-simulations", type=int, default=100)
    parser.add_argument("--mcts-rollout-steps", type=int, default=120)
    parser.add_argument("--neural-mcts-simulations", type=int, default=80)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
