from __future__ import annotations

from pathlib import Path
import unittest

import torch

import main
from einstein_chess.agents import MCTSAgent, NeuralMCTSAgent, PolicyValueAgent
from einstein_chess.engine import PlayerColor
from einstein_chess.players import HumanPlayer, RandomAIPlayer
from einstein_chess.training import PolicyValueNet


class MainCliTests(unittest.TestCase):
    def test_builds_basic_players(self) -> None:
        args = main._parse_args_from([])
        self.assertIsInstance(main._build_player(PlayerColor.RED, args), HumanPlayer)

        args = main._parse_args_from(["--blue", "random"])
        self.assertIsInstance(main._build_player(PlayerColor.BLUE, args), RandomAIPlayer)

        args = main._parse_args_from(["--red", "mcts"])
        self.assertIsInstance(main._build_player(PlayerColor.RED, args), MCTSAgent)

    def test_builds_neural_players_with_checkpoint(self) -> None:
        checkpoint = Path("tests") / "_tmp_main_policy.pt"
        try:
            model = PolicyValueNet(hidden_channels=8)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"hidden_channels": 8},
                },
                checkpoint,
            )

            args = main._parse_args_from(
                ["--blue", "policy", "--checkpoint", str(checkpoint)]
            )
            self.assertIsInstance(main._build_player(PlayerColor.BLUE, args), PolicyValueAgent)

            args = main._parse_args_from(
                [
                    "--blue",
                    "neural-mcts",
                    "--checkpoint",
                    str(checkpoint),
                    "--neural-mcts-simulations",
                    "2",
                ]
            )
            self.assertIsInstance(main._build_player(PlayerColor.BLUE, args), NeuralMCTSAgent)
        finally:
            if checkpoint.exists():
                checkpoint.unlink()


if __name__ == "__main__":
    unittest.main()
