from __future__ import annotations

from pathlib import Path
import random
import unittest

import torch

from einstein_chess.agents import NeuralMCTSAgent
from einstein_chess.engine import EinsteinGame, PlayerColor
from einstein_chess.match import MatchRunner
from einstein_chess.players import RandomAIPlayer
from einstein_chess.training import PolicyValueNet


class NeuralMCTSAgentTests(unittest.TestCase):
    def test_neural_mcts_agent_returns_legal_move_and_policy(self) -> None:
        checkpoint = Path("tests") / "_tmp_neural_mcts.pt"
        try:
            self._save_checkpoint(checkpoint)
            game = EinsteinGame(rng=random.Random(7))
            agent = NeuralMCTSAgent(
                checkpoint_path=checkpoint,
                simulations=6,
                rng=random.Random(11),
            )
            legal_moves = game.get_legal_moves()

            move = agent.choose_move(game.snapshot(), legal_moves)

            self.assertIn(move, legal_moves)
            self.assertEqual(sum(agent.last_visit_counts.values()), 6)
            self.assertAlmostEqual(sum(agent.last_policy.values()), 1.0)
            self.assertAlmostEqual(sum(agent.last_priors.values()), 1.0, places=5)
        finally:
            if checkpoint.exists():
                checkpoint.unlink()

    def test_neural_mcts_agent_runs_in_match_runner(self) -> None:
        checkpoint = Path("tests") / "_tmp_neural_mcts_match.pt"
        try:
            self._save_checkpoint(checkpoint)
            players = {
                PlayerColor.RED: NeuralMCTSAgent(
                    checkpoint_path=checkpoint,
                    simulations=4,
                    rng=random.Random(1),
                ),
                PlayerColor.BLUE: RandomAIPlayer(rng=random.Random(2)),
            }

            result = MatchRunner(
                players,
                rng=random.Random(3),
                charge_layout_time=False,
                max_turns=80,
            ).play()

            self.assertGreater(len(result.steps), 0)
            self.assertIn(result.reason, {"goal", "capture_all", "max_turns_exceeded"})
        finally:
            if checkpoint.exists():
                checkpoint.unlink()

    def _save_checkpoint(self, path: Path) -> None:
        model = PolicyValueNet(hidden_channels=8)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {"hidden_channels": 8},
            },
            path,
        )


if __name__ == "__main__":
    unittest.main()
