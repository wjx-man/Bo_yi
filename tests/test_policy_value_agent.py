from __future__ import annotations

from pathlib import Path
import random
import unittest

import torch

from einstein_chess.agents import PolicyValueAgent
from einstein_chess.engine import EinsteinGame, PlayerColor
from einstein_chess.match import MatchRunner
from einstein_chess.players import RandomAIPlayer
from einstein_chess.training import PolicyValueNet


class PolicyValueAgentTests(unittest.TestCase):
    def test_policy_value_agent_returns_legal_move(self) -> None:
        checkpoint = Path("tests") / "_tmp_policy_agent.pt"
        try:
            self._save_checkpoint(checkpoint)
            game = EinsteinGame(rng=random.Random(7))
            agent = PolicyValueAgent(checkpoint)
            legal_moves = game.get_legal_moves()

            move = agent.choose_move(game.snapshot(), legal_moves)

            self.assertIn(move, legal_moves)
            self.assertAlmostEqual(float(agent.last_policy.sum()), 1.0, places=5)
            self.assertIsInstance(agent.last_value, float)
        finally:
            if checkpoint.exists():
                checkpoint.unlink()

    def test_policy_value_agent_loads_residual_checkpoint(self) -> None:
        checkpoint = Path("tests") / "_tmp_policy_agent_residual.pt"
        try:
            self._save_checkpoint(checkpoint, residual_blocks=2)
            game = EinsteinGame(rng=random.Random(7))
            agent = PolicyValueAgent(checkpoint)
            legal_moves = game.get_legal_moves()

            move = agent.choose_move(game.snapshot(), legal_moves)

            self.assertIn(move, legal_moves)
        finally:
            if checkpoint.exists():
                checkpoint.unlink()

    def test_policy_value_agent_runs_in_match_runner(self) -> None:
        checkpoint = Path("tests") / "_tmp_policy_match.pt"
        try:
            self._save_checkpoint(checkpoint)
            players = {
                PlayerColor.RED: PolicyValueAgent(checkpoint),
                PlayerColor.BLUE: RandomAIPlayer(rng=random.Random(3)),
            }

            result = MatchRunner(
                players,
                rng=random.Random(5),
                charge_layout_time=False,
                max_turns=80,
            ).play()

            self.assertGreater(len(result.steps), 0)
            self.assertIn(result.reason, {"goal", "capture_all", "max_turns_exceeded"})
        finally:
            if checkpoint.exists():
                checkpoint.unlink()

    def _save_checkpoint(self, path: Path, residual_blocks: int = 0) -> None:
        model = PolicyValueNet(hidden_channels=8, residual_blocks=residual_blocks)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": {
                    "hidden_channels": 8,
                    "residual_blocks": residual_blocks,
                },
            },
            path,
        )


if __name__ == "__main__":
    unittest.main()
