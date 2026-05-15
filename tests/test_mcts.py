from __future__ import annotations

import random
import unittest

from einstein_chess.agents import MCTSAgent
from einstein_chess.engine import EinsteinGame, Move, Piece, PlayerColor


class MCTSAgentTests(unittest.TestCase):
    def test_mcts_returns_legal_move_and_visit_policy(self) -> None:
        game = EinsteinGame(rng=random.Random(7))
        agent = MCTSAgent(simulations=12, rng=random.Random(11))
        legal_moves = game.get_legal_moves()

        move = agent.choose_move(game.snapshot(), legal_moves)

        self.assertIn(move, legal_moves)
        self.assertEqual(sum(agent.last_visit_counts.values()), 12)
        self.assertAlmostEqual(sum(agent.last_policy.values()), 1.0)
        self.assertEqual(set(agent.last_visit_counts), set(legal_moves))

    def test_mcts_takes_immediate_goal_win(self) -> None:
        game = EinsteinGame(rng=random.Random(7))
        game.board = [[None for _ in range(5)] for _ in range(5)]
        game.board[3][3] = Piece(PlayerColor.RED, 6)
        game.board[0][0] = Piece(PlayerColor.BLUE, 1)
        game.current_player = PlayerColor.RED
        game.dice_roll = 6
        game.winner = None
        expected = Move(PlayerColor.RED, 6, (3, 3), (4, 4))

        agent = MCTSAgent(simulations=4, rng=random.Random(11))
        move = agent.choose_move(game.snapshot(), game.get_legal_moves())

        self.assertEqual(move, expected)
        self.assertEqual(agent.last_visit_counts[expected], 1)
        self.assertEqual(agent.last_policy[expected], 1.0)

    def test_mcts_can_run_inside_match_runner(self) -> None:
        from einstein_chess.match import MatchRunner
        from einstein_chess.players import RandomAIPlayer

        players = {
            PlayerColor.RED: MCTSAgent(simulations=8, rng=random.Random(1)),
            PlayerColor.BLUE: RandomAIPlayer(rng=random.Random(2)),
        }
        result = MatchRunner(
            players,
            rng=random.Random(3),
            charge_layout_time=False,
            max_turns=80,
        ).play()

        self.assertIn(
            result.reason,
            {"goal", "capture_all", "max_turns_exceeded"},
        )
        self.assertGreater(len(result.steps), 0)


if __name__ == "__main__":
    unittest.main()
