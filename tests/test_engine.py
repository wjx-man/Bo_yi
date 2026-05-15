from __future__ import annotations

import random
import unittest

from einstein_chess.engine import EinsteinGame, Move, Piece, PlayerColor


class EinsteinGameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.game = EinsteinGame(rng=random.Random(7))

    def _clear_board(self) -> None:
        self.game.board = [[None for _ in range(5)] for _ in range(5)]
        self.game.move_history = []
        self.game.winner = None

    def test_default_board_encoding_uses_signed_numbers(self) -> None:
        board = self.game.encode_board()
        self.assertEqual(board[0][0], 1)
        self.assertEqual(board[4][4], -1)
        self.assertEqual(board[2][2], 0)

    def test_exact_roll_prefers_same_number_only(self) -> None:
        candidates = self.game.get_candidate_numbers(PlayerColor.RED, 3)
        self.assertEqual(candidates, [3])

    def test_missing_roll_uses_nearest_lower_and_higher_numbers(self) -> None:
        self.game.board[0][1] = None
        self.game.board[0][2] = None
        candidates = self.game.get_candidate_numbers(PlayerColor.RED, 2)
        self.assertEqual(candidates, [1, 4])

    def test_pass_turn_when_current_roll_has_no_legal_moves(self) -> None:
        self._clear_board()
        self.game.board[4][4] = Piece(PlayerColor.RED, 1)
        self.game.board[0][0] = Piece(PlayerColor.BLUE, 1)
        self.game.current_player = PlayerColor.RED
        self.game.dice_roll = 1
        self.assertEqual(self.game.get_legal_moves(), [])
        self.game.pass_turn()
        self.assertEqual(self.game.current_player, PlayerColor.BLUE)

    def test_can_capture_own_piece(self) -> None:
        self._clear_board()
        self.game.board[0][0] = Piece(PlayerColor.RED, 1)
        self.game.board[0][1] = Piece(PlayerColor.RED, 2)
        self.game.board[4][4] = Piece(PlayerColor.BLUE, 1)
        self.game.current_player = PlayerColor.RED
        self.game.dice_roll = 1

        move = Move(
            color=PlayerColor.RED,
            piece_number=1,
            from_position=(0, 0),
            to_position=(0, 1),
        )

        self.assertIn(move, self.game.get_legal_moves())
        self.game.apply_move(move)
        self.assertEqual(self.game.get_piece_numbers(PlayerColor.RED), [1])

    def test_apply_move_updates_board_and_turn(self) -> None:
        self.game.current_player = PlayerColor.RED
        self.game.dice_roll = 3
        move = Move(
            color=PlayerColor.RED,
            piece_number=3,
            from_position=(0, 2),
            to_position=(1, 2),
        )
        self.game.apply_move(move)
        self.assertIsNone(self.game.get_piece((0, 2)))
        piece = self.game.get_piece((1, 2))
        self.assertIsNotNone(piece)
        self.assertEqual(piece.number, 3)
        self.assertEqual(self.game.current_player, PlayerColor.BLUE)

    def test_reaching_goal_corner_wins_game(self) -> None:
        self._clear_board()
        self.game.board[3][3] = Piece(PlayerColor.RED, 6)
        self.game.board[0][0] = Piece(PlayerColor.BLUE, 1)
        self.game.current_player = PlayerColor.RED
        self.game.dice_roll = 6

        move = Move(
            color=PlayerColor.RED,
            piece_number=6,
            from_position=(3, 3),
            to_position=(4, 4),
        )
        winner = self.game.apply_move(move)

        self.assertEqual(winner, PlayerColor.RED)
        self.assertEqual(self.game.winner, PlayerColor.RED)

    def test_capturing_last_enemy_piece_wins_game(self) -> None:
        self._clear_board()
        self.game.board[3][2] = Piece(PlayerColor.RED, 4)
        self.game.board[4][3] = Piece(PlayerColor.BLUE, 2)
        self.game.current_player = PlayerColor.RED
        self.game.dice_roll = 4

        move = Move(
            color=PlayerColor.RED,
            piece_number=4,
            from_position=(3, 2),
            to_position=(4, 3),
        )
        winner = self.game.apply_move(move)

        self.assertEqual(winner, PlayerColor.RED)
        self.assertEqual(self.game.winner, PlayerColor.RED)


if __name__ == "__main__":
    unittest.main()
