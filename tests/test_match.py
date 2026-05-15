from __future__ import annotations

import random
import unittest
from typing import Sequence

from einstein_chess.engine import GameSnapshot, Move, PlayerColor, Position
from einstein_chess.match import MatchRunner
from einstein_chess.players import PlayerAgent


class SequenceClock:
    def __init__(self, values: Sequence[float]) -> None:
        self.values = list(values)
        self.index = 0

    def __call__(self) -> float:
        if self.index >= len(self.values):
            return self.values[-1]
        value = self.values[self.index]
        self.index += 1
        return value


class FirstLegalAgent(PlayerAgent):
    mode = "first_legal"

    def choose_move(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move:
        del snapshot
        return legal_moves[0]


class IllegalAgent(PlayerAgent):
    mode = "illegal"

    def choose_move(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move:
        del snapshot, legal_moves
        return Move(PlayerColor.RED, 6, (4, 4), (4, 3))


class BadLayoutAgent(FirstLegalAgent):
    def choose_layout(
        self,
        color: PlayerColor,
        starting_positions: Sequence[Position],
        rng: random.Random,
    ) -> dict[int, Position]:
        del color, rng
        return {number: starting_positions[0] for number in range(1, 7)}


class MatchRunnerTests(unittest.TestCase):
    def _players(self, red: PlayerAgent, blue: PlayerAgent) -> dict[PlayerColor, PlayerAgent]:
        return {PlayerColor.RED: red, PlayerColor.BLUE: blue}

    def test_valid_move_is_timed_and_logged(self) -> None:
        runner = MatchRunner(
            self._players(FirstLegalAgent(), FirstLegalAgent()),
            rng=random.Random(7),
            time_provider=SequenceClock([0.0, 0.0, 0.0, 0.0, 0.0, 2.0]),
            charge_layout_time=False,
            max_turns=1,
        )

        result = runner.play()

        self.assertEqual(result.reason, "max_turns_exceeded")
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].elapsed_seconds, 2.0)
        self.assertEqual(result.red_seconds_left, 898.0)

    def test_timeout_loses_the_game(self) -> None:
        runner = MatchRunner(
            self._players(FirstLegalAgent(), FirstLegalAgent()),
            rng=random.Random(7),
            time_provider=SequenceClock([0.0, 0.0, 0.0, 0.0, 0.0, 901.0]),
            charge_layout_time=False,
        )

        result = runner.play()

        self.assertEqual(result.winner, PlayerColor.BLUE)
        self.assertEqual(result.loser, PlayerColor.RED)
        self.assertEqual(result.reason, "timeout")

    def test_illegal_move_loses_the_game(self) -> None:
        runner = MatchRunner(
            self._players(IllegalAgent(), FirstLegalAgent()),
            rng=random.Random(7),
            time_provider=SequenceClock([0.0, 0.0, 0.0, 0.0, 0.0, 0.1]),
            charge_layout_time=False,
        )

        result = runner.play()

        self.assertEqual(result.winner, PlayerColor.BLUE)
        self.assertEqual(result.loser, PlayerColor.RED)
        self.assertEqual(result.reason, "illegal_move")

    def test_bad_layout_loses_before_gameplay(self) -> None:
        runner = MatchRunner(
            self._players(BadLayoutAgent(), FirstLegalAgent()),
            rng=random.Random(7),
            time_provider=SequenceClock([0.0, 0.0]),
            charge_layout_time=False,
        )

        result = runner.play()

        self.assertEqual(result.winner, PlayerColor.BLUE)
        self.assertEqual(result.loser, PlayerColor.RED)
        self.assertEqual(result.reason, "layout_error:ValueError")

    def test_random_layout_mode_ignores_agent_layout_choice(self) -> None:
        runner = MatchRunner(
            self._players(BadLayoutAgent(), BadLayoutAgent()),
            rng=random.Random(7),
            time_provider=SequenceClock([0.0, 0.0]),
            charge_layout_time=False,
            layout_mode="random",
            max_turns=0,
        )

        result = runner.play()

        self.assertEqual(result.reason, "max_turns_exceeded")
        occupied = [
            value
            for row in result.final_snapshot.board
            for value in row
            if value != 0
        ]
        self.assertEqual(sorted(occupied), [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6])


if __name__ == "__main__":
    unittest.main()
