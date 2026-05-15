from __future__ import annotations

from abc import ABC, abstractmethod
import random
from typing import Sequence

from .engine import GameSnapshot, Move, PlayerColor, Position


class PlayerAgent(ABC):
    mode = "abstract"

    def __init__(self, name: str = "") -> None:
        self.name = name or self.mode

    @property
    def is_human(self) -> bool:
        return False

    def choose_layout(
        self,
        color: PlayerColor,
        starting_positions: Sequence[Position],
        rng: random.Random,
    ) -> dict[int, Position]:
        del color, rng
        numbers = [1, 2, 3, 4, 5, 6]
        return {
            number: position
            for number, position in zip(numbers, starting_positions)
        }

    @abstractmethod
    def choose_move(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move:
        raise NotImplementedError


class HumanPlayer(PlayerAgent):
    mode = "human"

    @property
    def is_human(self) -> bool:
        return True

    def choose_move(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move:
        del snapshot, legal_moves
        raise RuntimeError("HumanPlayer moves should be provided by the UI.")


class RandomAIPlayer(PlayerAgent):
    mode = "random_ai"

    def __init__(self, name: str = "RandomAI", rng: random.Random | None = None) -> None:
        super().__init__(name=name)
        self.rng = rng or random.Random()

    def choose_move(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move:
        del snapshot
        if not legal_moves:
            raise ValueError("No legal moves are available.")
        return self.rng.choice(list(legal_moves))

