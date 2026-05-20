from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import random
from typing import Mapping, Sequence

Position = tuple[int, int]


class PlayerColor(str, Enum):
    RED = "red"
    BLUE = "blue"

    @property
    def label(self) -> str:
        return "\u7ea2\u65b9" if self is PlayerColor.RED else "\u84dd\u65b9"

    @property
    def short_label(self) -> str:
        return "\u7ea2" if self is PlayerColor.RED else "\u84dd"

    @property
    def opponent(self) -> "PlayerColor":
        return PlayerColor.BLUE if self is PlayerColor.RED else PlayerColor.RED

    @property
    def goal(self) -> Position:
        return (4, 4) if self is PlayerColor.RED else (0, 0)

    @property
    def deltas(self) -> tuple[Position, Position, Position]:
        if self is PlayerColor.RED:
            return ((1, 0), (0, 1), (1, 1))
        return ((-1, 0), (0, -1), (-1, -1))

    @property
    def start_positions(self) -> tuple[Position, ...]:
        if self is PlayerColor.RED:
            return ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0))
        return ((4, 4), (4, 3), (4, 2), (3, 4), (3, 3), (2, 4))


@dataclass(frozen=True)
class Piece:
    color: PlayerColor
    number: int


@dataclass(frozen=True)
class Move:
    color: PlayerColor
    piece_number: int
    from_position: Position
    to_position: Position


@dataclass(frozen=True)
class GameSnapshot:
    board: tuple[tuple[int, ...], ...]
    current_player: PlayerColor
    dice_roll: int
    winner: PlayerColor | None
    legal_moves: tuple[Move, ...]
    turn_index: int

    def as_dict(self) -> dict[str, object]:
        return {
            "board": self.board,
            "current_player": self.current_player.value,
            "dice_roll": self.dice_roll,
            "winner": None if self.winner is None else self.winner.value,
            "legal_moves": [move.__dict__ for move in self.legal_moves],
            "turn_index": self.turn_index,
        }


class EinsteinGame:
    BOARD_SIZE = 5
    PIECE_NUMBERS = (1, 2, 3, 4, 5, 6)

    def __init__(
        self,
        red_layout: Mapping[int, Position] | Sequence[int] | None = None,
        blue_layout: Mapping[int, Position] | Sequence[int] | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.rng = rng or random.Random()
        self.board: list[list[Piece | None]] = []
        self.current_player = PlayerColor.RED
        self.dice_roll = 1
        self.winner: PlayerColor | None = None
        self.turn_index = 1
        self.move_history: list[Move] = []
        self.reset(red_layout=red_layout, blue_layout=blue_layout)

    @classmethod
    def default_layout_for(cls, color: PlayerColor) -> dict[int, Position]:
        return {
            number: position
            for number, position in zip(cls.PIECE_NUMBERS, color.start_positions)
        }

    @classmethod
    def random_layout_for(
        cls, color: PlayerColor, rng: random.Random | None = None
    ) -> dict[int, Position]:
        local_rng = rng or random.Random()
        numbers = list(cls.PIECE_NUMBERS)
        local_rng.shuffle(numbers)
        return {
            number: position
            for number, position in zip(numbers, color.start_positions)
        }

    def clone(self) -> "EinsteinGame":
        clone = EinsteinGame(rng=random.Random())
        clone.board = [[piece for piece in row] for row in self.board]
        clone.current_player = self.current_player
        clone.dice_roll = self.dice_roll
        clone.winner = self.winner
        clone.turn_index = self.turn_index
        clone.move_history = list(self.move_history)
        clone.rng.setstate(self.rng.getstate())
        return clone

    def reset(
        self,
        red_layout: Mapping[int, Position] | Sequence[int] | None = None,
        blue_layout: Mapping[int, Position] | Sequence[int] | None = None,
        starting_player: PlayerColor = PlayerColor.RED,
    ) -> None:
        self.board = [
            [None for _ in range(self.BOARD_SIZE)] for _ in range(self.BOARD_SIZE)
        ]
        self.move_history = []
        self.current_player = starting_player
        self.turn_index = 1
        self.winner = None

        self._place_layout(
            PlayerColor.RED,
            self._normalize_layout(PlayerColor.RED, red_layout),
        )
        self._place_layout(
            PlayerColor.BLUE,
            self._normalize_layout(PlayerColor.BLUE, blue_layout),
        )
        self.roll_dice()

    def roll_dice(self) -> int:
        self.dice_roll = self.rng.randint(1, 6)
        return self.dice_roll

    def _normalize_layout(
        self,
        color: PlayerColor,
        layout: Mapping[int, Position] | Sequence[int] | None,
    ) -> dict[int, Position]:
        if layout is None:
            return self.default_layout_for(color)

        if isinstance(layout, Mapping):
            normalized = dict(layout)
        else:
            if len(layout) != len(self.PIECE_NUMBERS):
                raise ValueError("Layout sequence must contain 6 numbers.")
            normalized = {
                number: position
                for number, position in zip(layout, color.start_positions)
            }

        expected_numbers = set(self.PIECE_NUMBERS)
        actual_numbers = set(normalized.keys())
        if actual_numbers != expected_numbers:
            raise ValueError(f"{color.label} layout must contain pieces 1-6.")

        positions = list(normalized.values())
        if len(set(positions)) != len(positions):
            raise ValueError(f"{color.label} layout contains duplicate positions.")

        valid_positions = set(color.start_positions)
        if set(positions) != valid_positions:
            raise ValueError(
                f"{color.label} layout must fill the 6 starting cells exactly."
            )

        return normalized

    def _place_layout(self, color: PlayerColor, layout: Mapping[int, Position]) -> None:
        for number, (row, col) in layout.items():
            self.board[row][col] = Piece(color=color, number=number)

    def get_piece(self, position: Position) -> Piece | None:
        row, col = position
        return self.board[row][col]

    def find_piece(self, color: PlayerColor, number: int) -> Position | None:
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                piece = self.board[row][col]
                if piece is not None and piece.color is color and piece.number == number:
                    return (row, col)
        return None

    def get_piece_numbers(self, color: PlayerColor) -> list[int]:
        numbers: list[int] = []
        for row in self.board:
            for piece in row:
                if piece is not None and piece.color is color:
                    numbers.append(piece.number)
        return sorted(numbers)

    def get_candidate_numbers(self, color: PlayerColor, dice_roll: int) -> list[int]:
        alive = self.get_piece_numbers(color)
        if dice_roll in alive:
            return [dice_roll]

        smaller = [number for number in alive if number < dice_roll]
        larger = [number for number in alive if number > dice_roll]
        candidates: list[int] = []
        if smaller:
            candidates.append(max(smaller))
        if larger:
            candidates.append(min(larger))
        return candidates

    def get_legal_moves(
        self,
        color: PlayerColor | None = None,
        dice_roll: int | None = None,
    ) -> list[Move]:
        if self.winner is not None:
            return []

        actual_color = color or self.current_player
        actual_roll = self.dice_roll if dice_roll is None else dice_roll
        moves: list[Move] = []
        for number in self.get_candidate_numbers(actual_color, actual_roll):
            position = self.find_piece(actual_color, number)
            if position is None:
                continue
            row, col = position
            for delta_row, delta_col in actual_color.deltas:
                next_row = row + delta_row
                next_col = col + delta_col
                if not self._is_inside(next_row, next_col):
                    continue
                moves.append(
                    Move(
                        color=actual_color,
                        piece_number=number,
                        from_position=position,
                        to_position=(next_row, next_col),
                    )
                )
        return moves

    def pass_turn(self) -> None:
        if self.winner is not None:
            raise ValueError("Game is already over.")
        if self.get_legal_moves():
            raise ValueError("Current player still has legal moves.")
        self.current_player = self.current_player.opponent
        self.turn_index += 1
        self.roll_dice()

    def apply_move(self, move: Move) -> PlayerColor | None:
        if self.winner is not None:
            raise ValueError("Game is already over.")

        legal_moves = self.get_legal_moves()
        if not legal_moves:
            raise ValueError("No legal moves are available; pass the turn first.")
        if move not in legal_moves:
            raise ValueError("Illegal move.")

        from_row, from_col = move.from_position
        to_row, to_col = move.to_position
        piece = self.board[from_row][from_col]
        if piece is None:
            raise ValueError("Source cell is empty.")
        if piece.color is not move.color or piece.number != move.piece_number:
            raise ValueError("Source piece does not match the move.")

        self.board[from_row][from_col] = None
        self.board[to_row][to_col] = Piece(color=move.color, number=move.piece_number)
        self.move_history.append(move)

        self.winner = self._resolve_winner(move.color, move.to_position)
        if self.winner is None:
            self.current_player = self.current_player.opponent
            self.turn_index += 1
            self.roll_dice()
        return self.winner

    def encode_board(self) -> tuple[tuple[int, ...], ...]:
        encoded_rows: list[tuple[int, ...]] = []
        for row in self.board:
            encoded_row: list[int] = []
            for piece in row:
                if piece is None:
                    encoded_row.append(0)
                elif piece.color is PlayerColor.RED:
                    encoded_row.append(piece.number)
                else:
                    encoded_row.append(-piece.number)
            encoded_rows.append(tuple(encoded_row))
        return tuple(encoded_rows)

    def snapshot(self) -> GameSnapshot:
        return GameSnapshot(
            board=self.encode_board(),
            current_player=self.current_player,
            dice_roll=self.dice_roll,
            winner=self.winner,
            legal_moves=tuple(self.get_legal_moves()),
            turn_index=self.turn_index,
        )

    def _resolve_winner(
        self, moved_color: PlayerColor, destination: Position
    ) -> PlayerColor | None:
        if destination == moved_color.goal:
            return moved_color
        if not self.get_piece_numbers(moved_color.opponent):
            return moved_color
        return None

    def _is_inside(self, row: int, col: int) -> bool:
        return 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE
