from __future__ import annotations

from dataclasses import dataclass
import random
import time
from typing import Callable, Mapping, Sequence

from .engine import EinsteinGame, GameSnapshot, Move, PlayerColor, Position
from .players import PlayerAgent


TimeProvider = Callable[[], float]


@dataclass(frozen=True)
class MatchStep:
    turn_index: int
    color: PlayerColor
    dice_roll: int
    legal_move_count: int
    move: Move | None
    elapsed_seconds: float
    red_seconds_left: float
    blue_seconds_left: float
    note: str


@dataclass(frozen=True)
class MatchResult:
    winner: PlayerColor | None
    loser: PlayerColor | None
    reason: str
    final_snapshot: GameSnapshot
    steps: tuple[MatchStep, ...]
    red_seconds_left: float
    blue_seconds_left: float


class MatchRunner:
    """Runs a complete timed game between two PlayerAgent instances."""

    def __init__(
        self,
        players: Mapping[PlayerColor, PlayerAgent],
        total_seconds: float = 15 * 60,
        game: EinsteinGame | None = None,
        rng: random.Random | None = None,
        time_provider: TimeProvider | None = None,
        charge_layout_time: bool = True,
        max_turns: int = 1000,
    ) -> None:
        self.players = dict(players)
        self.total_seconds = float(total_seconds)
        self.rng = rng or random.Random()
        self.game = game or EinsteinGame(rng=self.rng)
        self.time_provider = time_provider or time.perf_counter
        self.charge_layout_time = charge_layout_time
        self.max_turns = max_turns
        self.remaining_seconds = {
            PlayerColor.RED: self.total_seconds,
            PlayerColor.BLUE: self.total_seconds,
        }
        self.steps: list[MatchStep] = []

    def play(self) -> MatchResult:
        setup_result = self._setup_game()
        if setup_result is not None:
            return setup_result

        while self.game.winner is None:
            if len(self.steps) >= self.max_turns:
                return self._finish(
                    winner=None,
                    loser=None,
                    reason="max_turns_exceeded",
                )

            color = self.game.current_player
            if self.remaining_seconds[color] <= 0:
                return self._finish(
                    winner=color.opponent,
                    loser=color,
                    reason="timeout",
                )

            legal_moves = self.game.get_legal_moves()
            if not legal_moves:
                turn_index = self.game.turn_index
                dice_roll = self.game.dice_roll
                self._append_step(
                    turn_index=turn_index,
                    color=color,
                    dice_roll=dice_roll,
                    legal_move_count=0,
                    move=None,
                    elapsed_seconds=0.0,
                    note="pass_no_legal_moves",
                )
                self.game.pass_turn()
                continue

            turn_index = self.game.turn_index
            move, elapsed_seconds, error = self._call_choose_move(color, legal_moves)
            self.remaining_seconds[color] -= elapsed_seconds

            if self.remaining_seconds[color] < 0:
                return self._finish(
                    winner=color.opponent,
                    loser=color,
                    reason="timeout",
                )
            if error is not None:
                return self._finish(
                    winner=color.opponent,
                    loser=color,
                    reason=f"agent_error:{type(error).__name__}",
                )
            if move not in legal_moves:
                return self._finish(
                    winner=color.opponent,
                    loser=color,
                    reason="illegal_move",
                )

            dice_roll = self.game.dice_roll
            winner = self.game.apply_move(move)
            self._append_step(
                turn_index=turn_index,
                color=color,
                dice_roll=dice_roll,
                legal_move_count=len(legal_moves),
                move=move,
                elapsed_seconds=elapsed_seconds,
                note="move",
            )
            if winner is not None:
                return self._finish(
                    winner=winner,
                    loser=winner.opponent,
                    reason=self._win_reason(move),
                )

        return self._finish(
            winner=self.game.winner,
            loser=None if self.game.winner is None else self.game.winner.opponent,
            reason="finished",
        )

    def _setup_game(self) -> MatchResult | None:
        layouts: dict[PlayerColor, Mapping[int, Position]] = {}
        for color in (PlayerColor.RED, PlayerColor.BLUE):
            layout, elapsed_seconds, error = self._call_choose_layout(color)
            if self.charge_layout_time:
                self.remaining_seconds[color] -= elapsed_seconds
                if self.remaining_seconds[color] < 0:
                    return self._finish(
                        winner=color.opponent,
                        loser=color,
                        reason="layout_timeout",
                    )
            if error is not None:
                return self._finish(
                    winner=color.opponent,
                    loser=color,
                    reason=f"layout_error:{type(error).__name__}",
                )
            try:
                layout = self.game._normalize_layout(color, layout)
            except Exception as exc:
                return self._finish(
                    winner=color.opponent,
                    loser=color,
                    reason=f"layout_error:{type(exc).__name__}",
                )
            layouts[color] = layout

        try:
            self.game.reset(
                red_layout=layouts[PlayerColor.RED],
                blue_layout=layouts[PlayerColor.BLUE],
            )
        except Exception as exc:
            return self._finish(winner=None, loser=None, reason=f"setup_error:{type(exc).__name__}")
        return None

    def _call_choose_layout(
        self, color: PlayerColor
    ) -> tuple[Mapping[int, Position], float, Exception | None]:
        agent = self.players[color]
        start = self.time_provider()
        try:
            layout = agent.choose_layout(color, color.start_positions, self.rng)
            error = None
        except Exception as exc:
            layout = EinsteinGame.default_layout_for(color)
            error = exc
        elapsed = max(0.0, self.time_provider() - start)
        return layout, elapsed, error

    def _call_choose_move(
        self, color: PlayerColor, legal_moves: Sequence[Move]
    ) -> tuple[Move | None, float, Exception | None]:
        agent = self.players[color]
        snapshot = self.game.snapshot()
        start = self.time_provider()
        try:
            move = agent.choose_move(snapshot, legal_moves)
            error = None
        except Exception as exc:
            move = None
            error = exc
        elapsed = max(0.0, self.time_provider() - start)
        return move, elapsed, error

    def _append_step(
        self,
        turn_index: int,
        color: PlayerColor,
        dice_roll: int,
        legal_move_count: int,
        move: Move | None,
        elapsed_seconds: float,
        note: str,
    ) -> None:
        self.steps.append(
            MatchStep(
                turn_index=turn_index,
                color=color,
                dice_roll=dice_roll,
                legal_move_count=legal_move_count,
                move=move,
                elapsed_seconds=elapsed_seconds,
                red_seconds_left=self.remaining_seconds[PlayerColor.RED],
                blue_seconds_left=self.remaining_seconds[PlayerColor.BLUE],
                note=note,
            )
        )

    def _finish(
        self,
        winner: PlayerColor | None,
        loser: PlayerColor | None,
        reason: str,
    ) -> MatchResult:
        return MatchResult(
            winner=winner,
            loser=loser,
            reason=reason,
            final_snapshot=self.game.snapshot(),
            steps=tuple(self.steps),
            red_seconds_left=self.remaining_seconds[PlayerColor.RED],
            blue_seconds_left=self.remaining_seconds[PlayerColor.BLUE],
        )

    def _win_reason(self, move: Move) -> str:
        if move.to_position == move.color.goal:
            return "goal"
        return "capture_all"
