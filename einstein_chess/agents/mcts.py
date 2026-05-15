from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from ..engine import EinsteinGame, GameSnapshot, Move, Piece, PlayerColor
from ..players import PlayerAgent


@dataclass
class MCTSSearchStats:
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class MCTSAgent(PlayerAgent):
    mode = "mcts"

    def __init__(
        self,
        name: str = "MCTS",
        simulations: int = 200,
        exploration: float = 1.4,
        max_rollout_steps: int = 160,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(name=name)
        if simulations < 1:
            raise ValueError("simulations must be at least 1.")
        if max_rollout_steps < 1:
            raise ValueError("max_rollout_steps must be at least 1.")
        self.simulations = simulations
        self.exploration = exploration
        self.max_rollout_steps = max_rollout_steps
        self.rng = rng or random.Random()
        self.last_visit_counts: dict[Move, int] = {}
        self.last_policy: dict[Move, float] = {}

    def choose_move(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move:
        moves = tuple(legal_moves)
        if not moves:
            raise ValueError("No legal moves are available.")

        immediate_win = self._find_immediate_win(snapshot, moves)
        if immediate_win is not None:
            self.last_visit_counts = {move: int(move == immediate_win) for move in moves}
            self.last_policy = {move: float(move == immediate_win) for move in moves}
            return immediate_win

        root_color = snapshot.current_player
        stats = {move: MCTSSearchStats() for move in moves}

        for _ in range(self.simulations):
            move = self._select_root_move(stats)
            game = self._game_from_snapshot(snapshot)
            game.apply_move(move)
            reward = self._rollout(game, root_color)
            stats[move].visits += 1
            stats[move].value_sum += reward

        self.last_visit_counts = {
            move: move_stats.visits for move, move_stats in stats.items()
        }
        self.last_policy = self._build_policy(self.last_visit_counts)
        return max(
            moves,
            key=lambda move: (stats[move].visits, stats[move].mean_value),
        )

    def _select_root_move(self, stats: dict[Move, MCTSSearchStats]) -> Move:
        unvisited = [move for move, move_stats in stats.items() if move_stats.visits == 0]
        if unvisited:
            return self.rng.choice(unvisited)

        total_visits = sum(move_stats.visits for move_stats in stats.values())
        log_total = math.log(max(1, total_visits))
        return max(
            stats,
            key=lambda move: (
                stats[move].mean_value
                + self.exploration
                * math.sqrt(log_total / stats[move].visits)
            ),
        )

    def _rollout(self, game: EinsteinGame, root_color: PlayerColor) -> float:
        for _ in range(self.max_rollout_steps):
            if game.winner is not None:
                return self._reward(game.winner, root_color)

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                game.pass_turn()
                continue

            move = self._select_rollout_move(game, legal_moves)
            game.apply_move(move)

        return self._evaluate_non_terminal(game, root_color)

    def _select_rollout_move(
        self, game: EinsteinGame, legal_moves: Sequence[Move]
    ) -> Move:
        for move in legal_moves:
            if move.to_position == move.color.goal:
                return move

        capturing_moves = [
            move
            for move in legal_moves
            if game.get_piece(move.to_position) is not None
            and game.get_piece(move.to_position).color is move.color.opponent
        ]
        if capturing_moves:
            return self.rng.choice(capturing_moves)

        return self.rng.choice(list(legal_moves))

    def _find_immediate_win(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move | None:
        for move in legal_moves:
            game = self._game_from_snapshot(snapshot)
            winner = game.apply_move(move)
            if winner is snapshot.current_player:
                return move
        return None

    def _reward(self, winner: PlayerColor, root_color: PlayerColor) -> float:
        return 1.0 if winner is root_color else -1.0

    def _evaluate_non_terminal(
        self, game: EinsteinGame, root_color: PlayerColor
    ) -> float:
        own_distance = self._best_goal_distance(game, root_color)
        enemy_distance = self._best_goal_distance(game, root_color.opponent)
        own_count = len(game.get_piece_numbers(root_color))
        enemy_count = len(game.get_piece_numbers(root_color.opponent))
        distance_score = (enemy_distance - own_distance) / 8.0
        material_score = (own_count - enemy_count) / 6.0
        return max(-1.0, min(1.0, distance_score + 0.4 * material_score))

    def _best_goal_distance(self, game: EinsteinGame, color: PlayerColor) -> int:
        distances: list[int] = []
        goal_row, goal_col = color.goal
        for row in range(EinsteinGame.BOARD_SIZE):
            for col in range(EinsteinGame.BOARD_SIZE):
                piece = game.board[row][col]
                if piece is not None and piece.color is color:
                    distances.append(max(abs(goal_row - row), abs(goal_col - col)))
        if not distances:
            return 8
        return min(distances)

    def _build_policy(self, visit_counts: dict[Move, int]) -> dict[Move, float]:
        total_visits = sum(visit_counts.values())
        if total_visits == 0:
            return {move: 0.0 for move in visit_counts}
        return {
            move: visits / total_visits
            for move, visits in visit_counts.items()
        }

    def _game_from_snapshot(self, snapshot: GameSnapshot) -> EinsteinGame:
        game = EinsteinGame(rng=self.rng)
        game.board = [
            [None for _ in range(EinsteinGame.BOARD_SIZE)]
            for _ in range(EinsteinGame.BOARD_SIZE)
        ]
        for row_index, row in enumerate(snapshot.board):
            for col_index, encoded_piece in enumerate(row):
                if encoded_piece == 0:
                    continue
                color = PlayerColor.RED if encoded_piece > 0 else PlayerColor.BLUE
                game.board[row_index][col_index] = Piece(
                    color=color,
                    number=abs(encoded_piece),
                )
        game.current_player = snapshot.current_player
        game.dice_roll = snapshot.dice_roll
        game.winner = snapshot.winner
        game.turn_index = snapshot.turn_index
        game.move_history = []
        return game
