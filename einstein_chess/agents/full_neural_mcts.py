from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import random
from typing import Sequence

import numpy as np
import torch

from ..engine import EinsteinGame, GameSnapshot, Move, Piece, PlayerColor
from ..players import PlayerAgent
from ..training.action_codec import ACTION_SIZE, legal_action_mask, move_to_action_id
from ..training.model import PolicyValueNet
from ..training.state_encoder import encode_state


StateKey = tuple[tuple[tuple[int, ...], ...], str, int]


@dataclass
class FullNeuralMCTSSearchStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


@dataclass
class FullNeuralMCTSNode:
    player: PlayerColor
    stats: dict[Move, FullNeuralMCTSSearchStats]


class FullNeuralMCTSAgent(PlayerAgent):
    """Multi-layer neural PUCT search with sampled or enumerated dice chance nodes."""

    mode = "full_neural_mcts"

    def __init__(
        self,
        checkpoint_path: str | Path,
        name: str = "FullNeuralMCTS",
        simulations: int = 80,
        c_puct: float = 1.5,
        max_depth: int = 12,
        chance_mode: str = "sample",
        device: str = "cpu",
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(name=name)
        if simulations < 1:
            raise ValueError("simulations must be at least 1.")
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        if chance_mode not in {"sample", "enumerate"}:
            raise ValueError("chance_mode must be 'sample' or 'enumerate'.")
        self.checkpoint_path = Path(checkpoint_path)
        self.simulations = simulations
        self.c_puct = c_puct
        self.max_depth = max_depth
        self.chance_mode = chance_mode
        self.device = torch.device(device)
        self.rng = rng or random.Random()
        self.model = self._load_model(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.last_visit_counts: dict[Move, int] = {}
        self.last_policy: dict[Move, float] = {}
        self.last_priors: dict[Move, float] = {}
        self._nodes: dict[StateKey, FullNeuralMCTSNode] = {}
        self._prediction_cache: dict[StateKey, tuple[np.ndarray, float]] = {}

    def choose_move(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move:
        moves = tuple(legal_moves)
        if not moves:
            raise ValueError("No legal moves are available.")

        self._nodes = {}
        self._prediction_cache = {}

        immediate_win = self._find_immediate_win(snapshot, moves)
        if immediate_win is not None:
            self.last_visit_counts = {move: int(move == immediate_win) for move in moves}
            self.last_policy = {move: float(move == immediate_win) for move in moves}
            self.last_priors = self._priors_for_moves(snapshot, moves)
            return immediate_win

        root_game = self._game_from_snapshot(snapshot)
        root_node = self._get_or_expand_node(snapshot, moves)
        for _ in range(self.simulations):
            self._search(root_game.clone(), depth=0)

        self.last_visit_counts = {
            move: move_stats.visits for move, move_stats in root_node.stats.items()
        }
        self.last_policy = self._build_policy(self.last_visit_counts)
        self.last_priors = {
            move: move_stats.prior for move, move_stats in root_node.stats.items()
        }
        return max(
            moves,
            key=lambda move: (
                root_node.stats[move].visits,
                root_node.stats[move].mean_value,
            ),
        )

    def _search(self, game: EinsteinGame, depth: int) -> float:
        if game.winner is not None:
            return 1.0 if game.winner is game.current_player else -1.0

        snapshot = game.snapshot()
        legal_moves = tuple(snapshot.legal_moves)
        if not legal_moves:
            if depth >= self.max_depth:
                return self._evaluate_snapshot(snapshot)
            child = game.clone()
            child.pass_turn()
            return -self._search(child, depth + 1)

        key = self._state_key(snapshot)
        node = self._nodes.get(key)
        if node is None:
            self._get_or_expand_node(snapshot, legal_moves)
            return self._evaluate_snapshot(snapshot)

        if depth >= self.max_depth:
            return self._evaluate_snapshot(snapshot)

        move = self._select_move(node)
        move_stats = node.stats[move]
        value = self._evaluate_child_after_move(game, move, depth)
        move_stats.visits += 1
        move_stats.value_sum += value
        return value

    def _evaluate_child_after_move(
        self,
        game: EinsteinGame,
        move: Move,
        depth: int,
    ) -> float:
        child = game.clone()
        winner = child.apply_move(move)
        if winner is not None:
            return 1.0 if winner is game.current_player else -1.0

        if self.chance_mode == "enumerate":
            total = 0.0
            for dice_roll in EinsteinGame.PIECE_NUMBERS:
                dice_child = child.clone()
                dice_child.dice_roll = dice_roll
                total += -self._search(dice_child, depth + 1)
            return total / len(EinsteinGame.PIECE_NUMBERS)

        child.dice_roll = self.rng.randint(1, 6)
        return -self._search(child, depth + 1)

    def _select_move(self, node: FullNeuralMCTSNode) -> Move:
        total_visits = sum(move_stats.visits for move_stats in node.stats.values())
        sqrt_total = math.sqrt(total_visits + 1)
        return max(
            node.stats,
            key=lambda move: (
                node.stats[move].mean_value
                + self.c_puct
                * node.stats[move].prior
                * sqrt_total
                / (1 + node.stats[move].visits)
            ),
        )

    def _get_or_expand_node(
        self,
        snapshot: GameSnapshot,
        legal_moves: Sequence[Move],
    ) -> FullNeuralMCTSNode:
        key = self._state_key(snapshot)
        node = self._nodes.get(key)
        if node is not None:
            return node

        priors = self._priors_for_moves(snapshot, legal_moves)
        node = FullNeuralMCTSNode(
            player=snapshot.current_player,
            stats={
                move: FullNeuralMCTSSearchStats(prior=priors[move])
                for move in legal_moves
            },
        )
        self._nodes[key] = node
        return node

    def _priors_for_moves(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> dict[Move, float]:
        policy, _ = self._predict(snapshot)
        priors = {
            move: float(policy[move_to_action_id(move)])
            for move in legal_moves
        }
        total = sum(priors.values())
        if total <= 0:
            uniform = 1.0 / len(legal_moves)
            return {move: uniform for move in legal_moves}
        return {move: prior / total for move, prior in priors.items()}

    def _evaluate_snapshot(self, snapshot: GameSnapshot) -> float:
        _, value = self._predict(snapshot)
        return value

    def _predict(self, snapshot: GameSnapshot) -> tuple[np.ndarray, float]:
        key = self._state_key(snapshot)
        cached = self._prediction_cache.get(key)
        if cached is not None:
            return cached

        state = torch.from_numpy(encode_state(snapshot)).unsqueeze(0).to(self.device)
        mask = legal_action_mask(snapshot)
        with torch.no_grad():
            policy_logits, value = self.model(state)
        logits = policy_logits.squeeze(0).detach().cpu().numpy()
        policy = _softmax_masked(logits, mask)
        predicted_value = float(value.squeeze(0).detach().cpu())
        self._prediction_cache[key] = (policy, predicted_value)
        return policy, predicted_value

    def _find_immediate_win(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move | None:
        for move in legal_moves:
            game = self._game_from_snapshot(snapshot)
            winner = game.apply_move(move)
            if winner is snapshot.current_player:
                return move
        return None

    def _build_policy(self, visit_counts: dict[Move, int]) -> dict[Move, float]:
        total_visits = sum(visit_counts.values())
        if total_visits == 0:
            return {move: 0.0 for move in visit_counts}
        return {
            move: visits / total_visits
            for move, visits in visit_counts.items()
        }

    def _load_model(self, checkpoint_path: Path) -> PolicyValueNet:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        config = checkpoint.get("config", {})
        hidden_channels = int(config.get("hidden_channels", 64))
        residual_blocks = int(config.get("residual_blocks", 0))
        model = PolicyValueNet(
            hidden_channels=hidden_channels,
            residual_blocks=residual_blocks,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

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

    def _state_key(self, snapshot: GameSnapshot) -> StateKey:
        return (snapshot.board, snapshot.current_player.value, snapshot.dice_roll)


def _softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    if not np.any(mask > 0):
        return policy
    valid_logits = logits[mask > 0]
    max_logit = float(np.max(valid_logits))
    valid_exp = np.exp(valid_logits - max_logit)
    valid_exp_sum = float(valid_exp.sum())
    if valid_exp_sum > 0:
        policy[mask > 0] = valid_exp / valid_exp_sum
    return policy
