"""Root Neural MCTS 智能体。

这个版本用于 V2 和 V3 数据生成。它只在当前根局面统计合法走法：

网络策略头提供每个走法的初始推荐程度 prior；
执行一步后，网络价值头直接评价下一局面；
搜索统计再对网络的初始建议进行修正。
"""

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


@dataclass
class NeuralMCTSSearchStats:
    """根节点某条走法的网络先验和搜索统计。"""

    prior: float
    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        """返回该走法多次价值评估的平均结果。"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class NeuralMCTSAgent(PlayerAgent):
    """使用策略价值网络指导根节点搜索的智能体。"""

    mode = "neural_mcts"

    def __init__(
        self,
        checkpoint_path: str | Path,
        name: str = "NeuralMCTS",
        simulations: int = 80,
        c_puct: float = 1.5,
        device: str = "cpu",
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(name=name)
        if simulations < 1:
            raise ValueError("simulations must be at least 1.")
        self.checkpoint_path = Path(checkpoint_path)
        self.simulations = simulations
        self.c_puct = c_puct
        self.device = torch.device(device)
        self.rng = rng or random.Random()
        # 搜索阶段网络被冻结，只负责推理，不会在这里进行反向训练。
        self.model = self._load_model(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.last_visit_counts: dict[Move, int] = {}
        self.last_policy: dict[Move, float] = {}
        self.last_priors: dict[Move, float] = {}

    def choose_move(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move:
        """使用网络先验和一步后价值评估选择当前走法。"""
        moves = tuple(legal_moves)
        if not moves:
            raise ValueError("No legal moves are available.")

        # 明确的一步获胜不需要依赖网络判断。
        immediate_win = self._find_immediate_win(snapshot, moves)
        if immediate_win is not None:
            self.last_visit_counts = {move: int(move == immediate_win) for move in moves}
            self.last_policy = {move: float(move == immediate_win) for move in moves}
            self.last_priors = self._root_priors(snapshot, moves)
            return immediate_win

        root_color = snapshot.current_player
        # V1/V2/V3 策略头先告诉搜索器哪些合法走法更值得关注。
        priors = self._root_priors(snapshot, moves)
        stats = {
            move: NeuralMCTSSearchStats(prior=priors[move])
            for move in moves
        }

        for _ in range(self.simulations):
            # PUCT 同时参考平均价值、网络 prior 和访问次数。
            move = self._select_root_move(stats)
            game = self._game_from_snapshot(snapshot)
            winner = game.apply_move(move)
            if winner is not None:
                reward = 1.0 if winner is root_color else -1.0
            else:
                # 与基础 MCTS 不同，这里不进行长 rollout，而是让价值头评价下一局面。
                reward = self._evaluate_leaf(game.snapshot(), root_color)
            stats[move].visits += 1
            stats[move].value_sum += reward

        self.last_visit_counts = {
            move: move_stats.visits for move, move_stats in stats.items()
        }
        self.last_policy = self._build_policy(self.last_visit_counts)
        self.last_priors = priors
        # 搜索完成后仍然选择访问次数最多的走法。
        return max(
            moves,
            key=lambda move: (stats[move].visits, stats[move].mean_value),
        )

    def _select_root_move(
        self, stats: dict[Move, NeuralMCTSSearchStats]
    ) -> Move:
        """使用 PUCT 选择下一次要测试的根节点走法。"""
        total_visits = sum(move_stats.visits for move_stats in stats.values())
        sqrt_total = math.sqrt(total_visits + 1)
        return max(
            stats,
            key=lambda move: (
                stats[move].mean_value
                + self.c_puct
                * stats[move].prior
                * sqrt_total
                / (1 + stats[move].visits)
            ),
        )

    def _root_priors(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> dict[Move, float]:
        """从网络策略头取得当前合法走法的初始推荐概率。"""
        policy = self._predict_policy(snapshot)
        priors = {
            move: float(policy[move_to_action_id(move)])
            for move in legal_moves
        }
        total = sum(priors.values())
        # 网络没有给出有效概率时，退化为所有合法走法均匀分布。
        if total <= 0:
            uniform = 1.0 / len(legal_moves)
            return {move: uniform for move in legal_moves}
        return {move: prior / total for move, prior in priors.items()}

    def _predict_policy(self, snapshot: GameSnapshot) -> np.ndarray:
        """将局面输入网络，并只在合法动作中生成策略概率。"""
        state = torch.from_numpy(encode_state(snapshot)).unsqueeze(0).to(self.device)
        mask = legal_action_mask(snapshot)
        # no_grad 表示这里只做预测，不修改网络参数。
        with torch.no_grad():
            policy_logits, _ = self.model(state)
        logits = policy_logits.squeeze(0).detach().cpu().numpy()
        return _softmax_masked(logits, mask)

    def _evaluate_leaf(self, snapshot: GameSnapshot, root_color: PlayerColor) -> float:
        """使用价值头评价执行一步后的局面，并转换为根节点玩家视角。"""
        state = torch.from_numpy(encode_state(snapshot)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, value = self.model(state)
        leaf_value = float(value.squeeze(0).detach().cpu())
        # 网络 value 总是从 snapshot.current_player 角度输出；视角不同时需要取负号。
        if snapshot.current_player is root_color:
            return leaf_value
        return -leaf_value

    def _find_immediate_win(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move | None:
        """检查是否存在一步获胜走法。"""
        for move in legal_moves:
            game = self._game_from_snapshot(snapshot)
            winner = game.apply_move(move)
            if winner is snapshot.current_player:
                return move
        return None

    def _build_policy(self, visit_counts: dict[Move, int]) -> dict[Move, float]:
        """将根节点访问次数转换为自我对弈策略标签。"""
        total_visits = sum(visit_counts.values())
        if total_visits == 0:
            return {move: 0.0 for move in visit_counts}
        return {
            move: visits / total_visits
            for move, visits in visit_counts.items()
        }

    def _load_model(self, checkpoint_path: Path) -> PolicyValueNet:
        """从 checkpoint 中恢复与训练时结构一致的策略价值网络。"""
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
        """将只读快照恢复为可修改棋局，供搜索执行假想走法。"""
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


def _softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """只在合法动作位置上将 logits 转换为概率。"""
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


# 下一步阅读：einstein_chess/agents/full_neural_mcts.py
