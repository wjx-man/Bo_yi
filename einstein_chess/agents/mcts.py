"""项目最早期的基础 MCTS 智能体。

这个版本用于解决冷启动问题：在还没有棋谱、训练数据和神经网络时，
它只依赖规则引擎，通过大量假想比赛判断当前合法走法的好坏。

需要注意：它只保存当前根局面各个走法的统计，没有保存完整多层搜索树。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Sequence

from ..engine import EinsteinGame, GameSnapshot, Move, Piece, PlayerColor
from ..players import PlayerAgent


@dataclass
class MCTSSearchStats:
    """某一条根节点走法的搜索统计。"""

    visits: int = 0
    value_sum: float = 0.0

    @property
    def mean_value(self) -> float:
        """返回该走法所有模拟结果的平均分。"""
        if self.visits == 0:
            return 0.0
        return self.value_sum / self.visits


class MCTSAgent(PlayerAgent):
    """根节点 UCT 选择 + 启发式随机 rollout 的基础智能体。"""

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
        # simulations 表示每次真正走棋前，要进行多少次内部假想比赛。
        self.simulations = simulations
        self.exploration = exploration
        self.max_rollout_steps = max_rollout_steps
        self.rng = rng or random.Random()
        # 这两个字段会被自我对弈模块读取，并转换为神经网络的策略答案。
        self.last_visit_counts: dict[Move, int] = {}
        self.last_policy: dict[Move, float] = {}

    def choose_move(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move:
        """对当前所有合法走法进行多次假想比赛，并返回最终选择。"""
        moves = tuple(legal_moves)
        if not moves:
            raise ValueError("No legal moves are available.")

        # 如果存在一步即可获胜的走法，就没有必要继续搜索。
        immediate_win = self._find_immediate_win(snapshot, moves)
        if immediate_win is not None:
            self.last_visit_counts = {move: int(move == immediate_win) for move in moves}
            self.last_policy = {move: float(move == immediate_win) for move in moves}
            return immediate_win

        # 所有 rollout 的收益都统一从当前根节点玩家的角度计算。
        root_color = snapshot.current_player
        stats = {move: MCTSSearchStats() for move in moves}

        # 每次循环都是一场只存在于内存中的“假想比赛”。
        for _ in range(self.simulations):
            # 先决定本次要测试当前局面的哪一条走法。
            move = self._select_root_move(stats)
            # 每次模拟都从原始局面的副本开始，不能修改真实比赛棋盘。
            game = self._game_from_snapshot(snapshot)
            game.apply_move(move)
            # 继续模拟后续对局，得到该走法对根节点玩家的收益。
            reward = self._rollout(game, root_color)
            stats[move].visits += 1
            stats[move].value_sum += reward

        # 访问次数分布既用于最终决策，也会作为后续神经网络的 policy 标签。
        self.last_visit_counts = {
            move: move_stats.visits for move, move_stats in stats.items()
        }
        self.last_policy = self._build_policy(self.last_visit_counts)
        # 最终优先选择访问次数最多的走法；次数相同时再比较平均价值。
        return max(
            moves,
            key=lambda move: (stats[move].visits, stats[move].mean_value),
        )

    def _select_root_move(self, stats: dict[Move, MCTSSearchStats]) -> Move:
        """使用 UCT 思想选择下一次要测试的根节点走法。"""
        # 未访问过的走法必须先尝试，避免某些可能优秀的走法从未被观察。
        unvisited = [move for move, move_stats in stats.items() if move_stats.visits == 0]
        if unvisited:
            return self.rng.choice(unvisited)

        # 所有走法都访问过后，平衡“当前平均收益”和“探索不足的奖励”。
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
        """从已经执行根节点走法的局面开始，模拟后续比赛。"""
        for _ in range(self.max_rollout_steps):
            if game.winner is not None:
                return self._reward(game.winner, root_color)

            legal_moves = game.get_legal_moves()
            if not legal_moves:
                game.pass_turn()
                continue

            # rollout 不追求完美下棋，只使用少量常识加随机选择快速走到未来。
            move = self._select_rollout_move(game, legal_moves)
            game.apply_move(move)

        # 达到最大模拟步数仍未结束时，用启发式函数估计局面，而不是无限模拟。
        return self._evaluate_non_terminal(game, root_color)

    def _select_rollout_move(
        self, game: EinsteinGame, legal_moves: Sequence[Move]
    ) -> Move:
        """为假想比赛选择走法：能获胜优先，其次吃子，否则随机。"""
        # 第一优先级：一步到达目标角。
        for move in legal_moves:
            if move.to_position == move.color.goal:
                return move

        # 第二优先级：吃掉对方棋子。这里不会把“自吃”当作优先动作。
        capturing_moves = [
            move
            for move in legal_moves
            if game.get_piece(move.to_position) is not None
            and game.get_piece(move.to_position).color is move.color.opponent
        ]
        if capturing_moves:
            return self.rng.choice(capturing_moves)

        # 没有明显战术机会时，从合法走法中随机选择。
        return self.rng.choice(list(legal_moves))

    def _find_immediate_win(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move | None:
        """逐个执行合法走法，检查是否存在一步获胜。"""
        for move in legal_moves:
            game = self._game_from_snapshot(snapshot)
            winner = game.apply_move(move)
            if winner is snapshot.current_player:
                return move
        return None

    def _reward(self, winner: PlayerColor, root_color: PlayerColor) -> float:
        """将终局胜负转换为根节点玩家视角的 +1 或 -1。"""
        return 1.0 if winner is root_color else -1.0

    def _evaluate_non_terminal(
        self, game: EinsteinGame, root_color: PlayerColor
    ) -> float:
        """模拟未结束时，根据目标距离和棋子数量粗略评价局面。"""
        own_distance = self._best_goal_distance(game, root_color)
        enemy_distance = self._best_goal_distance(game, root_color.opponent)
        own_count = len(game.get_piece_numbers(root_color))
        enemy_count = len(game.get_piece_numbers(root_color.opponent))
        # 我方越接近目标角、对方越远，distance_score 越高。
        distance_score = (enemy_distance - own_distance) / 8.0
        # 我方剩余棋子越多，material_score 越高。
        material_score = (own_count - enemy_count) / 6.0
        return max(-1.0, min(1.0, distance_score + 0.4 * material_score))

    def _best_goal_distance(self, game: EinsteinGame, color: PlayerColor) -> int:
        """返回某一方最接近目标角的棋子距离。"""
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
        """将走法访问次数归一化为总和为 1 的策略分布。"""
        total_visits = sum(visit_counts.values())
        if total_visits == 0:
            return {move: 0.0 for move in visit_counts}
        return {
            move: visits / total_visits
            for move, visits in visit_counts.items()
        }

    def _game_from_snapshot(self, snapshot: GameSnapshot) -> EinsteinGame:
        """将只读快照恢复成可修改棋局，供内部搜索使用。"""
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


# 下一步阅读：einstein_chess/training/self_play.py
