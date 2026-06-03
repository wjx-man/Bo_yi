"""基于人工规则打分的爱恩斯坦棋智能体。"""

from __future__ import annotations

import random

from src.env.board import occupant_at
from src.env.env import EinsteinChessEnv
from src.env.rules import DELTAS, GOAL_CORNERS, id_to_action, opponent

from .base import Agent


class RuleBasedAgent(Agent):
    """使用简单战术规则为合法动作打分，作为评估基准。"""

    name = "rule_based"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, env: EinsteinChessEnv) -> int:
        # 先为每个合法动作打分，再从最高分动作中随机选择，避免固定走法。
        scored = [(self._score_action(env, action), action) for action in env.legal_actions()]
        if not scored:
            raise RuntimeError("No legal actions available")
        best = max(score for score, _ in scored)
        best_actions = [action for score, action in scored if score == best]
        return self.rng.choice(best_actions)

    def _score_action(self, env: EinsteinChessEnv, action: int) -> float:
        """综合前进距离、到达目标角和吃子情况计算动作分数。"""
        player = env.current_player
        piece_id, direction = id_to_action(action)
        pos = env.positions[player][piece_id]
        assert pos is not None
        dr, dc = DELTAS[player][direction]
        target = (pos[0] + dr, pos[1] + dc)
        goal = GOAL_CORNERS[player]
        old_dist = abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])
        new_dist = abs(goal[0] - target[0]) + abs(goal[1] - target[1])
        # 向目标角靠近时得分增加，远离目标角时得分降低。
        score = float(old_dist - new_dist)
        if target == goal:
            # 到达目标角会直接获胜，因此给予远高于普通走法的分数。
            score += 100.0
        captured = occupant_at(env.positions, target)
        if captured is not None and captured.player == opponent(player):
            score += 10.0
        elif captured is not None and captured.player == player:
            # 规则允许吃己方棋子，但通常不是好选择，所以给予惩罚。
            score -= 5.0
        return score

