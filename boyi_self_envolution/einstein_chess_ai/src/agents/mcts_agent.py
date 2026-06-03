"""轻量级蒙特卡洛树搜索风格智能体。"""

from __future__ import annotations

import math
import random

from src.agents.base import Agent
from src.agents.random_agent import RandomAgent
from src.agents.rule_based import RuleBasedAgent
from src.env.env import EinsteinChessEnv
from src.env.rules import BLUE, GOAL_CORNERS, PIECE_IDS, RED, opponent


class MCTSAgent(Agent):
    """只在根节点分配模拟次数的轻量级 MCTS 基准。

    每次选择动作时，算法反复模拟未来棋局，并用 UCB 在“尝试新动作”和
    “继续验证高分动作”之间平衡。模拟阶段主要使用规则智能体，并加入少量随机探索。
    """

    name = "mcts"

    def __init__(
        self,
        simulations: int = 80,
        max_depth: int = 12,
        exploration: float = 1.4,
        seed: int | None = None,
        rollout_random_prob: float = 0.15,
    ) -> None:
        self.simulations = int(simulations)
        self.max_depth = int(max_depth)
        self.exploration = float(exploration)
        self.rng = random.Random(seed)
        self.rollout_random_prob = float(rollout_random_prob)
        self.rule_rollout = RuleBasedAgent(seed=seed)
        self.random_rollout = RandomAgent(seed=seed)

    def select_action(self, env: EinsteinChessEnv) -> int:
        """模拟每个根动作的未来结果，返回平均价值最高的动作。"""
        actions = env.legal_actions()
        if not actions:
            raise RuntimeError("No legal actions available")
        if len(actions) == 1:
            return actions[0]

        root_player = env.current_player
        visits = {action: 0 for action in actions}
        totals = {action: 0.0 for action in actions}

        for _ in range(max(1, self.simulations)):
            # 每次模拟选择一个根动作，并在复制的环境中继续向后推演。
            action = self._select_root_action(actions, visits, totals)
            sim_env = env.clone()
            sim_env.rng.seed(self.rng.randint(0, 2**31 - 1))
            _, _, done, _ = sim_env.step(action)
            value = self._evaluate_rollout(sim_env, root_player, remaining_depth=self.max_depth - 1, done=done)
            visits[action] += 1
            totals[action] += value

        return max(actions, key=lambda action: totals[action] / max(1, visits[action]))

    def _select_root_action(self, actions: list[int], visits: dict[int, int], totals: dict[int, float]) -> int:
        """使用 UCB 公式选择下一次需要模拟的根动作。"""
        unvisited = [action for action in actions if visits[action] == 0]
        if unvisited:
            return self.rng.choice(unvisited)
        total_visits = sum(visits.values())
        return max(
            actions,
            # 前半部分偏好平均收益高的动作，后半部分偏好访问次数少的动作。
            key=lambda action: totals[action] / visits[action]
            + self.exploration * math.sqrt(math.log(total_visits + 1) / visits[action]),
        )

    def _evaluate_rollout(self, env: EinsteinChessEnv, root_player: str, remaining_depth: int, done: bool) -> float:
        """继续模拟若干步；终局返回胜负，否则使用启发式函数估值。"""
        if done or env.is_terminal():
            return self._terminal_value(env, root_player)
        for _ in range(max(0, remaining_depth)):
            if env.is_terminal():
                return self._terminal_value(env, root_player)
            agent = self.random_rollout if self.rng.random() < self.rollout_random_prob else self.rule_rollout
            _, _, done, _ = env.step(agent.select_action(env))
            if done:
                return self._terminal_value(env, root_player)
        return self._heuristic_value(env, root_player)

    def _terminal_value(self, env: EinsteinChessEnv, root_player: str) -> float:
        return 1.0 if env.winner == root_player else -1.0

    def _heuristic_value(self, env: EinsteinChessEnv, root_player: str) -> float:
        """根据存活棋子数量和向目标角的推进程度估计局面价值。"""
        opp = opponent(root_player)
        own_alive = len(env.alive_pieces(root_player))
        opp_alive = len(env.alive_pieces(opp))
        material = (own_alive - opp_alive) / 6.0
        own_progress = self._progress_score(env, root_player)
        opp_progress = self._progress_score(env, opp)
        value = 0.65 * material + 0.35 * (own_progress - opp_progress)
        return max(-1.0, min(1.0, value))

    def _progress_score(self, env: EinsteinChessEnv, player: str) -> float:
        goal = GOAL_CORNERS[player]
        distances = []
        for piece_id in PIECE_IDS:
            pos = env.positions[player][piece_id]
            if pos is not None:
                distances.append(abs(goal[0] - pos[0]) + abs(goal[1] - pos[1]))
        if not distances:
            return -1.0
        return 1.0 - min(distances) / 8.0

