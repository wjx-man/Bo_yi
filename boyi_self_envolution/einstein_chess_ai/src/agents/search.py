"""用于对比评估的传统搜索智能体：Minimax、Alpha-Beta 和 MCTS。"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Iterable

from src.env.board import occupant_at
from src.env.env import EinsteinChessEnv
from src.env.rules import BLUE, DELTAS, GOAL_CORNERS, PIECE_IDS, RED, id_to_action, opponent

from .base import Agent


DICE_VALUES = tuple(range(1, 7))


def _advance_with_dice(env: EinsteinChessEnv, action: int, next_dice: int) -> EinsteinChessEnv:
    """复制环境、执行动作，并手动注入机会节点的下一次骰子结果。"""
    child = env.clone()
    player = child.current_player
    legal = child.legal_actions()
    child.apply_action(int(action), legal_actions_before=legal)
    child.step_count += 1

    if not child.is_terminal() and child.step_count >= child.max_steps:
        red_alive = len(child.alive_pieces(RED))
        blue_alive = len(child.alive_pieces(BLUE))
        child.winner = RED if red_alive >= blue_alive else BLUE
        child.win_reason = "max_steps_material"

    if not child.is_terminal():
        child.current_player = opponent(player)
        child.dice = int(next_dice)
    return child


def _terminal_value(env: EinsteinChessEnv, root_player: str) -> float | None:
    if env.winner is None:
        return None
    return 10_000.0 if env.winner == root_player else -10_000.0


def heuristic_value(env: EinsteinChessEnv, root_player: str) -> float:
    """从根玩家视角评价局面，分数越大表示越有利。"""
    terminal = _terminal_value(env, root_player)
    if terminal is not None:
        return terminal

    opp = opponent(root_player)
    score = 0.0
    # 棋子数量、推进程度和吃子压力共同构成非终局估值。
    score += 4.0 * (len(env.alive_pieces(root_player)) - len(env.alive_pieces(opp)))
    score += _progress_score(env, root_player)
    score -= _progress_score(env, opp)
    score += 0.4 * (_capture_pressure(env, root_player) - _capture_pressure(env, opp))
    return score


def normalized_heuristic_value(env: EinsteinChessEnv, root_player: str) -> float:
    """Return a compact value in [-1, 1] for MCTS rollouts."""
    terminal = _terminal_value(env, root_player)
    if terminal is not None:
        return 1.0 if terminal > 0 else -1.0
    return max(-1.0, min(1.0, heuristic_value(env, root_player) / 30.0))


def _progress_score(env: EinsteinChessEnv, player: str) -> float:
    goal = GOAL_CORNERS[player]
    score = 0.0
    best_distance = 8
    for pid in PIECE_IDS:
        pos = env.positions[player][pid]
        if pos is None:
            continue
        dist = abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])
        best_distance = min(best_distance, dist)
        score += (8 - dist) * 0.8
        if dist <= 2:
            score += 1.5
    score += (8 - best_distance) * 1.2
    return score


def _capture_pressure(env: EinsteinChessEnv, player: str) -> float:
    pressure = 0.0
    for action in env.legal_actions() if env.current_player == player else _pseudo_legal_actions(env, player):
        piece_id, direction = id_to_action(action)
        pos = env.positions[player][piece_id]
        if pos is None:
            continue
        dr, dc = DELTAS[player][direction]
        target = (pos[0] + dr, pos[1] + dc)
        captured = occupant_at(env.positions, target)
        if captured is not None and captured.player == opponent(player):
            pressure += 1.0
    return pressure


def _pseudo_legal_actions(env: EinsteinChessEnv, player: str) -> Iterable[int]:
    old_player = env.current_player
    try:
        env.current_player = player
        return list(env.legal_actions())
    finally:
        env.current_player = old_player


class MinimaxAgent(Agent):
    """对未来骰子结果求平均的期望极小化搜索智能体。"""

    name = "minimax"

    def __init__(self, depth: int = 2, seed: int | None = None) -> None:
        self.depth = int(depth)
        self.rng = random.Random(seed)

    def select_action(self, env: EinsteinChessEnv) -> int:
        """搜索每个合法动作的期望价值，选择最高分动作。"""
        actions = env.legal_actions()
        if not actions:
            raise RuntimeError("No legal actions available")
        root_player = env.current_player
        cache: dict[tuple, float] = {}
        scored = [(self._action_value(env, action, root_player, self.depth, cache), action) for action in actions]
        best = max(score for score, _ in scored)
        return self.rng.choice([action for score, action in scored if score == best])

    def _search(self, env: EinsteinChessEnv, root_player: str, depth: int, cache: dict[tuple, float]) -> float:
        """在决策节点执行 max/min 搜索，并使用缓存避免重复计算。"""
        terminal = _terminal_value(env, root_player)
        if terminal is not None or depth <= 0:
            return heuristic_value(env, root_player)
        key = _cache_key(env, depth)
        if key in cache:
            return cache[key]
        actions = env.legal_actions()
        if not actions:
            return heuristic_value(env, root_player)
        values = [self._action_value(env, action, root_player, depth, cache) for action in actions]
        # 根玩家希望分数最大，对手希望根玩家分数最小。
        value = max(values) if env.current_player == root_player else min(values)
        cache[key] = value
        return value

    def _action_value(
        self,
        env: EinsteinChessEnv,
        action: int,
        root_player: str,
        depth: int,
        cache: dict[tuple, float],
    ) -> float:
        """将六种等概率骰子结果的搜索值取平均，得到动作期望价值。"""
        first_child = _advance_with_dice(env, action, DICE_VALUES[0])
        if first_child.is_terminal():
            return heuristic_value(first_child, root_player)
        total = 0.0
        for dice in DICE_VALUES:
            child = first_child if dice == DICE_VALUES[0] else _advance_with_dice(env, action, dice)
            total += self._search(child, root_player, depth - 1, cache)
        return total / len(DICE_VALUES)


class AlphaBetaAgent(MinimaxAgent):
    """在决策节点加入 Alpha-Beta 剪枝的期望极小化搜索。"""

    name = "alpha_beta"

    def select_action(self, env: EinsteinChessEnv) -> int:
        actions = env.legal_actions()
        if not actions:
            raise RuntimeError("No legal actions available")
        root_player = env.current_player
        cache: dict[tuple, float] = {}
        alpha = -math.inf
        beta = math.inf
        best_score = -math.inf
        best_actions: list[int] = []
        for action in actions:
            score = self._action_value_ab(env, action, root_player, self.depth, alpha, beta, cache)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)
            alpha = max(alpha, best_score)
        return self.rng.choice(best_actions)

    def _search_ab(
        self,
        env: EinsteinChessEnv,
        root_player: str,
        depth: int,
        alpha: float,
        beta: float,
        cache: dict[tuple, float],
    ) -> float:
        """使用 alpha 和 beta 跳过不可能影响最终选择的搜索分支。"""
        terminal = _terminal_value(env, root_player)
        if terminal is not None or depth <= 0:
            return heuristic_value(env, root_player)
        key = _cache_key(env, depth)
        if key in cache:
            return cache[key]
        actions = env.legal_actions()
        if not actions:
            return heuristic_value(env, root_player)

        cutoff = False
        if env.current_player == root_player:
            value = -math.inf
            for action in actions:
                value = max(value, self._action_value_ab(env, action, root_player, depth, alpha, beta, cache))
                alpha = max(alpha, value)
                if alpha >= beta:
                    # 当前分支已经不可能优于已知选择，可以停止继续搜索。
                    cutoff = True
                    break
        else:
            value = math.inf
            for action in actions:
                value = min(value, self._action_value_ab(env, action, root_player, depth, alpha, beta, cache))
                beta = min(beta, value)
                if alpha >= beta:
                    cutoff = True
                    break
        if not cutoff:
            cache[key] = value
        return value

    def _action_value_ab(
        self,
        env: EinsteinChessEnv,
        action: int,
        root_player: str,
        depth: int,
        alpha: float,
        beta: float,
        cache: dict[tuple, float],
    ) -> float:
        first_child = _advance_with_dice(env, action, DICE_VALUES[0])
        if first_child.is_terminal():
            return heuristic_value(first_child, root_player)
        total = 0.0
        for dice in DICE_VALUES:
            child = first_child if dice == DICE_VALUES[0] else _advance_with_dice(env, action, dice)
            total += self._search_ab(child, root_player, depth - 1, alpha, beta, cache)
        return total / len(DICE_VALUES)


@dataclass
class _MCTSNode:
    """完整 MCTS 搜索树中的一个节点。"""

    env: EinsteinChessEnv
    root_player: str
    parent: "_MCTSNode | None" = None
    action_from_parent: int | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, "_MCTSNode"] = field(default_factory=dict)
    untried_actions: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.untried_actions = list(self.env.legal_actions())

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visits == 0 else self.value_sum / self.visits


class MCTSAgent(Agent):
    """使用选择、扩展、模拟、回传四阶段的蒙特卡洛树搜索智能体。"""

    name = "mcts"

    def __init__(
        self,
        simulations: int = 128,
        rollout_steps: int = 80,
        exploration: float = 1.4,
        seed: int | None = None,
    ) -> None:
        self.simulations = int(simulations)
        self.rollout_steps = int(rollout_steps)
        self.exploration = float(exploration)
        self.rng = random.Random(seed)

    def select_action(self, env: EinsteinChessEnv) -> int:
        """执行多次树搜索模拟，选择访问次数最多且平均价值高的动作。"""
        actions = env.legal_actions()
        if not actions:
            raise RuntimeError("No legal actions available")
        if len(actions) == 1:
            return actions[0]
        root = _MCTSNode(env.clone(), env.current_player)
        for _ in range(max(1, self.simulations)):
            # MCTS 的一次迭代：选择/扩展节点 -> 模拟 -> 回传结果。
            node = self._select(root)
            value = self._rollout(node.env.clone(), root.root_player)
            self._backpropagate(node, value)
        return max(root.children.items(), key=lambda item: (item[1].visits, item[1].mean_value))[0]

    def _select(self, node: _MCTSNode) -> _MCTSNode:
        while not node.env.is_terminal():
            if node.untried_actions:
                return self._expand(node)
            if not node.children:
                return node
            node = self._best_child(node)
        return node

    def _expand(self, node: _MCTSNode) -> _MCTSNode:
        action = self.rng.choice(node.untried_actions)
        node.untried_actions.remove(action)
        child_env = _advance_with_dice(node.env, action, self.rng.choice(DICE_VALUES))
        child = _MCTSNode(child_env, node.root_player, parent=node, action_from_parent=action)
        node.children[action] = child
        return child

    def _best_child(self, node: _MCTSNode) -> _MCTSNode:
        """使用 UCB 同时考虑子节点平均价值和探索奖励。"""
        sign = 1.0 if node.env.current_player == node.root_player else -1.0
        parent_log = math.log(max(1, node.visits))

        def score(child: _MCTSNode) -> float:
            if child.visits == 0:
                return math.inf
            exploit = sign * child.mean_value
            explore = self.exploration * math.sqrt(parent_log / child.visits)
            return exploit + explore

        return max(node.children.values(), key=score)

    def _rollout(self, env: EinsteinChessEnv, root_player: str) -> float:
        """从当前节点随机模拟若干步，并返回终局或启发式价值。"""
        steps = 0
        while not env.is_terminal() and steps < self.rollout_steps:
            actions = env.legal_actions()
            if not actions:
                break
            action = self.rng.choice(actions)
            env = _advance_with_dice(env, action, self.rng.choice(DICE_VALUES))
            steps += 1
        return normalized_heuristic_value(env, root_player)

    def _backpropagate(self, node: _MCTSNode, value: float) -> None:
        """将一次模拟得到的价值沿父节点链向上传播。"""
        while node is not None:
            node.visits += 1
            node.value_sum += value
            node = node.parent


def _cache_key(env: EinsteinChessEnv, depth: int) -> tuple:
    red = tuple((pid, env.positions[RED][pid]) for pid in PIECE_IDS)
    blue = tuple((pid, env.positions[BLUE][pid]) for pid in PIECE_IDS)
    return (red, blue, env.current_player, env.dice, env.step_count, depth)
