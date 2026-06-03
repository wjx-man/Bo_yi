"""模型棋力评估工具。"""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

from src.agents.base import Agent
from src.env.env import EinsteinChessEnv
from src.env.rules import BLUE, RED, opponent


@dataclass
class EvaluationResult:
    """一组对局评估结果的汇总指标。"""

    iteration: int
    opponent: str
    num_games: int
    win_rate: float
    avg_steps: float
    avg_reward: float
    reach_corner_wins: int
    capture_all_wins: int
    timeout_wins: int

    def as_dict(self) -> dict:
        return self.__dict__.copy()


class Evaluator:
    """运行智能体对战，并统计胜率、平均步数和获胜原因。"""

    def __init__(self, max_steps: int = 300) -> None:
        self.max_steps = max_steps

    def evaluate(
        self,
        agent: Agent,
        opponent_agent: Agent,
        num_games: int = 20,
        iteration: int = 0,
        seed: int | None = None,
    ) -> EvaluationResult:
        """让待评估智能体交替执红、执蓝，降低先手和颜色带来的偏差。"""
        wins = 0
        total_steps = 0
        total_reward = 0.0
        reasons: Counter[str] = Counter()
        for game_idx in range(num_games):
            # 每局使用不同但可复现的随机种子，减少单次骰子运气的影响。
            env = EinsteinChessEnv(seed=None if seed is None else seed + game_idx, max_steps=self.max_steps)
            env.reset(first_player=RED if game_idx % 2 == 0 else BLUE, seed=None if seed is None else seed + game_idx)
            controlled_player = RED if game_idx % 2 == 0 else BLUE
            agents = {controlled_player: agent, opponent(controlled_player): opponent_agent}
            done = False
            game_reward = 0.0
            while not done:
                actor = agents[env.current_player]
                action_player = env.current_player
                action = actor.select_action(env)
                _, reward, done, _ = env.step(action)
                # 奖励统一换算成待评估智能体的视角，方便跨颜色统计。
                if action_player == controlled_player:
                    game_reward += reward
                else:
                    game_reward -= reward
            wins += int(env.winner == controlled_player)
            total_steps += env.step_count
            total_reward += game_reward
            reasons[env.win_reason or "unknown"] += int(env.winner == controlled_player)
        return EvaluationResult(
            iteration=iteration,
            opponent=getattr(opponent_agent, "name", type(opponent_agent).__name__),
            num_games=num_games,
            win_rate=wins / max(1, num_games),
            avg_steps=total_steps / max(1, num_games),
            avg_reward=total_reward / max(1, num_games),
            reach_corner_wins=reasons["reach_corner"],
            capture_all_wins=reasons["capture_all"],
            timeout_wins=reasons["timeout"],
        )

    def best_of_seven(self, red_agent: Agent, blue_agent: Agent, seed: int | None = None) -> dict:
        """按指定先手顺序运行七局四胜制比赛。"""
        first_players = [RED, BLUE, BLUE, RED, RED, BLUE, BLUE]
        wins = {RED: 0, BLUE: 0}
        games = []
        start = time.perf_counter()
        for idx, first in enumerate(first_players, start=1):
            env = EinsteinChessEnv(seed=None if seed is None else seed + idx, max_steps=self.max_steps)
            env.reset(first_player=first, seed=None if seed is None else seed + idx)
            agents = {RED: red_agent, BLUE: blue_agent}
            done = False
            while not done:
                action = agents[env.current_player].select_action(env)
                _, _, done, _ = env.step(action)
            wins[env.winner] += 1
            games.append({"game": idx, "first_player": first, "winner": env.winner, "reason": env.win_reason})
            if wins[RED] == 4 or wins[BLUE] == 4:
                break
        return {
            "winner": RED if wins[RED] > wins[BLUE] else BLUE,
            "wins": wins,
            "games": games,
            "elapsed_seconds": time.perf_counter() - start,
        }

