"""随机合法走法智能体，用作最基础的棋力基准。"""

from __future__ import annotations

import random

from src.env.env import EinsteinChessEnv

from .base import Agent


class RandomAgent(Agent):
    """在所有合法动作中等概率随机选择一个。"""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, env: EinsteinChessEnv) -> int:
        # 随机智能体不评价局面，只用于验证其他智能体是否优于随机水平。
        actions = env.legal_actions()
        if not actions:
            raise RuntimeError("No legal actions available")
        return self.rng.choice(actions)

