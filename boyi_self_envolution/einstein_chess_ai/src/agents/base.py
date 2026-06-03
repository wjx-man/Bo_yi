"""所有智能体共享的接口定义。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.env.env import EinsteinChessEnv


class Agent(ABC):
    """动作选择智能体的抽象基类。

    GUI、评估器和训练代码只依赖这个接口，因此可以自由替换人类、规则、
    搜索算法或神经网络智能体。
    """

    name = "agent"

    @abstractmethod
    def select_action(self, env: EinsteinChessEnv) -> int:
        """根据当前环境返回一个合法动作 ID。"""

    def observe(self, transition: dict[str, Any]) -> None:
        """可选的学习回调，当前实现主要由训练器集中更新模型。"""


class HumanAgent(Agent):
    """GUI 中的人类玩家占位智能体。"""

    name = "human"

    def __init__(self) -> None:
        self.pending_action: int | None = None

    def set_action(self, action: int) -> None:
        """保存用户在 GUI 中点击得到的动作。"""
        self.pending_action = int(action)

    def select_action(self, env: EinsteinChessEnv) -> int:
        """取出待执行动作，并再次检查它是否合法。"""
        if self.pending_action is None:
            raise RuntimeError("Human action has not been provided yet")
        action = self.pending_action
        self.pending_action = None
        if action not in env.legal_actions():
            raise ValueError("Human selected an illegal action")
        return action

