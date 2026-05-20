"""Base agent interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.env.env import EinsteinChessEnv


class Agent(ABC):
    """Base class for action-selecting agents."""

    name = "agent"

    @abstractmethod
    def select_action(self, env: EinsteinChessEnv) -> int:
        """Return a legal action id."""

    def observe(self, transition: dict[str, Any]) -> None:
        """Optional learning hook."""


class HumanAgent(Agent):
    """Placeholder agent used by GUI controllers for human clicks."""

    name = "human"

    def __init__(self) -> None:
        self.pending_action: int | None = None

    def set_action(self, action: int) -> None:
        """Provide an action selected from the GUI."""
        self.pending_action = int(action)

    def select_action(self, env: EinsteinChessEnv) -> int:
        """Return the pending GUI action."""
        if self.pending_action is None:
            raise RuntimeError("Human action has not been provided yet")
        action = self.pending_action
        self.pending_action = None
        if action not in env.legal_actions():
            raise ValueError("Human selected an illegal action")
        return action

