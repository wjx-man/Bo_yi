"""Random legal-move baseline."""

from __future__ import annotations

import random

from src.env.env import EinsteinChessEnv

from .base import Agent


class RandomAgent(Agent):
    """Select a uniformly random legal action."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, env: EinsteinChessEnv) -> int:
        actions = env.legal_actions()
        if not actions:
            raise RuntimeError("No legal actions available")
        return self.rng.choice(actions)

