"""Rule-based Einstein chess baseline."""

from __future__ import annotations

import random

from src.env.board import occupant_at
from src.env.env import EinsteinChessEnv
from src.env.rules import DELTAS, GOAL_CORNERS, id_to_action, opponent

from .base import Agent


class RuleBasedAgent(Agent):
    """A small tactical baseline for evaluation and warm-up games."""

    name = "rule_based"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def select_action(self, env: EinsteinChessEnv) -> int:
        scored = [(self._score_action(env, action), action) for action in env.legal_actions()]
        if not scored:
            raise RuntimeError("No legal actions available")
        best = max(score for score, _ in scored)
        best_actions = [action for score, action in scored if score == best]
        return self.rng.choice(best_actions)

    def _score_action(self, env: EinsteinChessEnv, action: int) -> float:
        player = env.current_player
        piece_id, direction = id_to_action(action)
        pos = env.positions[player][piece_id]
        assert pos is not None
        dr, dc = DELTAS[player][direction]
        target = (pos[0] + dr, pos[1] + dc)
        goal = GOAL_CORNERS[player]
        old_dist = abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])
        new_dist = abs(goal[0] - target[0]) + abs(goal[1] - target[1])
        score = float(old_dist - new_dist)
        if target == goal:
            score += 100.0
        captured = occupant_at(env.positions, target)
        if captured is not None and captured.player == opponent(player):
            score += 10.0
        elif captured is not None and captured.player == player:
            score -= 5.0
        return score

