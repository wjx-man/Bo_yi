"""Monte Carlo Tree Search style baseline agent."""

from __future__ import annotations

import math
import random

from src.agents.base import Agent
from src.agents.random_agent import RandomAgent
from src.agents.rule_based import RuleBasedAgent
from src.env.env import EinsteinChessEnv
from src.env.rules import BLUE, GOAL_CORNERS, PIECE_IDS, RED, opponent


class MCTSAgent(Agent):
    """A lightweight root-parallel MCTS baseline.

    Each move runs `simulations` UCB-guided playouts from the root and limits
    each playout to `max_depth` steps. Rollouts use the rule-based policy with
    a small amount of random exploration.
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
        actions = env.legal_actions()
        if not actions:
            raise RuntimeError("No legal actions available")
        if len(actions) == 1:
            return actions[0]

        root_player = env.current_player
        visits = {action: 0 for action in actions}
        totals = {action: 0.0 for action in actions}

        for _ in range(max(1, self.simulations)):
            action = self._select_root_action(actions, visits, totals)
            sim_env = env.clone()
            sim_env.rng.seed(self.rng.randint(0, 2**31 - 1))
            _, _, done, _ = sim_env.step(action)
            value = self._evaluate_rollout(sim_env, root_player, remaining_depth=self.max_depth - 1, done=done)
            visits[action] += 1
            totals[action] += value

        return max(actions, key=lambda action: totals[action] / max(1, visits[action]))

    def _select_root_action(self, actions: list[int], visits: dict[int, int], totals: dict[int, float]) -> int:
        unvisited = [action for action in actions if visits[action] == 0]
        if unvisited:
            return self.rng.choice(unvisited)
        total_visits = sum(visits.values())
        return max(
            actions,
            key=lambda action: totals[action] / visits[action]
            + self.exploration * math.sqrt(math.log(total_visits + 1) / visits[action]),
        )

    def _evaluate_rollout(self, env: EinsteinChessEnv, root_player: str, remaining_depth: int, done: bool) -> float:
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

