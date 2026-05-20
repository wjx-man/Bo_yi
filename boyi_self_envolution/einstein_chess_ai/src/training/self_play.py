"""Self-play data generation and game-record saving."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.actor_critic_agent import ActorCriticAgent
from src.env.env import EinsteinChessEnv
from src.utils.serialization import save_json

from .replay_buffer import TransitionReplayBuffer


class SelfPlayRunner:
    """Generate self-play transitions with the current policy."""

    def __init__(
        self,
        agent: ActorCriticAgent,
        replay_buffer: TransitionReplayBuffer,
        game_record_dir: str | Path = "data/game_records",
        max_steps: int = 300,
    ) -> None:
        self.agent = agent
        self.replay_buffer = replay_buffer
        self.game_record_dir = Path(game_record_dir)
        self.max_steps = max_steps
        self.game_record_dir.mkdir(parents=True, exist_ok=True)

    def play_game(self, game_id: int, seed: int | None = None, save_record: bool = True) -> dict[str, Any]:
        """Run one self-play game and return its record."""
        env = EinsteinChessEnv(seed=seed, max_steps=self.max_steps)
        obs = env.reset(seed=seed)
        record: dict[str, Any] = {
            "game_id": game_id,
            "first_player": env.current_player,
            "initial_board": obs["board"],
            "moves": [],
            "winner": None,
            "win_reason": None,
            "total_steps": 0,
        }

        done = False
        step_id = 0
        while not done and step_id < self.max_steps:
            state = env.get_encoded_state()
            legal_mask = env.legal_action_mask()
            board_before = obs["board"]
            diagnostics = self.agent.evaluate_policy(env)
            next_obs, reward, done, info = env.step(diagnostics.action)
            next_state = env.get_encoded_state()
            next_mask = env.legal_action_mask()
            transition = {
                "state": state,
                "legal_mask": legal_mask,
                "action": diagnostics.action,
                "reward": reward,
                "next_state": next_state,
                "next_legal_mask": next_mask,
                "done": done,
                "old_log_prob": diagnostics.log_prob,
                "old_value": diagnostics.value,
                "dice": info["dice"],
                "player": info["current_player"],
                "game_id": game_id,
                "step_id": step_id,
            }
            self.replay_buffer.add(transition)
            move = {
                "step": step_id,
                "player": info["current_player"],
                "dice": info["dice"],
                "candidate_pieces": info["candidate_pieces"],
                "legal_actions": info["legal_actions"],
                "action": diagnostics.action,
                "piece_id": info["piece_id"],
                "direction": info["direction"],
                "board_before": board_before,
                "board_after": next_obs["board"],
                "captured_piece": info["captured_piece"],
                "reward": reward,
                "value": diagnostics.value,
                "policy": diagnostics.probabilities,
                "red_time_left": info["red_time_left"],
                "blue_time_left": info["blue_time_left"],
            }
            record["moves"].append(move)
            obs = next_obs
            step_id += 1

        record["winner"] = env.winner
        record["win_reason"] = env.win_reason
        record["total_steps"] = step_id
        if save_record:
            save_json(record, self.game_record_dir / f"game_{game_id:06d}.json")
        return record

