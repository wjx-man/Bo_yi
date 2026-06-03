"""自我博弈数据生成和棋局记录保存。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agents.actor_critic_agent import ActorCriticAgent
from src.env.env import EinsteinChessEnv
from src.utils.serialization import save_json

from .replay_buffer import TransitionReplayBuffer


class SelfPlayRunner:
    """让当前策略与自己对弈，并生成强化学习所需的 transition。"""

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
        """运行一局自我博弈，返回完整棋局记录。"""
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
            # 在执行动作前保存状态和合法动作掩码，它们是训练输入。
            state = env.get_encoded_state()
            legal_mask = env.legal_action_mask()
            board_before = obs["board"]

            # 同一个 agent 同时控制红蓝双方；环境会按当前玩家视角编码状态。
            diagnostics = self.agent.evaluate_policy(env)
            next_obs, reward, done, info = env.step(diagnostics.action)

            # env.step 后环境已经切换到下一位玩家，因此 next_state 是对手视角。
            next_state = env.get_encoded_state()
            next_mask = env.legal_action_mask()

            # 一条 transition 描述“在状态 state 执行动作 action 后发生了什么”。
            # old_log_prob 用于训练时限制新策略相对旧策略的变化幅度。
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

            # move 是给人阅读和回放使用的详细记录，不直接参与梯度更新。
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

