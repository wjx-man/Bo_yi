"""自我对弈训练数据生成模块。

这个文件让两个智能体自己和自己下棋，并把每一步保存成神经网络训练样本：

state  = 当前局面题目
policy = 搜索器对各个动作的推荐程度
value  = 当前行动方最终是赢还是输
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Sequence

import numpy as np

from ..agents.mcts import MCTSAgent
from ..engine import EinsteinGame, Move, PlayerColor
from ..players import PlayerAgent
from .action_codec import move_to_action_id, policy_dict_to_vector
from .state_encoder import encode_state


@dataclass(frozen=True)
class SelfPlaySample:
    """一局棋中的单步样本；value 要等整局结束后才能填写。"""

    state: np.ndarray
    policy: np.ndarray
    player: PlayerColor
    dice_roll: int
    action_id: int
    turn_index: int
    game_id: int


@dataclass(frozen=True)
class SelfPlayDataset:
    """可直接保存为 NPZ 文件的完整自我对弈数据集。"""

    states: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    players: np.ndarray
    dice_rolls: np.ndarray
    action_ids: np.ndarray
    game_ids: np.ndarray
    turn_indices: np.ndarray
    winners: np.ndarray

    def save_npz(self, path: str | Path) -> None:
        """压缩保存全部数组，供训练脚本和分析脚本读取。"""
        np.savez_compressed(
            path,
            states=self.states,
            policies=self.policies,
            values=self.values,
            players=self.players,
            dice_rolls=self.dice_rolls,
            action_ids=self.action_ids,
            game_ids=self.game_ids,
            turn_indices=self.turn_indices,
            winners=self.winners,
        )


def generate_self_play_dataset(
    num_games: int,
    agent_kind: str = "mcts",
    simulations: int = 100,
    exploration: float = 1.4,
    c_puct: float = 1.5,
    max_rollout_steps: int = 160,
    full_neural_mcts_depth: int = 12,
    chance_mode: str = "sample",
    max_turns: int = 300,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    layout: str = "default",
    rng: random.Random | None = None,
    output_path: str | Path | None = None,
) -> SelfPlayDataset:
    """生成多局自我对弈数据，并可选保存到磁盘。"""
    if num_games < 1:
        raise ValueError("num_games must be at least 1.")
    local_rng = rng or random.Random()
    all_samples: list[SelfPlaySample] = []
    all_values: list[float] = []
    all_winners: list[int] = []

    # 每一局都会生成多个单步样本，因此“一局棋”不等于“一个训练样本”。
    for game_id in range(num_games):
        samples, winner = generate_self_play_game(
            game_id=game_id,
            agent_kind=agent_kind,
            simulations=simulations,
            exploration=exploration,
            c_puct=c_puct,
            max_rollout_steps=max_rollout_steps,
            full_neural_mcts_depth=full_neural_mcts_depth,
            chance_mode=chance_mode,
            max_turns=max_turns,
            checkpoint_path=checkpoint_path,
            device=device,
            layout=layout,
            rng=random.Random(local_rng.randrange(2**31)),
        )
        all_samples.extend(samples)
        # 整局结束后，才根据最终胜者为本局所有样本填写 value。
        all_values.extend(_values_for_samples(samples, winner))
        winner_code = _player_code(winner)
        all_winners.extend([winner_code for _ in samples])

    dataset = _build_dataset(all_samples, all_values, all_winners)
    if output_path is not None:
        dataset.save_npz(output_path)
    return dataset


def generate_self_play_game(
    game_id: int,
    simulations: int,
    agent_kind: str = "mcts",
    exploration: float = 1.4,
    c_puct: float = 1.5,
    max_rollout_steps: int = 160,
    full_neural_mcts_depth: int = 12,
    chance_mode: str = "sample",
    max_turns: int = 300,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    layout: str = "default",
    rng: random.Random | None = None,
) -> tuple[list[SelfPlaySample], PlayerColor | None]:
    """运行一局自我对弈，返回该局所有单步样本和最终胜者。"""
    local_rng = rng or random.Random()
    game_rng = random.Random(local_rng.randrange(2**31))
    # layout="default" 用于早期固定开局数据，layout="random" 用于 V3 之后的数据。
    game = EinsteinGame(
        red_layout=_build_layout(PlayerColor.RED, layout, game_rng),
        blue_layout=_build_layout(PlayerColor.BLUE, layout, game_rng),
        rng=game_rng,
    )
    # 红蓝双方使用相同类型的智能体，但使用独立随机数生成器。
    agents = {
        PlayerColor.RED: _build_self_play_agent(
            agent_kind=agent_kind,
            color=PlayerColor.RED,
            simulations=simulations,
            exploration=exploration,
            c_puct=c_puct,
            max_rollout_steps=max_rollout_steps,
            full_neural_mcts_depth=full_neural_mcts_depth,
            chance_mode=chance_mode,
            checkpoint_path=checkpoint_path,
            device=device,
            rng=random.Random(local_rng.randrange(2**31)),
        ),
        PlayerColor.BLUE: _build_self_play_agent(
            agent_kind=agent_kind,
            color=PlayerColor.BLUE,
            simulations=simulations,
            exploration=exploration,
            c_puct=c_puct,
            max_rollout_steps=max_rollout_steps,
            full_neural_mcts_depth=full_neural_mcts_depth,
            chance_mode=chance_mode,
            checkpoint_path=checkpoint_path,
            device=device,
            rng=random.Random(local_rng.randrange(2**31)),
        ),
    }

    samples: list[SelfPlaySample] = []
    turns_played = 0
    while game.winner is None and turns_played < max_turns:
        turns_played += 1
        legal_moves = game.get_legal_moves()
        if not legal_moves:
            game.pass_turn()
            continue

        # 在真正走棋之前保存局面，因为训练目标是学习“看到这个局面时应该怎么走”。
        snapshot = game.snapshot()
        player = snapshot.current_player
        agent = agents[player]
        move = agent.choose_move(snapshot, legal_moves)
        if move not in legal_moves:
            raise ValueError(f"{agent.mode} returned an illegal move.")

        # agent.last_policy 来自搜索访问次数分布，是比单个最终动作更丰富的策略答案。
        samples.append(
            SelfPlaySample(
                state=encode_state(snapshot),
                policy=policy_dict_to_vector(agent.last_policy),
                player=player,
                dice_roll=snapshot.dice_roll,
                action_id=move_to_action_id(move),
                turn_index=snapshot.turn_index,
                game_id=game_id,
            )
        )
        # 保存样本后再执行走法，进入下一回合。
        game.apply_move(move)

    return samples, game.winner


def _build_layout(
    color: PlayerColor,
    layout: str,
    rng: random.Random,
) -> dict[int, tuple[int, int]] | None:
    """根据命令行选择默认开局或随机开局。"""
    if layout == "default":
        return None
    if layout == "random":
        return EinsteinGame.random_layout_for(color, rng=rng)
    raise ValueError("layout must be 'default' or 'random'.")


def _build_self_play_agent(
    agent_kind: str,
    color: PlayerColor,
    simulations: int,
    exploration: float,
    c_puct: float,
    max_rollout_steps: int,
    full_neural_mcts_depth: int,
    chance_mode: str,
    checkpoint_path: str | Path | None,
    device: str,
    rng: random.Random,
) -> PlayerAgent:
    """根据版本阶段创建用于自我对弈的智能体。"""
    if agent_kind == "mcts":
        # V1 数据：基础 MCTS 自我对弈，不需要 checkpoint。
        return MCTSAgent(
            name=f"{color.value}_mcts_self_play",
            simulations=simulations,
            exploration=exploration,
            max_rollout_steps=max_rollout_steps,
            rng=rng,
        )
    if agent_kind == "neural-mcts":
        # V2/V3 数据：已有网络 + Root Neural MCTS。
        if checkpoint_path is None:
            raise ValueError("neural-mcts self-play requires checkpoint_path.")
        from ..agents.neural_mcts import NeuralMCTSAgent

        return NeuralMCTSAgent(
            checkpoint_path=checkpoint_path,
            name=f"{color.value}_neural_mcts_self_play",
            simulations=simulations,
            c_puct=c_puct,
            device=device,
            rng=rng,
        )
    if agent_kind == "full-neural-mcts":
        # V4 Full 数据：V3 网络 + 完整多层 Neural MCTS。
        if checkpoint_path is None:
            raise ValueError("full-neural-mcts self-play requires checkpoint_path.")
        from ..agents.full_neural_mcts import FullNeuralMCTSAgent

        return FullNeuralMCTSAgent(
            checkpoint_path=checkpoint_path,
            name=f"{color.value}_full_neural_mcts_self_play",
            simulations=simulations,
            c_puct=c_puct,
            max_depth=full_neural_mcts_depth,
            chance_mode=chance_mode,
            device=device,
            rng=rng,
        )
    raise ValueError("agent_kind must be 'mcts', 'neural-mcts', or 'full-neural-mcts'.")


def _values_for_samples(
    samples: Sequence[SelfPlaySample], winner: PlayerColor | None
) -> list[float]:
    """从每个样本当时行动方的视角生成最终胜负标签。"""
    if winner is None:
        return [0.0 for _ in samples]
    return [1.0 if sample.player is winner else -1.0 for sample in samples]


def _build_dataset(
    samples: Sequence[SelfPlaySample],
    values: Sequence[float],
    winners: Sequence[int],
) -> SelfPlayDataset:
    """将 Python 样本列表堆叠为形状固定的 NumPy 数组。"""
    if not samples:
        raise ValueError("No self-play samples were generated.")

    return SelfPlayDataset(
        states=np.stack([sample.state for sample in samples]).astype(np.float32),
        policies=np.stack([sample.policy for sample in samples]).astype(np.float32),
        values=np.asarray(values, dtype=np.float32),
        players=np.asarray([_player_code(sample.player) for sample in samples], dtype=np.int8),
        dice_rolls=np.asarray([sample.dice_roll for sample in samples], dtype=np.int8),
        action_ids=np.asarray([sample.action_id for sample in samples], dtype=np.int8),
        game_ids=np.asarray([sample.game_id for sample in samples], dtype=np.int32),
        turn_indices=np.asarray([sample.turn_index for sample in samples], dtype=np.int32),
        winners=np.asarray(winners, dtype=np.int8),
    )


def _player_code(player: PlayerColor | None) -> int:
    """将玩家颜色转换为便于保存和统计的整数编码。"""
    if player is PlayerColor.RED:
        return 1
    if player is PlayerColor.BLUE:
        return -1
    return 0


# 下一步阅读：einstein_chess/training/state_encoder.py
