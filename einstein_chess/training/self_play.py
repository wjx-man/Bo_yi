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
    state: np.ndarray
    policy: np.ndarray
    player: PlayerColor
    dice_roll: int
    action_id: int
    turn_index: int
    game_id: int


@dataclass(frozen=True)
class SelfPlayDataset:
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
    max_turns: int = 300,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    layout: str = "default",
    rng: random.Random | None = None,
    output_path: str | Path | None = None,
) -> SelfPlayDataset:
    if num_games < 1:
        raise ValueError("num_games must be at least 1.")
    local_rng = rng or random.Random()
    all_samples: list[SelfPlaySample] = []
    all_values: list[float] = []
    all_winners: list[int] = []

    for game_id in range(num_games):
        samples, winner = generate_self_play_game(
            game_id=game_id,
            agent_kind=agent_kind,
            simulations=simulations,
            exploration=exploration,
            c_puct=c_puct,
            max_rollout_steps=max_rollout_steps,
            max_turns=max_turns,
            checkpoint_path=checkpoint_path,
            device=device,
            layout=layout,
            rng=random.Random(local_rng.randrange(2**31)),
        )
        all_samples.extend(samples)
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
    max_turns: int = 300,
    checkpoint_path: str | Path | None = None,
    device: str = "cpu",
    layout: str = "default",
    rng: random.Random | None = None,
) -> tuple[list[SelfPlaySample], PlayerColor | None]:
    local_rng = rng or random.Random()
    game_rng = random.Random(local_rng.randrange(2**31))
    game = EinsteinGame(
        red_layout=_build_layout(PlayerColor.RED, layout, game_rng),
        blue_layout=_build_layout(PlayerColor.BLUE, layout, game_rng),
        rng=game_rng,
    )
    agents = {
        PlayerColor.RED: _build_self_play_agent(
            agent_kind=agent_kind,
            color=PlayerColor.RED,
            simulations=simulations,
            exploration=exploration,
            c_puct=c_puct,
            max_rollout_steps=max_rollout_steps,
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

        snapshot = game.snapshot()
        player = snapshot.current_player
        agent = agents[player]
        move = agent.choose_move(snapshot, legal_moves)
        if move not in legal_moves:
            raise ValueError(f"{agent.mode} returned an illegal move.")

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
        game.apply_move(move)

    return samples, game.winner


def _build_layout(
    color: PlayerColor,
    layout: str,
    rng: random.Random,
) -> dict[int, tuple[int, int]] | None:
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
    checkpoint_path: str | Path | None,
    device: str,
    rng: random.Random,
) -> PlayerAgent:
    if agent_kind == "mcts":
        return MCTSAgent(
            name=f"{color.value}_mcts_self_play",
            simulations=simulations,
            exploration=exploration,
            max_rollout_steps=max_rollout_steps,
            rng=rng,
        )
    if agent_kind == "neural-mcts":
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
    raise ValueError("agent_kind must be 'mcts' or 'neural-mcts'.")


def _values_for_samples(
    samples: Sequence[SelfPlaySample], winner: PlayerColor | None
) -> list[float]:
    if winner is None:
        return [0.0 for _ in samples]
    return [1.0 if sample.player is winner else -1.0 for sample in samples]


def _build_dataset(
    samples: Sequence[SelfPlaySample],
    values: Sequence[float],
    winners: Sequence[int],
) -> SelfPlayDataset:
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
    if player is PlayerColor.RED:
        return 1
    if player is PlayerColor.BLUE:
        return -1
    return 0
