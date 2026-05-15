from __future__ import annotations

from pathlib import Path
import random
import unittest

import numpy as np
import torch

from einstein_chess.engine import EinsteinGame
from einstein_chess.training import (
    ACTION_SIZE,
    STATE_CHANNELS,
    action_id_to_move,
    encode_state,
    generate_self_play_dataset,
    legal_action_mask,
    move_to_action_id,
    policy_dict_to_vector,
    PolicyValueNet,
)


class TrainingDataTests(unittest.TestCase):
    def test_action_codec_round_trips_legal_moves(self) -> None:
        game = EinsteinGame(rng=random.Random(7))
        snapshot = game.snapshot()

        for move in snapshot.legal_moves:
            action_id = move_to_action_id(move)
            self.assertGreaterEqual(action_id, 0)
            self.assertLess(action_id, ACTION_SIZE)
            self.assertEqual(action_id_to_move(action_id, snapshot), move)

    def test_legal_action_mask_and_policy_vector_are_18_dimensional(self) -> None:
        game = EinsteinGame(rng=random.Random(7))
        snapshot = game.snapshot()
        move = snapshot.legal_moves[0]

        mask = legal_action_mask(snapshot)
        policy = policy_dict_to_vector({move: 1.0})

        self.assertEqual(mask.shape, (ACTION_SIZE,))
        self.assertEqual(policy.shape, (ACTION_SIZE,))
        self.assertEqual(mask[move_to_action_id(move)], 1.0)
        self.assertAlmostEqual(float(policy.sum()), 1.0)

    def test_state_encoder_returns_15_channel_board_tensor(self) -> None:
        game = EinsteinGame(rng=random.Random(7))
        snapshot = game.snapshot()

        state = encode_state(snapshot)

        self.assertEqual(state.shape, (STATE_CHANNELS, 5, 5))
        self.assertEqual(state.dtype, np.float32)
        self.assertEqual(state[0, 0, 0], 1.0)
        self.assertEqual(state[6, 4, 4], 1.0)
        self.assertTrue(np.all(state[12] == 1.0))
        self.assertTrue(np.all(state[13] == snapshot.dice_roll / 6.0))
        self.assertGreater(float(state[14].sum()), 0.0)

    def test_generate_self_play_dataset_and_save_npz(self) -> None:
        output_path = Path("tests") / "_tmp_self_play_dataset.npz"
        try:
            dataset = generate_self_play_dataset(
                num_games=1,
                simulations=4,
                max_rollout_steps=20,
                max_turns=80,
                rng=random.Random(13),
                output_path=output_path,
            )

            self.assertEqual(dataset.states.ndim, 4)
            self.assertEqual(dataset.states.shape[1:], (STATE_CHANNELS, 5, 5))
            self.assertEqual(dataset.policies.shape, (dataset.states.shape[0], ACTION_SIZE))
            self.assertEqual(dataset.values.shape, (dataset.states.shape[0],))
            self.assertTrue(np.allclose(dataset.policies.sum(axis=1), 1.0))
            self.assertTrue(set(dataset.values.tolist()).issubset({-1.0, 0.0, 1.0}))
            self.assertTrue(output_path.exists())

            with np.load(output_path) as loaded:
                self.assertEqual(loaded["states"].shape, dataset.states.shape)
                self.assertEqual(loaded["policies"].shape, dataset.policies.shape)
                self.assertEqual(loaded["values"].shape, dataset.values.shape)
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_generate_neural_mcts_self_play_dataset(self) -> None:
        checkpoint_path = Path("tests") / "_tmp_neural_self_play.pt"
        try:
            model = PolicyValueNet(hidden_channels=8)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": {"hidden_channels": 8},
                },
                checkpoint_path,
            )

            dataset = generate_self_play_dataset(
                num_games=1,
                agent_kind="neural-mcts",
                checkpoint_path=checkpoint_path,
                simulations=3,
                max_turns=40,
                rng=random.Random(19),
            )

            self.assertEqual(dataset.states.ndim, 4)
            self.assertEqual(dataset.states.shape[1:], (STATE_CHANNELS, 5, 5))
            self.assertEqual(dataset.policies.shape, (dataset.states.shape[0], ACTION_SIZE))
            self.assertTrue(np.allclose(dataset.policies.sum(axis=1), 1.0))
        finally:
            if checkpoint_path.exists():
                checkpoint_path.unlink()

    def test_random_layout_self_play_dataset(self) -> None:
        dataset = generate_self_play_dataset(
            num_games=1,
            simulations=3,
            max_rollout_steps=12,
            max_turns=40,
            layout="random",
            rng=random.Random(23),
        )

        self.assertEqual(dataset.states.ndim, 4)
        self.assertEqual(dataset.states.shape[1:], (STATE_CHANNELS, 5, 5))
        self.assertEqual(dataset.policies.shape, (dataset.states.shape[0], ACTION_SIZE))
        self.assertTrue(np.allclose(dataset.policies.sum(axis=1), 1.0))


if __name__ == "__main__":
    unittest.main()
