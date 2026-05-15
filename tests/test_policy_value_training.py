from __future__ import annotations

from pathlib import Path
import random
import unittest

import torch

from einstein_chess.training import (
    ACTION_SIZE,
    NPZSelfPlayDataset,
    PolicyValueNet,
    STATE_CHANNELS,
    generate_self_play_dataset,
    policy_value_loss,
)


class PolicyValueTrainingTests(unittest.TestCase):
    def test_dataset_model_and_loss_are_compatible(self) -> None:
        output_path = Path("tests") / "_tmp_policy_value_data.npz"
        try:
            generate_self_play_dataset(
                num_games=1,
                simulations=3,
                max_rollout_steps=12,
                max_turns=40,
                rng=random.Random(17),
                output_path=output_path,
            )
            dataset = NPZSelfPlayDataset(output_path)
            state, policy, value = dataset[0]

            self.assertEqual(tuple(state.shape), (STATE_CHANNELS, 5, 5))
            self.assertEqual(tuple(policy.shape), (ACTION_SIZE,))
            self.assertEqual(tuple(value.shape), ())

            model = PolicyValueNet(hidden_channels=16)
            states = state.unsqueeze(0)
            policies = policy.unsqueeze(0)
            values = value.unsqueeze(0)
            policy_logits, predicted_values = model(states)

            self.assertEqual(tuple(policy_logits.shape), (1, ACTION_SIZE))
            self.assertEqual(tuple(predicted_values.shape), (1,))

            total_loss, policy_loss, value_loss = policy_value_loss(
                policy_logits,
                predicted_values,
                policies,
                values,
            )
            total_loss.backward()

            self.assertTrue(torch.isfinite(total_loss))
            self.assertTrue(torch.isfinite(policy_loss))
            self.assertTrue(torch.isfinite(value_loss))
        finally:
            if output_path.exists():
                output_path.unlink()


if __name__ == "__main__":
    unittest.main()
