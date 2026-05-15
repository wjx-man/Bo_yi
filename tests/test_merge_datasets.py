from __future__ import annotations

from pathlib import Path
import random
import unittest

import numpy as np

from einstein_chess.training import generate_self_play_dataset
from scripts.merge_datasets import merge_npz_datasets


class MergeDatasetsTests(unittest.TestCase):
    def test_merge_npz_datasets_concatenates_samples_and_offsets_game_ids(self) -> None:
        first_path = Path("tests") / "_tmp_merge_first.npz"
        second_path = Path("tests") / "_tmp_merge_second.npz"
        try:
            first = generate_self_play_dataset(
                num_games=1,
                simulations=2,
                max_rollout_steps=10,
                max_turns=30,
                rng=random.Random(1),
                output_path=first_path,
            )
            second = generate_self_play_dataset(
                num_games=1,
                simulations=2,
                max_rollout_steps=10,
                max_turns=30,
                rng=random.Random(2),
                output_path=second_path,
            )

            merged = merge_npz_datasets([first_path, second_path])

            expected_samples = first.states.shape[0] + second.states.shape[0]
            self.assertEqual(merged["states"].shape[0], expected_samples)
            self.assertEqual(merged["policies"].shape[0], expected_samples)
            self.assertEqual(merged["values"].shape[0], expected_samples)
            self.assertGreaterEqual(len(np.unique(merged["game_ids"])), 2)
            self.assertTrue(np.allclose(merged["policies"].sum(axis=1), 1.0))
        finally:
            for path in (first_path, second_path):
                if path.exists():
                    path.unlink()


if __name__ == "__main__":
    unittest.main()
