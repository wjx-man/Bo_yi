from __future__ import annotations

import csv
from pathlib import Path
import unittest

import pandas as pd

from scripts.plot_training_log import _plot_lines


class ReportingScriptsTests(unittest.TestCase):
    def test_plot_lines_writes_png(self) -> None:
        output_path = Path("tests") / "_tmp_training_plot.png"
        try:
            frame = pd.DataFrame(
                {
                    "epoch": [1, 2, 3],
                    "train_total_loss": [3.0, 2.0, 1.5],
                    "val_total_loss": [3.2, 2.4, 2.1],
                }
            )

            _plot_lines(
                frame=frame,
                columns=("train_total_loss", "val_total_loss"),
                title="test",
                ylabel="loss",
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            self.assertGreater(output_path.stat().st_size, 0)
        finally:
            if output_path.exists():
                output_path.unlink()

    def test_eval_summary_csv_shape(self) -> None:
        path = Path("tests") / "_tmp_eval_summary.csv"
        try:
            rows = [
                {
                    "games": 2,
                    "red": "policy",
                    "blue": "random",
                    "red_wins": 1,
                    "blue_wins": 1,
                }
            ]
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)

            with path.open(newline="", encoding="utf-8") as file:
                loaded = list(csv.DictReader(file))

            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["red"], "policy")
        finally:
            if path.exists():
                path.unlink()


if __name__ == "__main__":
    unittest.main()
