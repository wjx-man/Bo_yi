from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("artifacts") / "figures"


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.log)

    loss_path = args.output_dir / f"{args.prefix}_loss.png"
    top1_path = args.output_dir / f"{args.prefix}_top1.png"
    value_mae_path = args.output_dir / f"{args.prefix}_value_mae.png"

    _plot_lines(
        frame=frame,
        columns=(
            "train_total_loss",
            "val_total_loss",
            "train_policy_loss",
            "val_policy_loss",
            "train_value_loss",
            "val_value_loss",
        ),
        title="Policy-Value Training Loss",
        ylabel="loss",
        output_path=loss_path,
    )
    _plot_lines(
        frame=frame,
        columns=("train_policy_top1", "val_policy_top1"),
        title="Policy Top-1 Accuracy",
        ylabel="accuracy",
        output_path=top1_path,
    )
    _plot_lines(
        frame=frame,
        columns=("train_value_mae", "val_value_mae"),
        title="Value Mean Absolute Error",
        ylabel="MAE",
        output_path=value_mae_path,
    )

    print(f"saved_loss_curve: {loss_path}")
    print(f"saved_top1_curve: {top1_path}")
    print(f"saved_value_mae_curve: {value_mae_path}")


def _plot_lines(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))
    for column in columns:
        if column in frame:
            plt.plot(frame["epoch"], frame[column], label=column)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loss and accuracy curves from a policy-value training CSV log."
    )
    parser.add_argument("log", type=Path, help="Training CSV log path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", type=str, default="training")
    return parser.parse_args()


if __name__ == "__main__":
    main()
