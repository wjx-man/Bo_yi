"""Training curve plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_metric(csv_path: str | Path, x_col: str, y_col: str, output_path: str | Path, title: str) -> None:
    """Plot one metric from a CSV file if data exists."""
    csv_path = Path(csv_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        return
    if df.empty or y_col not in df:
        return
    plt.figure(figsize=(7, 4))
    plt.plot(df[x_col], df[y_col], marker="o", linewidth=1.5)
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _read_csv_or_none(csv_path: str | Path) -> pd.DataFrame | None:
    """Read a CSV file, returning None when it is absent or malformed."""
    csv_path = Path(csv_path)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(csv_path)
    except pd.errors.ParserError:
        return None
    return None if df.empty else df


def plot_eval_win_rates(csv_path: str | Path, output_path: str | Path) -> None:
    """Plot separate baseline win-rate curves plus the average curve."""
    df = _read_csv_or_none(csv_path)
    if df is None or not {"iteration", "opponent", "win_rate"}.issubset(df.columns):
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pivot = df.pivot_table(index="iteration", columns="opponent", values="win_rate", aggfunc="last")
    if pivot.empty:
        return
    plt.figure(figsize=(8, 4.5))
    for column in pivot.columns:
        is_average = str(column).lower() == "average"
        plt.plot(
            pivot.index,
            pivot[column] * 100.0,
            marker="o",
            linewidth=2.6 if is_average else 1.6,
            label=str(column),
        )
    plt.title("Win Rate vs Baselines")
    plt.xlabel("iteration")
    plt.ylabel("win rate (%)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_latest_eval_bar(csv_path: str | Path, output_path: str | Path) -> None:
    """Plot the latest evaluation as a baseline comparison bar chart."""
    df = _read_csv_or_none(csv_path)
    if df is None or not {"iteration", "opponent", "win_rate"}.issubset(df.columns):
        return
    latest_iteration = df["iteration"].max()
    latest = df[df["iteration"] == latest_iteration].copy()
    latest = latest.sort_values("opponent")
    if latest.empty:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = ["#455a64" if opponent == "average" else "#1976d2" for opponent in latest["opponent"]]
    plt.figure(figsize=(7, 4.2))
    plt.bar(latest["opponent"], latest["win_rate"] * 100.0, color=colors)
    plt.title(f"Latest Eval Win Rate (iter {latest_iteration})")
    plt.xlabel("baseline")
    plt.ylabel("win rate (%)")
    plt.ylim(0, 100)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_all_curves(log_dir: str | Path, plot_dir: str | Path) -> None:
    """Generate the required training/evaluation curves."""
    log_dir = Path(log_dir)
    plot_dir = Path(plot_dir)
    plot_metric(log_dir / "train_metrics.csv", "iteration", "avg_reward", plot_dir / "reward_curve.png", "Average Reward")
    plot_metric(log_dir / "train_metrics.csv", "iteration", "policy_loss", plot_dir / "policy_loss_curve.png", "Policy Loss")
    plot_metric(log_dir / "train_metrics.csv", "iteration", "value_loss", plot_dir / "value_loss_curve.png", "Value Loss")
    plot_metric(log_dir / "train_metrics.csv", "iteration", "entropy", plot_dir / "entropy_curve.png", "Entropy")
    plot_eval_win_rates(log_dir / "eval_metrics.csv", plot_dir / "win_rate_curve.png")
    plot_latest_eval_bar(log_dir / "eval_metrics.csv", plot_dir / "baseline_win_rate_bar.png")
