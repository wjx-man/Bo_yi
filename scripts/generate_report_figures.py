from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path("artifacts") / "figures" / "report"


@dataclass(frozen=True)
class TrainingRun:
    key: str
    label: str
    log_path: Path
    note: str


@dataclass(frozen=True)
class DatasetRun:
    key: str
    label: str
    path: Path
    note: str


TRAINING_RUNS = (
    TrainingRun(
        key="v1",
        label="V1 MCTS data",
        log_path=Path("artifacts/logs/train_500g_100s_gpu.csv"),
        note="Initial policy-value network trained from basic MCTS self-play.",
    ),
    TrainingRun(
        key="v2",
        label="V2 mixed",
        log_path=Path("artifacts/logs/train_v2_mixed.csv"),
        note="Mixed basic MCTS and root neural-MCTS self-play.",
    ),
    TrainingRun(
        key="v3",
        label="V3 random layout",
        log_path=Path("artifacts/logs/train_v3_random.csv"),
        note="Added random-opening root neural-MCTS data.",
    ),
    TrainingRun(
        key="resnet_v4",
        label="ResNet V4 trial",
        log_path=Path("artifacts/logs/train_resnet_v4.csv"),
        note="Residual CNN experiment; rejected after match evaluation.",
    ),
    TrainingRun(
        key="v4_full",
        label="V4 full-MCTS final",
        log_path=Path("artifacts/logs/train_v4_full.csv"),
        note="Final CNN model trained with full neural-MCTS self-play data.",
    ),
)


DATASET_RUNS = (
    DatasetRun(
        key="d0_mcts_500",
        label="D0 MCTS",
        path=Path("artifacts/data/self_play_500g_100s.npz"),
        note="Basic MCTS self-play.",
    ),
    DatasetRun(
        key="d1_root_500",
        label="D1 root NMCTS",
        path=Path("artifacts/data/self_play_neural_mcts_500g_80s.npz"),
        note="Root neural-MCTS self-play.",
    ),
    DatasetRun(
        key="d2_mixed_v2",
        label="D2 mixed V2",
        path=Path("artifacts/data/self_play_mixed_v2_1000g.npz"),
        note="Merged D0 and D1.",
    ),
    DatasetRun(
        key="d3_random_root",
        label="D3 random root",
        path=Path("artifacts/data/self_play_neural_mcts_v2_random_1000g_80s.npz"),
        note="Root neural-MCTS with random layouts.",
    ),
    DatasetRun(
        key="d4_mixed_v3",
        label="D4 mixed V3",
        path=Path("artifacts/data/self_play_mixed_v3_random.npz"),
        note="V3 training data.",
    ),
    DatasetRun(
        key="d5_full_v3",
        label="D5 full NMCTS",
        path=Path("artifacts/data/self_play_full_neural_mcts_v3_random_1000g_80s.npz"),
        note="Full neural-MCTS with V3 network.",
    ),
    DatasetRun(
        key="d6_mixed_v4",
        label="D6 final mix",
        path=Path("artifacts/data/self_play_mixed_v4_full.npz"),
        note="Final V4_full training data.",
    ),
)


EVALUATION_GROUPS = (
    {
        "key": "v3_vs_v2_random",
        "label": "V3 vs V2",
        "winner": "V3",
        "wins": 130,
        "games": 200,
        "note": "Random layouts; root neural-MCTS; red/blue swapped.",
    },
    {
        "key": "full_vs_root_v3",
        "label": "Full MCTS vs Root",
        "winner": "Full MCTS",
        "wins": 117,
        "games": 200,
        "note": "Same V3 network; isolates search improvement.",
    },
    {
        "key": "resnet_v4_vs_v3",
        "label": "ResNet V4 vs V3",
        "winner": "ResNet V4",
        "wins": 65,
        "games": 200,
        "note": "Failed model-structure experiment.",
    },
    {
        "key": "v4_full_vs_v3",
        "label": "V4_full vs V3",
        "winner": "V4_full",
        "wins": 110,
        "games": 200,
        "note": "Final model versus previous best under full neural-MCTS.",
    },
)


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    figure_rows: list[dict[str, str]] = []
    training_summaries = _load_training_summaries()
    dataset_summaries = _load_dataset_summaries()

    for run in TRAINING_RUNS:
        if not run.log_path.is_file():
            continue
        frame = pd.read_csv(run.log_path)
        figure_rows.extend(_plot_training_run(run, frame, args.output_dir))

    figure_rows.append(_plot_combined_training_metric(
        metric="val_total_loss",
        ylabel="validation loss",
        title="Validation Loss Across Model Generations",
        filename="comparison_validation_loss.png",
        output_dir=args.output_dir,
    ))
    figure_rows.append(_plot_combined_training_metric(
        metric="val_policy_top1",
        ylabel="top-1 accuracy",
        title="Validation Policy Top-1 Across Model Generations",
        filename="comparison_validation_top1.png",
        output_dir=args.output_dir,
    ))
    figure_rows.append(_plot_best_training_metrics(training_summaries, args.output_dir))
    figure_rows.append(_plot_dataset_growth(dataset_summaries, args.output_dir))
    figure_rows.append(_plot_dataset_reward_curve(dataset_summaries, args.output_dir))
    figure_rows.append(_plot_evaluation_progression(args.output_dir))
    figure_rows.append(_plot_training_pipeline(args.output_dir))
    figure_rows.append(_plot_state_action_encoding(args.output_dir))
    figure_rows.append(_plot_final_action_distribution(args.output_dir))
    figure_rows.append(_plot_final_dice_distribution(args.output_dir))
    optional_policy_ablation = _plot_policy_vs_full_ablation(args.output_dir)
    if optional_policy_ablation is not None:
        figure_rows.append(optional_policy_ablation)
    optional_baselines = _plot_final_baseline_win_rates(args.output_dir)
    if optional_baselines is not None:
        figure_rows.append(optional_baselines)

    _write_csv(args.output_dir / "training_summary.csv", training_summaries)
    _write_csv(args.output_dir / "dataset_summary.csv", dataset_summaries)
    _write_csv(args.output_dir / "evaluation_summary.csv", _evaluation_rows())
    _write_figure_index(args.output_dir / "FIGURE_INDEX.md", figure_rows)
    _write_additional_eval_commands(args.output_dir / "ADDITIONAL_EVAL_COMMANDS.md")

    print(f"saved_report_figures: {args.output_dir}")
    print(f"saved_figure_index: {args.output_dir / 'FIGURE_INDEX.md'}")
    print(f"figures: {len(figure_rows)}")


def _plot_training_run(
    run: TrainingRun,
    frame: pd.DataFrame,
    output_dir: Path,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.append(_plot_lines(
        frame=frame,
        columns=(
            "train_total_loss",
            "val_total_loss",
            "train_policy_loss",
            "val_policy_loss",
            "train_value_loss",
            "val_value_loss",
        ),
        title=f"{run.label}: Training Loss",
        ylabel="loss",
        output_path=output_dir / f"{run.key}_loss.png",
        section="Training process / loss curve",
        description=f"{run.label} loss curves. {run.note}",
    ))
    rows.append(_plot_lines(
        frame=frame,
        columns=("train_policy_top1", "val_policy_top1"),
        title=f"{run.label}: Policy Top-1",
        ylabel="top-1 accuracy",
        output_path=output_dir / f"{run.key}_policy_top1.png",
        section="Training process / policy accuracy",
        description=f"{run.label} policy-head top-1 accuracy.",
    ))
    rows.append(_plot_lines(
        frame=frame,
        columns=("train_value_mae", "val_value_mae"),
        title=f"{run.label}: Value MAE",
        ylabel="mean absolute error",
        output_path=output_dir / f"{run.key}_value_mae.png",
        section="Training process / value accuracy",
        description=f"{run.label} value-head MAE curve.",
    ))
    return rows


def _plot_lines(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    title: str,
    ylabel: str,
    output_path: Path,
    section: str,
    description: str,
) -> dict[str, str]:
    plt.figure(figsize=(10, 6))
    for column in columns:
        if column in frame:
            plt.plot(frame["epoch"], frame[column], label=column, linewidth=2)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return _figure_row(output_path, section, description)


def _plot_combined_training_metric(
    metric: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: Path,
) -> dict[str, str]:
    output_path = output_dir / filename
    plt.figure(figsize=(10, 6))
    for run in TRAINING_RUNS:
        if not run.log_path.is_file():
            continue
        frame = pd.read_csv(run.log_path)
        if metric in frame:
            plt.plot(frame["epoch"], frame[metric], label=run.label, linewidth=2)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return _figure_row(
        output_path,
        "Training process / cross-generation comparison",
        f"Compares {metric} for all model generations.",
    )


def _plot_best_training_metrics(
    rows: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, str]:
    output_path = output_dir / "comparison_best_training_metrics.png"
    labels = [str(row["model"]) for row in rows]
    best_loss = [float(row["best_val_loss"]) for row in rows]
    best_top1 = [float(row["best_val_top1"]) for row in rows]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax2 = ax1.twinx()
    ax1.bar(x - width / 2, best_loss, width=width, label="best val loss", color="#4C78A8")
    ax2.bar(x + width / 2, best_top1, width=width, label="best val top1", color="#F58518")
    ax1.set_ylabel("best validation loss")
    ax2.set_ylabel("best validation top-1")
    ax1.set_title("Best Validation Metrics by Model Generation")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.grid(True, axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return _figure_row(
        output_path,
        "Training process / hyperparameter results",
        "Best validation loss and policy top-1 for each trained model.",
    )


def _plot_dataset_growth(
    rows: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, str]:
    output_path = output_dir / "self_play_dataset_growth.png"
    labels = [str(row["dataset"]) for row in rows]
    games = [int(row["games"]) for row in rows]
    samples = [int(row["samples"]) for row in rows]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax1.bar(x - width / 2, games, width=width, label="games", color="#54A24B")
    ax2.bar(x + width / 2, samples, width=width, label="samples", color="#E45756")
    ax1.set_ylabel("games")
    ax2.set_ylabel("samples")
    ax1.set_title("Self-Play Data Growth")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=25, ha="right")
    ax1.grid(True, axis="y", alpha=0.25)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return _figure_row(
        output_path,
        "Training data / self-play scale",
        "Number of games and training samples for each self-play dataset.",
    )


def _plot_dataset_reward_curve(
    rows: list[dict[str, object]],
    output_dir: Path,
) -> dict[str, str]:
    output_path = output_dir / "self_play_reward_curve.png"
    labels = [str(row["dataset"]) for row in rows]
    mean_values = [float(row["mean_value"]) for row in rows]
    red_win_rates = [float(row["red_winner_rate"]) for row in rows]

    plt.figure(figsize=(11, 6))
    plt.plot(labels, mean_values, marker="o", linewidth=2.4, label="mean sample value")
    plt.plot(labels, red_win_rates, marker="s", linewidth=2.4, label="red winner rate")
    plt.axhline(0.0, color="black", linewidth=1, alpha=0.4)
    plt.title("Self-Play Reward / Outcome Curve")
    plt.xlabel("dataset stage")
    plt.ylabel("value / rate")
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return _figure_row(
        output_path,
        "Training process / reward curve",
        "Reward-oriented curve from self-play data: mean value and red-side winner rate by data generation stage.",
    )


def _plot_evaluation_progression(output_dir: Path) -> dict[str, str]:
    output_path = output_dir / "evaluation_progression_win_rates.png"
    rows = _evaluation_rows()
    labels = [str(row["experiment"]) for row in rows]
    win_rates = [float(row["win_rate"]) for row in rows]
    colors = ["#4C78A8", "#72B7B2", "#E45756", "#54A24B"]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, win_rates, color=colors[: len(labels)])
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1.4, label="50% baseline")
    plt.ylim(0.0, 0.75)
    plt.ylabel("win rate")
    plt.title("Evaluation Results Across Main Project Stages")
    plt.xticks(rotation=20, ha="right")
    for bar, rate in zip(bars, win_rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return _figure_row(
        output_path,
        "Evaluation / longitudinal comparison",
        "Win-rate comparison for the main model and search improvements.",
    )


def _plot_training_pipeline(output_dir: Path) -> dict[str, str]:
    output_path = output_dir / "training_pipeline.png"
    labels = [
        "Rules\nEngine",
        "Basic\nMCTS",
        "Self-play\nD0",
        "Policy-Value\nV1/V2",
        "Root\nNeural MCTS",
        "Random Layout\nV3 Data",
        "Full\nNeural MCTS",
        "Final\nV4_full",
    ]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 3.6))
    ax.axis("off")
    for index, label in enumerate(labels):
        ax.text(
            x[index],
            0,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", fc="#F7F7F7", ec="#555555"),
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x[index + 1] - 0.42, 0),
                xytext=(x[index] + 0.42, 0),
                arrowprops=dict(arrowstyle="->", lw=1.6, color="#444444"),
            )
    ax.set_xlim(-0.8, len(labels) - 0.2)
    ax.set_ylim(-1.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return _figure_row(
        output_path,
        "Algorithm overview / training pipeline",
        "Overall self-play reinforcement learning pipeline used by the project.",
    )


def _plot_state_action_encoding(output_dir: Path) -> dict[str, str]:
    output_path = output_dir / "state_action_encoding.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    state_labels = [
        "red pieces 1-6",
        "blue pieces 1-6",
        "current player",
        "dice / 6",
        "candidate pieces",
    ]
    state_sizes = [6, 6, 1, 1, 1]
    axes[0].barh(state_labels, state_sizes, color=["#E45756", "#4C78A8", "#72B7B2", "#F58518", "#54A24B"])
    axes[0].set_title("State Encoding: 15 x 5 x 5")
    axes[0].set_xlabel("channels")
    axes[0].invert_yaxis()

    action_labels = ["piece 1", "piece 2", "piece 3", "piece 4", "piece 5", "piece 6"]
    axes[1].bar(action_labels, [3] * 6, color="#4C78A8")
    axes[1].set_title("Action Encoding: 6 pieces x 3 directions = 18")
    axes[1].set_ylabel("directions per piece")
    axes[1].set_ylim(0, 4)
    axes[1].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return _figure_row(
        output_path,
        "State/action modeling",
        "Visual summary of the 15-channel state encoding and 18-action encoding.",
    )


def _plot_final_action_distribution(output_dir: Path) -> dict[str, str]:
    output_path = output_dir / "final_dataset_action_distribution.png"
    dataset_path = Path("artifacts/data/self_play_mixed_v4_full.npz")
    with np.load(dataset_path) as data:
        action_ids = data["action_ids"]
    counts = np.bincount(action_ids.astype(int), minlength=18)
    labels = [f"a{i}" for i in range(18)]

    plt.figure(figsize=(11, 6))
    bars = plt.bar(labels, counts, color="#4C78A8")
    plt.title("Final Training Dataset Action Distribution")
    plt.xlabel("action id")
    plt.ylabel("count")
    plt.grid(True, axis="y", alpha=0.25)
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return _figure_row(
        output_path,
        "Training data / action distribution",
        "Distribution of the 18 encoded actions in the final V4_full training dataset.",
    )


def _plot_final_dice_distribution(output_dir: Path) -> dict[str, str]:
    output_path = output_dir / "final_dataset_dice_distribution.png"
    dataset_path = Path("artifacts/data/self_play_mixed_v4_full.npz")
    with np.load(dataset_path) as data:
        dice_rolls = data["dice_rolls"]
    counts = np.bincount(dice_rolls.astype(int), minlength=7)[1:7]
    labels = [str(i) for i in range(1, 7)]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(labels, counts, color="#F58518")
    plt.title("Final Training Dataset Dice Distribution")
    plt.xlabel("dice roll")
    plt.ylabel("count")
    plt.grid(True, axis="y", alpha=0.25)
    for bar, count in zip(bars, counts):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(count)),
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return _figure_row(
        output_path,
        "Training data / dice distribution",
        "Distribution of dice rolls 1-6 in the final V4_full training dataset.",
    )


def _plot_policy_vs_full_ablation(output_dir: Path) -> dict[str, str] | None:
    full_rate = _paired_target_win_rate(
        target_red_path=Path("artifacts/logs/eval_random_full_v4_red_vs_policy_v4_blue.csv"),
        target_blue_path=Path("artifacts/logs/eval_random_policy_v4_red_vs_full_v4_blue.csv"),
    )
    if full_rate is None:
        return None

    output_path = output_dir / "ablation_policy_vs_full_neural_mcts.png"
    labels = ["Policy only", "Policy + Full MCTS"]
    rates = [1.0 - full_rate, full_rate]
    colors = ["#A0A0A0", "#54A24B"]
    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, rates, color=colors)
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1.3, label="50% baseline")
    plt.ylim(0.0, 1.0)
    plt.ylabel("win rate")
    plt.title("Ablation: Pure Network vs Full Neural MCTS")
    for bar, rate in zip(bars, rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return _figure_row(
        output_path,
        "Ablation / search module",
        "Pure policy-value network compared with the same network guided by full neural-MCTS.",
    )


def _plot_final_baseline_win_rates(output_dir: Path) -> dict[str, str] | None:
    rows = []
    candidates = [
        (
            "Random",
            Path("artifacts/logs/eval_random_full_v4_red_vs_random_blue.csv"),
            Path("artifacts/logs/eval_random_random_red_vs_full_v4_blue.csv"),
        ),
        (
            "MCTS50",
            Path("artifacts/logs/eval_random_full_v4_red_vs_mcts50_blue.csv"),
            Path("artifacts/logs/eval_random_mcts50_red_vs_full_v4_blue.csv"),
        ),
        (
            "Root NMCTS",
            Path("artifacts/logs/eval_random_full_v4_red_vs_root_v4_blue.csv"),
            Path("artifacts/logs/eval_random_root_v4_red_vs_full_v4_blue.csv"),
        ),
        (
            "V3 Full NMCTS",
            Path("artifacts/logs/eval_random_full_v4_loss_red_vs_full_v3_blue.csv"),
            Path("artifacts/logs/eval_random_full_v3_red_vs_full_v4_loss_blue.csv"),
        ),
    ]
    for label, target_red_path, target_blue_path in candidates:
        rate = _paired_target_win_rate(target_red_path, target_blue_path)
        if rate is not None:
            rows.append((label, rate))
    if not rows:
        return None

    output_path = output_dir / "final_agent_baseline_win_rates.png"
    labels = [row[0] for row in rows]
    rates = [row[1] for row in rows]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, rates, color="#4C78A8")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1.3, label="50% baseline")
    plt.ylim(0.0, 1.0)
    plt.ylabel("V4_full win rate")
    plt.title("Final Agent Win Rates Against Baselines")
    plt.xticks(rotation=15, ha="right")
    for bar, rate in zip(bars, rates):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{rate:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    plt.legend()
    plt.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return _figure_row(
        output_path,
        "Evaluation / final baselines",
        "Final V4_full full-neural-MCTS agent win rates against available baselines.",
    )


def _load_training_summaries() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in TRAINING_RUNS:
        if not run.log_path.is_file():
            continue
        frame = pd.read_csv(run.log_path)
        best_loss_index = frame["val_total_loss"].idxmin()
        best_top1_index = frame["val_policy_top1"].idxmax()
        rows.append(
            {
                "model": run.label,
                "key": run.key,
                "log": str(run.log_path),
                "epochs_recorded": int(frame["epoch"].max()),
                "best_val_loss": float(frame.loc[best_loss_index, "val_total_loss"]),
                "best_val_loss_epoch": int(frame.loc[best_loss_index, "epoch"]),
                "best_val_top1": float(frame.loc[best_top1_index, "val_policy_top1"]),
                "best_val_top1_epoch": int(frame.loc[best_top1_index, "epoch"]),
                "final_train_loss": float(frame.iloc[-1]["train_total_loss"]),
                "final_val_loss": float(frame.iloc[-1]["val_total_loss"]),
                "note": run.note,
            }
        )
    return rows


def _load_dataset_summaries() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for run in DATASET_RUNS:
        if not run.path.is_file():
            continue
        with np.load(run.path) as data:
            values = data["values"]
            winners = data["winners"]
            rows.append(
                {
                    "dataset": run.label,
                    "key": run.key,
                    "path": str(run.path),
                    "games": int(np.max(data["game_ids"])) + 1,
                    "samples": int(data["states"].shape[0]),
                    "mean_value": float(np.mean(values)),
                    "positive_value_rate": float(np.mean(values > 0)),
                    "negative_value_rate": float(np.mean(values < 0)),
                    "red_winner_rate": float(np.mean(winners == 1)),
                    "blue_winner_rate": float(np.mean(winners == -1)),
                    "policy_entropy_mean": float(_policy_entropy(data["policies"]).mean()),
                    "note": run.note,
                }
            )
    return rows


def _evaluation_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in EVALUATION_GROUPS:
        rows.append(
            {
                "experiment": item["label"],
                "winner_label": item["winner"],
                "wins": item["wins"],
                "games": item["games"],
                "win_rate": float(item["wins"] / item["games"]),
                "note": item["note"],
            }
        )
    return rows


def _paired_target_win_rate(
    target_red_path: Path,
    target_blue_path: Path,
) -> float | None:
    if not target_red_path.is_file() or not target_blue_path.is_file():
        return None

    red_summary = _read_eval_summary(target_red_path)
    blue_summary = _read_eval_summary(target_blue_path)
    target_wins = int(red_summary["red_wins"]) + int(blue_summary["blue_wins"])
    total_games = int(red_summary["games"]) + int(blue_summary["games"])
    if total_games <= 0:
        return None
    return target_wins / total_games


def _read_eval_summary(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        row = next(reader)
    return row


def _policy_entropy(policies: np.ndarray) -> np.ndarray:
    clipped = np.clip(policies, 1e-8, 1.0)
    return -np.sum(policies * np.log(clipped), axis=1)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_figure_index(path: Path, rows: list[dict[str, str]]) -> None:
    lines = [
        "# Report Figure Index",
        "",
        "This directory contains generated figures for the final design report.",
        "",
        "| Figure | Report section | Description |",
        "|---|---|---|",
    ]
    for row in rows:
        lines.append(f"| `{row['figure']}` | {row['section']} | {row['description']} |")
    lines.append("")
    lines.append("Recommended final model: `policy_value_v4_full_best_loss.pt`.")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_additional_eval_commands(path: Path) -> None:
    lines = [
        "# Additional Evaluation Commands",
        "",
        "Run these commands on the GPU machine, then run `python scripts/generate_report_figures.py` again.",
        "",
        "## 1. Pure Network vs Full Neural-MCTS Ablation",
        "",
        "```bash",
        "python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue policy --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --output artifacts/logs/eval_random_full_v4_red_vs_policy_v4_blue.csv",
        "```",
        "",
        "```bash",
        "python scripts/evaluate_agents.py --layout random --games 100 --red policy --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --output artifacts/logs/eval_random_policy_v4_red_vs_full_v4_blue.csv",
        "```",
        "",
        "## 2. Final Agent vs Random",
        "",
        "```bash",
        "python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue random --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --output artifacts/logs/eval_random_full_v4_red_vs_random_blue.csv",
        "```",
        "",
        "```bash",
        "python scripts/evaluate_agents.py --layout random --games 100 --red random --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --output artifacts/logs/eval_random_random_red_vs_full_v4_blue.csv",
        "```",
        "",
        "## 3. Final Agent vs MCTS50",
        "",
        "```bash",
        "python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --mcts-simulations 50 --output artifacts/logs/eval_random_full_v4_red_vs_mcts50_blue.csv",
        "```",
        "",
        "```bash",
        "python scripts/evaluate_agents.py --layout random --games 100 --red mcts --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --mcts-simulations 50 --output artifacts/logs/eval_random_mcts50_red_vs_full_v4_blue.csv",
        "```",
        "",
        "## 4. Final Full Neural-MCTS vs Root Neural-MCTS",
        "",
        "```bash",
        "python scripts/evaluate_agents.py --layout random --games 100 --red full-neural-mcts --blue neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --neural-mcts-simulations 80 --output artifacts/logs/eval_random_full_v4_red_vs_root_v4_blue.csv",
        "```",
        "",
        "```bash",
        "python scripts/evaluate_agents.py --layout random --games 100 --red neural-mcts --blue full-neural-mcts --checkpoint artifacts/checkpoints/policy_value_v4_full_best_loss.pt --device cuda --full-neural-mcts-simulations 80 --full-neural-mcts-depth 12 --chance-mode sample --neural-mcts-simulations 80 --output artifacts/logs/eval_random_root_v4_red_vs_full_v4_blue.csv",
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _figure_row(path: Path, section: str, description: str) -> dict[str, str]:
    return {
        "figure": path.name,
        "section": section,
        "description": description,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate report-ready figures from training logs, self-play data, and evaluation summaries."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    main()
