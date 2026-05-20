"""Generate report-ready figures for the Einstein chess project."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yaml
from matplotlib import patches


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "plots" / "report_assets"

RED = "#d95f5f"
BLUE = "#4d79d8"
GOLD = "#f2bd45"
INK = "#243044"
MUTED = "#687386"
BG = "#fbfaf7"
GRID = "#d8dde6"
GREEN = "#67b26f"
PURPLE = "#8b6fd8"


def setup() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titleweight": "bold",
        }
    )


def save(fig: plt.Figure, name: str) -> Path:
    path = OUT_DIR / name
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def draw_board_setup() -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    ax.set_aspect("equal")
    ax.set_xlim(-0.65, 5.95)
    ax.set_ylim(5.7, -0.95)
    ax.axis("off")

    red_start = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0)}
    blue_start = {(4, 4), (4, 3), (4, 2), (3, 4), (3, 3), (2, 4)}
    red_layout = {(0, 0): "1", (0, 1): "2", (0, 2): "3", (1, 0): "4", (1, 1): "5", (2, 0): "6"}
    blue_layout = {(4, 4): "1", (4, 3): "2", (4, 2): "3", (3, 4): "4", (3, 3): "5", (2, 4): "6"}

    for r in range(5):
        for c in range(5):
            color = "#ffffff"
            if (r, c) in red_start:
                color = "#fde3e3"
            if (r, c) in blue_start:
                color = "#e3eaff"
            ax.add_patch(patches.Rectangle((c, r), 1, 1, facecolor=color, edgecolor=GRID, linewidth=1.5))
            if (r, c) in red_layout:
                ax.add_patch(patches.Circle((c + 0.5, r + 0.5), 0.28, facecolor=RED, edgecolor="white", linewidth=2))
                ax.text(c + 0.5, r + 0.5, red_layout[(r, c)], ha="center", va="center", color="white", weight="bold")
            if (r, c) in blue_layout:
                ax.add_patch(patches.Circle((c + 0.5, r + 0.5), 0.28, facecolor=BLUE, edgecolor="white", linewidth=2))
                ax.text(c + 0.5, r + 0.5, blue_layout[(r, c)], ha="center", va="center", color="white", weight="bold")

    ax.text(0, -0.38, "Red start triangle", color=RED, weight="bold", fontsize=12)
    ax.text(3.1, 5.45, "Blue start triangle", color=BLUE, weight="bold", fontsize=12)
    ax.text(4.5, 4.5, "Red goal", ha="center", va="center", color=RED, weight="bold")
    ax.text(0.5, 0.5, "Blue goal", ha="center", va="center", color=BLUE, weight="bold")

    def arrow(start, delta, color, label):
        ax.arrow(
            start[0],
            start[1],
            delta[0],
            delta[1],
            width=0.025,
            head_width=0.16,
            head_length=0.18,
            color=color,
            length_includes_head=True,
            alpha=0.95,
        )
        ax.text(start[0] + delta[0] * 1.18, start[1] + delta[1] * 1.18, label, color=color, weight="bold", fontsize=9)

    arrow((1.35, 1.35), (0.65, 0), RED, "right")
    arrow((1.35, 1.35), (0, 0.65), RED, "down")
    arrow((1.35, 1.35), (0.55, 0.55), RED, "diag")
    arrow((3.65, 3.65), (-0.65, 0), BLUE, "left")
    arrow((3.65, 3.65), (0, -0.65), BLUE, "up")
    arrow((3.65, 3.65), (-0.55, -0.55), BLUE, "diag")

    ax.set_title("Einstein Chess Board, Goals, and Movement Directions", fontsize=17, pad=18)
    ax.text(
        5.25,
        1.0,
        "5 x 5 board\n6 pieces per side\nDice picks the active\npiece candidate",
        fontsize=11,
        color=MUTED,
        va="top",
        bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.05", facecolor="#ffffff", edgecolor=GRID),
    )
    return save(fig, "01_board_setup_rules.png")


def draw_action_space() -> Path:
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.axis("off")
    directions = ["dir 0: forward-right", "dir 1: forward-down", "dir 2: diagonal"]
    for i, label in enumerate(directions):
        ax.text(i + 1.3, 6.35, label, ha="center", va="bottom", weight="bold", color=INK)
    for piece in range(1, 7):
        y = 6 - piece
        ax.text(0.18, y + 0.5, f"piece {piece}", ha="left", va="center", weight="bold")
        for direction in range(3):
            x = direction + 0.75
            action_id = (piece - 1) * 3 + direction
            color = [RED, GOLD, BLUE][direction]
            ax.add_patch(
                patches.FancyBboxPatch(
                    (x, y + 0.08),
                    1.1,
                    0.84,
                    boxstyle="round,pad=0.03,rounding_size=0.03",
                    facecolor=color,
                    edgecolor="white",
                    linewidth=2,
                    alpha=0.94,
                )
            )
            ax.text(x + 0.55, y + 0.54, str(action_id), ha="center", va="center", color="white", weight="bold", fontsize=15)
    ax.text(4.55, 4.95, "Action ID formula", fontsize=14, weight="bold")
    ax.text(
        4.55,
        4.25,
        "action_id = (piece_id - 1) * 3 + direction\n\nThe legal mask keeps only actions matching\nthe dice-derived candidate pieces and board bounds.",
        fontsize=12,
        color=MUTED,
        va="top",
    )
    ax.set_xlim(0, 9.2)
    ax.set_ylim(-0.2, 6.9)
    ax.set_title("Discrete Action Space: 6 Pieces x 3 Directions = 18 Actions", fontsize=16, pad=14)
    return save(fig, "02_action_space_mapping.png")


def draw_state_encoding() -> Path:
    fig, ax = plt.subplots(figsize=(10.2, 6.5))
    ax.axis("off")
    groups = [
        ("0-5", "current player\npiece planes", RED, 6),
        ("6-11", "opponent\npiece planes", BLUE, 6),
        ("12-17", "dice result\none-hot planes", GOLD, 6),
    ]
    base_x = 0.7
    for idx, (channels, label, color, count) in enumerate(groups):
        x = base_x + idx * 3.0
        for k in range(count):
            ax.add_patch(
                patches.Rectangle(
                    (x + k * 0.055, 1.15 + k * 0.055),
                    1.65,
                    1.65,
                    facecolor=color,
                    alpha=0.14 + k * 0.012,
                    edgecolor=color,
                    linewidth=1,
                )
            )
        ax.text(x + 0.86, 3.15, f"channels {channels}", ha="center", weight="bold", fontsize=12, color=color)
        ax.text(x + 0.86, 0.72, label, ha="center", va="top", fontsize=11)
    ax.annotate("", xy=(8.75, 2.05), xytext=(9.45, 2.05), arrowprops=dict(arrowstyle="->", color=INK, lw=2))
    ax.add_patch(
        patches.FancyBboxPatch(
            (9.55, 1.25),
            2.0,
            1.55,
            boxstyle="round,pad=0.05,rounding_size=0.04",
            facecolor="#ffffff",
            edgecolor=GRID,
            linewidth=1.4,
        )
    )
    ax.text(10.55, 2.18, "18 x 5 x 5\nstate tensor", ha="center", va="center", weight="bold", fontsize=13)
    ax.text(
        0.95,
        5.05,
        "Current-player perspective: when blue acts, the board is mirrored so the network always learns movement toward the lower-right corner.",
        fontsize=12,
        color=MUTED,
    )
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0.2, 5.8)
    ax.set_title("State Encoding Used by the Actor-Critic Model", fontsize=16, pad=14)
    return save(fig, "03_state_encoding_planes.png")


def box(ax, xy, wh, text, fc="#ffffff", ec=GRID, color=INK, fontsize=10.5):
    x, y = xy
    w, h = wh
    ax.add_patch(
        patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.035,rounding_size=0.04",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.5,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=color, fontsize=fontsize, weight="bold")


def arrow_between(ax, start, end, color=INK):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=2, color=color, shrinkA=4, shrinkB=4))


def draw_model_architecture() -> Path:
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    ax.axis("off")
    box(ax, (0.2, 2.0), (1.45, 0.95), "Input\n18x5x5", fc="#ffffff")
    box(ax, (2.1, 2.0), (1.55, 0.95), "Conv 3x3\n18 -> 64", fc="#f3f6fb")
    box(ax, (4.1, 2.0), (1.75, 0.95), "2 Residual\nBlocks", fc="#f3f6fb")
    box(ax, (6.35, 2.0), (1.6, 0.95), "Conv 3x3\n64 -> 128", fc="#f3f6fb")
    box(ax, (8.45, 2.0), (1.45, 0.95), "Flatten +\nFC 256", fc="#f3f6fb")
    box(ax, (10.55, 3.05), (1.5, 0.85), "Actor Head\n18 logits", fc="#fde3e3", ec=RED, color=RED)
    box(ax, (10.55, 1.05), (1.5, 0.85), "Critic Head\nV(s)", fc="#e3eaff", ec=BLUE, color=BLUE)
    box(ax, (12.55, 3.05), (1.65, 0.85), "Legal-action\nmask", fc="#fff4da", ec=GOLD, color="#8a640b")
    box(ax, (12.55, 1.05), (1.65, 0.85), "Tanh value\n[-1, 1]", fc="#eaf6ed", ec=GREEN, color=GREEN)
    for x1, x2 in [(1.65, 2.1), (3.65, 4.1), (5.85, 6.35), (7.95, 8.45), (9.9, 10.55)]:
        arrow_between(ax, (x1, 2.48), (x2, 2.48))
    arrow_between(ax, (9.9, 2.48), (10.55, 3.47), RED)
    arrow_between(ax, (9.9, 2.48), (10.55, 1.48), BLUE)
    arrow_between(ax, (12.05, 3.47), (12.55, 3.47), GOLD)
    arrow_between(ax, (12.05, 1.48), (12.55, 1.48), GREEN)
    ax.text(4.96, 0.65, "Shared trunk learns board features; separate heads produce policy and value estimates.", ha="center", color=MUTED, fontsize=12)
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0.35, 4.35)
    ax.set_title("Actor-Critic Network Architecture", fontsize=16, pad=14)
    return save(fig, "04_actor_critic_architecture.png")


def draw_training_pipeline() -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ax.axis("off")
    steps = [
        ((0.6, 3.1), "Self-play\ncurrent policy", RED),
        ((3.0, 3.1), "Game records\nJSON", GOLD),
        ((5.3, 3.1), "Replay buffer\ntransitions", BLUE),
        ((7.65, 3.1), "Batch sample\n128 states", PURPLE),
        ((7.65, 1.0), "PPO-style\nAC update", GREEN),
        ((5.3, 1.0), "Evaluation\nrandom/rule", BLUE),
        ((3.0, 1.0), "Checkpoint\nbest/model", GOLD),
        ((0.6, 1.0), "Plots + report\nartifacts", RED),
    ]
    for (xy, text, color) in steps:
        box(ax, xy, (1.65, 0.9), text, fc="#ffffff", ec=color, color=color, fontsize=10.2)
    centers = [(x + 1.65, y + 0.45) for (x, y), _, _ in steps]
    next_centers = [(x, y + 0.45) for (x, y), _, _ in steps[1:]] + [(steps[0][0][0], steps[0][0][1] + 0.45)]
    for start, end in zip(centers, next_centers):
        arrow_between(ax, start, end, INK)
    ax.text(
        5.4,
        5.15,
        "Training loop: collect experience, optimize policy/value, compare against baselines, and refresh report figures.",
        ha="center",
        color=MUTED,
        fontsize=12,
    )
    ax.set_xlim(0, 10.1)
    ax.set_ylim(0.45, 5.75)
    ax.set_title("End-to-End Training and Reporting Pipeline", fontsize=16, pad=14)
    return save(fig, "05_training_pipeline.png")


def draw_evaluation_summary() -> Path:
    eval_path = ROOT / "logs" / "eval_metrics.csv"
    df = pd.read_csv(eval_path)
    df = df[df["opponent"].isin(["random", "rule_based", "average"])].copy()
    latest_iter = int(df["iteration"].max())
    latest = df[df["iteration"] == latest_iter]
    pivot = df.pivot_table(index="iteration", columns="opponent", values="win_rate", aggfunc="last").sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.2), gridspec_kw={"width_ratios": [1.8, 1]})
    for opponent, color in [("random", RED), ("rule_based", BLUE), ("average", GREEN)]:
        if opponent in pivot:
            ax1.plot(pivot.index, pivot[opponent] * 100, label=opponent.replace("_", " "), color=color, linewidth=2.2)
    ax1.set_title("Win Rate During Evaluation")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel("win rate (%)")
    ax1.grid(True, color=GRID, linewidth=0.8, alpha=0.8)
    ax1.legend(frameon=False, loc="lower right")
    ax1.set_ylim(0, 100)

    order = ["random", "rule_based", "average"]
    latest = latest.set_index("opponent").loc[order]
    colors = [RED, BLUE, GREEN]
    bars = ax2.bar([x.replace("_", "\n") for x in order], latest["win_rate"] * 100, color=colors, alpha=0.88)
    ax2.set_ylim(0, 100)
    ax2.set_title(f"Latest Snapshot: Iteration {latest_iter}")
    ax2.set_ylabel("win rate (%)")
    ax2.grid(axis="y", color=GRID, linewidth=0.8, alpha=0.8)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, f"{bar.get_height():.0f}%", ha="center", weight="bold")
    fig.suptitle("Baseline Evaluation Summary", fontsize=16, fontweight="bold", y=1.03)
    return save(fig, "06_evaluation_summary.png")


def load_game_records(max_records: int = 1600) -> pd.DataFrame:
    record_dir = ROOT / "data" / "game_records"
    files = sorted(record_dir.glob("game_*.json"))
    if not files:
        return pd.DataFrame(columns=["game_id", "first_player", "winner", "win_reason", "total_steps"])
    stride = max(1, len(files) // max_records)
    selected = files[::stride][:max_records]
    rows = []
    for path in selected:
        try:
            with path.open("r", encoding="utf-8") as fh:
                record = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        rows.append(
            {
                "game_id": record.get("game_id"),
                "first_player": record.get("first_player"),
                "winner": record.get("winner"),
                "win_reason": record.get("win_reason"),
                "total_steps": record.get("total_steps", len(record.get("moves", []))),
            }
        )
    return pd.DataFrame(rows)


def draw_game_statistics() -> Path:
    df = load_game_records()
    fig, axs = plt.subplots(1, 3, figsize=(13.4, 4.8))

    axs[0].hist(df["total_steps"].dropna(), bins=18, color=PURPLE, alpha=0.82, edgecolor="white")
    axs[0].set_title("Game Length Distribution")
    axs[0].set_xlabel("total steps")
    axs[0].set_ylabel("games")
    axs[0].grid(axis="y", color=GRID, alpha=0.8)

    reason_counts = df["win_reason"].fillna("unknown").value_counts()
    axs[1].pie(reason_counts.values, labels=reason_counts.index, autopct="%1.0f%%", colors=[GREEN, GOLD, RED, BLUE], startangle=90)
    axs[1].set_title("Win Reasons")

    win_counts = pd.crosstab(df["first_player"].fillna("unknown"), df["winner"].fillna("unknown"))
    win_counts = win_counts.reindex(index=["red", "blue"], fill_value=0)
    win_counts = win_counts.reindex(columns=["red", "blue"], fill_value=0)
    bottom = None
    for col, color in [("red", RED), ("blue", BLUE)]:
        axs[2].bar(win_counts.index, win_counts[col], bottom=bottom, label=f"{col} wins", color=color, alpha=0.86)
        bottom = win_counts[col] if bottom is None else bottom + win_counts[col]
    axs[2].set_title("Winner by First Player")
    axs[2].set_xlabel("first player")
    axs[2].set_ylabel("sampled games")
    axs[2].legend(frameon=False)
    axs[2].grid(axis="y", color=GRID, alpha=0.8)

    fig.suptitle(f"Self-Play Game Record Statistics (sample n={len(df)})", fontsize=16, fontweight="bold", y=1.04)
    return save(fig, "07_game_record_statistics.png")


def draw_hyperparameter_dashboard() -> Path:
    cfg_path = ROOT / "configs" / "default.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    keys = [
        "gamma",
        "learning_rate",
        "min_learning_rate",
        "scheduler",
        "batch_size",
        "replay_buffer_size",
        "entropy_coef",
        "value_coef",
        "ppo_clip_eps",
        "weight_decay",
        "grad_clip_norm",
        "num_iterations",
        "self_play_games_per_iter",
        "eval_games",
        "max_steps",
    ]
    rows = [(key, cfg.get(key)) for key in keys]

    fig, ax = plt.subplots(figsize=(9.2, 7.5))
    ax.axis("off")
    ax.set_title("Training Configuration Snapshot", fontsize=16, pad=16, weight="bold")
    row_h = 0.055
    y0 = 0.86
    ax.add_patch(patches.Rectangle((0.08, y0), 0.84, row_h, facecolor=INK, transform=ax.transAxes))
    ax.text(0.11, y0 + row_h / 2, "parameter", transform=ax.transAxes, va="center", color="white", weight="bold")
    ax.text(0.63, y0 + row_h / 2, "value", transform=ax.transAxes, va="center", color="white", weight="bold")
    for idx, (key, value) in enumerate(rows, start=1):
        y = y0 - idx * row_h
        fc = "#ffffff" if idx % 2 else "#f2f5fa"
        ax.add_patch(patches.Rectangle((0.08, y), 0.84, row_h, facecolor=fc, edgecolor=GRID, linewidth=0.8, transform=ax.transAxes))
        ax.text(0.11, y + row_h / 2, key, transform=ax.transAxes, va="center", color=INK)
        ax.text(0.63, y + row_h / 2, str(value), transform=ax.transAxes, va="center", color=MUTED)
    ax.text(0.08, 0.045, "Source: configs/default.yaml", transform=ax.transAxes, color=MUTED, fontsize=10)
    return save(fig, "08_hyperparameter_dashboard.png")


def main() -> None:
    setup()
    paths = [
        draw_board_setup(),
        draw_action_space(),
        draw_state_encoding(),
        draw_model_architecture(),
        draw_training_pipeline(),
        draw_evaluation_summary(),
        draw_game_statistics(),
        draw_hyperparameter_dashboard(),
    ]
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
