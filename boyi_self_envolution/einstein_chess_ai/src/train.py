"""读取配置并启动自我博弈训练的命令行入口。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from src.training.trainer import SelfPlayTrainer


def load_config(path: str | Path | None) -> dict[str, Any]:
    """读取 YAML 配置；未提供路径时返回空配置并使用代码默认值。"""
    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def build_parser() -> argparse.ArgumentParser:
    """定义可从命令行覆盖的训练超参数。"""
    parser = argparse.ArgumentParser(description="Train Einstein chess Actor-Critic agent.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--num_iterations", type=int)
    parser.add_argument("--self_play_games_per_iter", type=int)
    parser.add_argument("--train_steps_per_iter", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--min_lr", type=float)
    parser.add_argument("--scheduler", choices=["none", "cosine", "plateau"])
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--entropy_coef", type=float)
    parser.add_argument("--value_coef", type=float)
    parser.add_argument("--replay_buffer_size", type=int)
    parser.add_argument("--save_interval", type=int)
    parser.add_argument("--eval_interval", type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    # 命令行参数优先级高于 YAML，便于临时修改某个训练参数。
    overrides = vars(args)
    lr = overrides.pop("lr", None)
    if lr is not None:
        config["learning_rate"] = lr
    min_lr = overrides.pop("min_lr", None)
    if min_lr is not None:
        config["min_learning_rate"] = min_lr
    overrides.pop("config", None)
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    # 使用源码所在目录作为项目根目录，确保日志和检查点保存位置稳定。
    config["project_root"] = str(Path(__file__).resolve().parents[1])
    SelfPlayTrainer(config).train()


if __name__ == "__main__":
    main()
