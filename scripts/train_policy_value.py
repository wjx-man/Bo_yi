from __future__ import annotations

import argparse
import csv
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from einstein_chess.training import NPZSelfPlayDataset, PolicyValueNet, policy_value_loss


DEFAULT_CHECKPOINT = Path("artifacts") / "checkpoints" / "policy_value_latest.pt"
DEFAULT_BEST_VAL_LOSS_CHECKPOINT = (
    Path("artifacts") / "checkpoints" / "policy_value_best_val_loss.pt"
)
DEFAULT_BEST_VAL_TOP1_CHECKPOINT = (
    Path("artifacts") / "checkpoints" / "policy_value_best_val_top1.pt"
)
DEFAULT_LOG = Path("artifacts") / "logs" / "train_log.csv"


def main() -> None:
    args = _parse_args()
    _set_seed(args.seed)

    device = torch.device(args.device)
    dataset = NPZSelfPlayDataset(args.dataset)
    train_dataset, val_dataset = _split_dataset(dataset, args.val_fraction, args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = PolicyValueNet(hidden_channels=args.hidden_channels).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.best_val_loss_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.best_val_top1_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | int | str]] = []
    best_val_loss = float("inf")
    best_val_top1 = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            value_loss_weight=args.value_loss_weight,
        )
        val_metrics = _evaluate(
            model=model,
            loader=val_loader,
            device=device,
            value_loss_weight=args.value_loss_weight,
        )

        row = {
            "epoch": epoch,
            "train_total_loss": train_metrics["total_loss"],
            "train_policy_loss": train_metrics["policy_loss"],
            "train_value_loss": train_metrics["value_loss"],
            "train_policy_top1": train_metrics["policy_top1"],
            "train_value_mae": train_metrics["value_mae"],
            "val_total_loss": val_metrics["total_loss"],
            "val_policy_loss": val_metrics["policy_loss"],
            "val_value_loss": val_metrics["value_loss"],
            "val_policy_top1": val_metrics["policy_top1"],
            "val_value_mae": val_metrics["value_mae"],
        }
        rows.append(row)
        if row["val_total_loss"] < best_val_loss:
            best_val_loss = float(row["val_total_loss"])
            _save_checkpoint(
                path=args.best_val_loss_checkpoint,
                model=model,
                args=args,
                rows=rows,
                selection_metric="val_total_loss",
            )
        if row["val_policy_top1"] > best_val_top1:
            best_val_top1 = float(row["val_policy_top1"])
            _save_checkpoint(
                path=args.best_val_top1_checkpoint,
                model=model,
                args=args,
                rows=rows,
                selection_metric="val_policy_top1",
            )
        print(
            "epoch "
            f"{epoch:03d} | "
            f"train_loss={row['train_total_loss']:.4f} "
            f"policy={row['train_policy_loss']:.4f} "
            f"value={row['train_value_loss']:.4f} "
            f"top1={row['train_policy_top1']:.3f} | "
            f"val_loss={row['val_total_loss']:.4f} "
            f"val_top1={row['val_policy_top1']:.3f}"
        )

    _write_log(args.log, rows)
    _save_checkpoint(
        path=args.checkpoint,
        model=model,
        args=args,
        rows=rows,
        selection_metric="latest",
    )
    print(f"saved_checkpoint: {args.checkpoint}")
    print(f"saved_best_val_loss_checkpoint: {args.best_val_loss_checkpoint}")
    print(f"saved_best_val_top1_checkpoint: {args.best_val_top1_checkpoint}")
    print(f"saved_log: {args.log}")


def _run_epoch(
    model: PolicyValueNet,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    value_loss_weight: float,
) -> dict[str, float]:
    model.train()
    totals = _MetricTotals()
    for states, policies, values in loader:
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)

        optimizer.zero_grad(set_to_none=True)
        policy_logits, predicted_values = model(states)
        total_loss, policy_loss, value_loss = policy_value_loss(
            policy_logits=policy_logits,
            predicted_values=predicted_values,
            target_policies=policies,
            target_values=values,
            value_loss_weight=value_loss_weight,
        )
        total_loss.backward()
        optimizer.step()
        totals.update(policy_logits, predicted_values, policies, values, total_loss, policy_loss, value_loss)
    return totals.as_dict()


@torch.no_grad()
def _evaluate(
    model: PolicyValueNet,
    loader: DataLoader,
    device: torch.device,
    value_loss_weight: float,
) -> dict[str, float]:
    if len(loader.dataset) == 0:
        return {
            "total_loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "policy_top1": 0.0,
            "value_mae": 0.0,
        }

    model.eval()
    totals = _MetricTotals()
    for states, policies, values in loader:
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)

        policy_logits, predicted_values = model(states)
        total_loss, policy_loss, value_loss = policy_value_loss(
            policy_logits=policy_logits,
            predicted_values=predicted_values,
            target_policies=policies,
            target_values=values,
            value_loss_weight=value_loss_weight,
        )
        totals.update(policy_logits, predicted_values, policies, values, total_loss, policy_loss, value_loss)
    return totals.as_dict()


class _MetricTotals:
    def __init__(self) -> None:
        self.sample_count = 0
        self.total_loss = 0.0
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.policy_top1 = 0.0
        self.value_mae = 0.0

    def update(
        self,
        policy_logits: torch.Tensor,
        predicted_values: torch.Tensor,
        target_policies: torch.Tensor,
        target_values: torch.Tensor,
        total_loss: torch.Tensor,
        policy_loss: torch.Tensor,
        value_loss: torch.Tensor,
    ) -> None:
        batch_size = int(target_values.shape[0])
        self.sample_count += batch_size
        self.total_loss += float(total_loss.detach().cpu()) * batch_size
        self.policy_loss += float(policy_loss.detach().cpu()) * batch_size
        self.value_loss += float(value_loss.detach().cpu()) * batch_size

        predicted_actions = torch.argmax(policy_logits, dim=1)
        target_actions = torch.argmax(target_policies, dim=1)
        self.policy_top1 += float((predicted_actions == target_actions).float().sum().cpu())
        self.value_mae += float(
            torch.abs(predicted_values - target_values).sum().detach().cpu()
        )

    def as_dict(self) -> dict[str, float]:
        count = max(self.sample_count, 1)
        return {
            "total_loss": self.total_loss / count,
            "policy_loss": self.policy_loss / count,
            "value_loss": self.value_loss / count,
            "policy_top1": self.policy_top1 / count,
            "value_mae": self.value_mae / count,
        }


def _split_dataset(
    dataset: NPZSelfPlayDataset,
    val_fraction: float,
    seed: int,
) -> tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1).")
    val_size = int(len(dataset) * val_fraction)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("Training split is empty.")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def _write_log(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _save_checkpoint(
    path: Path,
    model: PolicyValueNet,
    args: argparse.Namespace,
    rows: list[dict[str, float | int | str]],
    selection_metric: str,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "hidden_channels": args.hidden_channels,
                "dataset": str(args.dataset),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "value_loss_weight": args.value_loss_weight,
            },
            "last_metrics": rows[-1] if rows else {},
            "selection_metric": selection_metric,
        },
        path,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyTorch policy-value network from 爱恩斯坦棋 NPZ self-play data."
    )
    parser.add_argument("dataset", type=Path, help="Path to self-play NPZ data.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--best-val-loss-checkpoint",
        type=Path,
        default=DEFAULT_BEST_VAL_LOSS_CHECKPOINT,
    )
    parser.add_argument(
        "--best-val-top1-checkpoint",
        type=Path,
        default=DEFAULT_BEST_VAL_TOP1_CHECKPOINT,
    )
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    return parser.parse_args()


if __name__ == "__main__":
    main()
