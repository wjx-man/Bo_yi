"""High-level self-play trainer."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from src.agents.actor_critic_agent import ActorCriticAgent
from src.agents.random_agent import RandomAgent
from src.agents.rule_based import RuleBasedAgent
from src.models.actor_critic import ActorCriticNet
from src.utils.device import configure_torch_backend, resolve_device
from src.utils.logger import CSVLogger
from src.utils.plotting import plot_all_curves
from src.utils.serialization import save_pickle
from src.utils.seed import set_global_seed

from .evaluator import Evaluator
from .losses import compute_actor_critic_loss
from .replay_buffer import TransitionReplayBuffer
from .self_play import SelfPlayRunner


class SelfPlayTrainer:
    """Coordinate self-play, replay sampling, optimization, and evaluation."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.project_root = Path(config.get("project_root", ".")).resolve()
        self._prepare_run_outputs()
        self.device = resolve_device(config.get("device", "auto"))
        configure_torch_backend(self.device)
        print(f"Using device: {self.device}", flush=True)
        set_global_seed(config.get("seed"))
        self.model = ActorCriticNet(
            in_channels=config.get("in_channels", 18),
            num_res_blocks=config.get("num_res_blocks", 2),
        ).to(self.device)
        self.agent = ActorCriticAgent(self.model, device=str(self.device), mode="sample", temperature=config.get("temperature", 1.0))
        self.replay = TransitionReplayBuffer(config.get("replay_buffer_size", 100_000), seed=config.get("seed"))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 3.0e-4),
            weight_decay=config.get("weight_decay", 1.0e-4),
        )
        self.scheduler = self._build_scheduler()
        self.self_play = SelfPlayRunner(
            self.agent,
            self.replay,
            game_record_dir=self.project_root / "data" / "game_records",
            max_steps=config.get("max_steps", 300),
        )
        self.evaluator = Evaluator(max_steps=config.get("max_steps", 300))
        self.last_eval_win_rate: float | None = None
        self.best_eval_win_rate: float = -1.0
        self.best_eval_iteration: int = 0
        self.train_logger = CSVLogger(
            self.project_root / "logs" / "train_metrics.csv",
            ["iteration", "avg_reward", "loss", "policy_loss", "value_loss", "entropy", "buffer_size", "lr"],
        )
        self.eval_logger = CSVLogger(
            self.project_root / "logs" / "eval_metrics.csv",
            [
                "iteration",
                "opponent",
                "num_games",
                "win_rate",
                "avg_steps",
                "avg_reward",
                "reach_corner_wins",
                "capture_all_wins",
                "timeout_wins",
            ],
        )

    def _prepare_run_outputs(self) -> None:
        """Backup and optionally clear previous training outputs before a new run."""
        output_paths = [
            self.project_root / "checkpoints",
            self.project_root / "logs",
            self.project_root / "plots",
            self.project_root / "data" / "game_records",
            self.project_root / "data" / "replay_buffer",
        ]
        existing = [path for path in output_paths if path.exists() and any(path.iterdir())]
        if not existing:
            return

        if self.config.get("backup_outputs_on_start", True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_root = self.project_root / "backups" / f"run_{timestamp}"
            backup_root.mkdir(parents=True, exist_ok=True)
            for src in existing:
                rel = src.relative_to(self.project_root)
                dest = backup_root / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(src, dest, dirs_exist_ok=True)
            config_path = self.project_root / "configs" / "default.yaml"
            if config_path.exists():
                dest = backup_root / "configs" / "default.yaml"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(config_path, dest)
            print(f"Backed up previous outputs to {backup_root}", flush=True)

        if self.config.get("clear_outputs_on_start", True):
            for path in output_paths:
                if not path.exists():
                    continue
                resolved = path.resolve()
                if self.project_root not in resolved.parents:
                    raise RuntimeError(f"Refusing to clear path outside project: {resolved}")
                shutil.rmtree(resolved)
                resolved.mkdir(parents=True, exist_ok=True)
            print("Cleared previous checkpoints, logs, plots, game records, and replay buffer.", flush=True)

    def train(self) -> None:
        """Run the configured training loop."""
        cfg = self.config
        global_game_id = 1
        total_iterations = cfg.get("num_iterations", 10)
        for iteration in range(1, total_iterations + 1):
            rewards = []
            games_this_iter = cfg.get("self_play_games_per_iter", 4)
            for _ in range(games_this_iter):
                record = self.self_play.play_game(global_game_id, seed=None if cfg.get("seed") is None else cfg["seed"] + global_game_id)
                rewards.append(sum(move["reward"] for move in record["moves"]))
                global_game_id += 1

            metrics = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
            train_steps = cfg.get("train_steps_per_iter", 8)
            if len(self.replay) > 0:
                for _ in range(train_steps):
                    batch = self.replay.sample(cfg.get("batch_size", 128))
                    loss, step_metrics = compute_actor_critic_loss(
                        self.model,
                        batch,
                        gamma=cfg.get("gamma", 0.95),
                        value_coef=cfg.get("value_coef", 0.5),
                        entropy_coef=cfg.get("entropy_coef", 0.01),
                        ppo_clip_eps=cfg.get("ppo_clip_eps", 0.2),
                        device=self.device,
                    )
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.get("grad_clip_norm", 1.0))
                    self.optimizer.step()
                    metrics = step_metrics

            evaluated = False
            if iteration % cfg.get("eval_interval", 5) == 0:
                self._evaluate(iteration)
                evaluated = True
            self._step_scheduler(metrics, evaluated)
            self.train_logger.log(
                {
                    "iteration": iteration,
                    "avg_reward": sum(rewards) / max(1, len(rewards)),
                    "buffer_size": len(self.replay),
                    "lr": self._current_lr(),
                    **metrics,
                }
            )
            if iteration % cfg.get("save_interval", 5) == 0:
                self.save_checkpoint(iteration)
                save_pickle(self.replay.state_dict(), self.project_root / "data" / "replay_buffer" / "replay.pkl")
            plot_all_curves(self.project_root / "logs", self.project_root / "plots")
            self._print_progress(
                iteration=iteration,
                total_iterations=total_iterations,
                games_this_iter=games_this_iter,
                avg_reward=sum(rewards) / max(1, len(rewards)),
                metrics=metrics,
            )

        self.save_checkpoint(cfg.get("num_iterations", 10))
        save_pickle(self.replay.state_dict(), self.project_root / "data" / "replay_buffer" / "replay.pkl")
        plot_all_curves(self.project_root / "logs", self.project_root / "plots")

    def _evaluate(self, iteration: int) -> None:
        greedy = ActorCriticAgent(self.model, device=str(self.device), mode="greedy")
        results = []
        baseline_agents = (RandomAgent(seed=iteration), RuleBasedAgent(seed=iteration))
        for opponent_agent in baseline_agents:
            result = self.evaluator.evaluate(
                greedy,
                opponent_agent,
                num_games=self.config.get("eval_games", 10),
                iteration=iteration,
                seed=self.config.get("seed"),
            )
            self.eval_logger.log(result.as_dict())
            results.append(result)
        self.last_eval_win_rate = sum(result.win_rate for result in results) / max(1, len(results))
        total_games = sum(result.num_games for result in results)
        average_row = {
            "iteration": iteration,
            "opponent": "average",
            "num_games": total_games,
            "win_rate": self.last_eval_win_rate,
            "avg_steps": sum(result.avg_steps for result in results) / max(1, len(results)),
            "avg_reward": sum(result.avg_reward for result in results) / max(1, len(results)),
            "reach_corner_wins": sum(result.reach_corner_wins for result in results),
            "capture_all_wins": sum(result.capture_all_wins for result in results),
            "timeout_wins": sum(result.timeout_wins for result in results),
        }
        self.eval_logger.log(average_row)
        detail = " | ".join(f"{result.opponent}={result.win_rate * 100:.0f}%" for result in results)
        print(f"Eval {iteration} | {detail} | average={self.last_eval_win_rate * 100:.0f}%", flush=True)
        if self.last_eval_win_rate > self.best_eval_win_rate:
            self.best_eval_win_rate = self.last_eval_win_rate
            self.best_eval_iteration = iteration
            path = self.save_checkpoint(iteration, filename="model_best.pt")
            print(
                f"New best checkpoint saved: {path} "
                f"(average win_rate={self.best_eval_win_rate * 100:.0f}%, iter={iteration})",
                flush=True,
            )

    def _build_scheduler(self):
        """Create an optional learning-rate scheduler."""
        scheduler_name = str(self.config.get("scheduler", "none")).lower()
        if scheduler_name == "none":
            return None
        if scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.config.get("num_iterations", 10)),
                eta_min=self.config.get("min_learning_rate", 1.0e-5),
            )
        if scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=self.config.get("plateau_factor", 0.5),
                patience=self.config.get("plateau_patience", 5),
                min_lr=self.config.get("min_learning_rate", 1.0e-5),
            )
        raise ValueError("scheduler must be one of: none, cosine, plateau")

    def _step_scheduler(self, metrics: dict[str, float], evaluated: bool) -> None:
        """Advance the configured scheduler."""
        if self.scheduler is None:
            return
        scheduler_name = str(self.config.get("scheduler", "none")).lower()
        if scheduler_name == "cosine":
            self.scheduler.step()
        elif scheduler_name == "plateau" and evaluated:
            score = self.last_eval_win_rate
            if score is not None:
                self.scheduler.step(score)

    def _current_lr(self) -> float:
        """Return the current optimizer learning rate."""
        return float(self.optimizer.param_groups[0]["lr"])

    def _print_progress(
        self,
        iteration: int,
        total_iterations: int,
        games_this_iter: int,
        avg_reward: float,
        metrics: dict[str, float],
    ) -> None:
        """Print one compact training progress line."""
        win_rate = "N/A" if self.last_eval_win_rate is None else f"{self.last_eval_win_rate * 100:.0f}%"
        print(
            f"Iter {iteration}/{total_iterations} | "
            f"games={games_this_iter} | "
            f"buffer={len(self.replay)} | "
            f"reward={avg_reward:.3f} | "
            f"loss={metrics.get('loss', 0.0):.4f} | "
            f"win_rate={win_rate} | "
            f"lr={self._current_lr():.2e}",
            flush=True,
        )

    def save_checkpoint(self, iteration: int, filename: str | None = None) -> Path:
        """Save model and optimizer state."""
        ckpt_dir = self.project_root / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / (filename or f"model_iter_{iteration:04d}.pt")
        torch.save(
            {
                "iteration": iteration,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": None if self.scheduler is None else self.scheduler.state_dict(),
                "best_eval_win_rate": self.best_eval_win_rate,
                "best_eval_iteration": self.best_eval_iteration,
                "config": self.config,
            },
            path,
        )
        if filename is None:
            torch.save({"model_state_dict": self.model.state_dict(), "config": self.config}, ckpt_dir / "model.pt")
        return path
