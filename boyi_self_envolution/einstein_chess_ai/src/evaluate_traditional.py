"""Evaluate a trained model against traditional search agents."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import time

from src.agents.base import Agent
from src.agents.actor_critic_agent import ActorCriticAgent
from src.agents.search import AlphaBetaAgent, MCTSAgent, MinimaxAgent
from src.env.env import EinsteinChessEnv
from src.env.rules import BLUE, RED, opponent
from src.utils.device import resolve_device
from src.utils.logger import CSVLogger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ActorCritic against search baselines.")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--num_games", type=int, default=1000)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max_steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimax_depth", type=int, default=3)
    parser.add_argument("--alphabeta_depth", type=int, default=4)
    parser.add_argument("--mcts_simulations", type=int, default=512)
    parser.add_argument("--mcts_rollout_steps", type=int, default=100)
    parser.add_argument("--mcts_exploration", type=float, default=1.2)
    parser.add_argument("--progress_interval", type=int, default=1)
    parser.add_argument("--output", default="logs/traditional_eval_metrics.csv")
    return parser


def evaluate_with_progress(
    agent: Agent,
    opponent_agent: Agent,
    num_games: int,
    max_steps: int,
    seed: int | None,
    progress_interval: int,
) -> dict:
    """Evaluate `agent` against one opponent and print periodic progress."""
    start = time.perf_counter()
    last_report_time = start
    last_report_finished = 0
    wins = 0
    total_steps = 0
    total_reward = 0.0
    reasons: Counter[str] = Counter()
    opponent_name = getattr(opponent_agent, "name", type(opponent_agent).__name__)

    for game_idx in range(num_games):
        finished = game_idx + 1
        if progress_interval > 0 and (finished == 1 or finished % progress_interval == 0):
            print(f"{opponent_name}: starting game {finished}/{num_games}", flush=True)
        env = EinsteinChessEnv(seed=None if seed is None else seed + game_idx, max_steps=max_steps)
        env.reset(first_player=RED if game_idx % 2 == 0 else BLUE, seed=None if seed is None else seed + game_idx)
        controlled_player = RED if game_idx % 2 == 0 else BLUE
        agents = {controlled_player: agent, opponent(controlled_player): opponent_agent}
        done = False
        game_reward = 0.0
        while not done:
            action_player = env.current_player
            action = agents[action_player].select_action(env)
            _, reward, done, _ = env.step(action)
            game_reward += reward if action_player == controlled_player else -reward

        won = env.winner == controlled_player
        wins += int(won)
        total_steps += env.step_count
        total_reward += game_reward
        reasons[env.win_reason or "unknown"] += int(won)

        if progress_interval > 0 and (finished % progress_interval == 0 or finished == num_games):
            now = time.perf_counter()
            elapsed = now - start
            interval_elapsed = now - last_report_time
            interval_games = finished - last_report_finished
            print(
                f"{opponent_name}: {finished}/{num_games} "
                f"win_rate={wins / finished:.3f} "
                f"avg_steps={total_steps / finished:.1f} "
                f"elapsed={elapsed:.1f}s "
                f"interval={interval_elapsed:.1f}s "
                f"sec_per_game={interval_elapsed / max(1, interval_games):.1f}",
                flush=True,
            )
            last_report_time = now
            last_report_finished = finished

    elapsed = time.perf_counter() - start
    return {
        "iteration": 0,
        "opponent": opponent_name,
        "num_games": num_games,
        "win_rate": wins / max(1, num_games),
        "avg_steps": total_steps / max(1, num_games),
        "avg_reward": total_reward / max(1, num_games),
        "reach_corner_wins": reasons["reach_corner"],
        "capture_all_wins": reasons["capture_all"],
        "timeout_wins": reasons["timeout"],
        "elapsed_seconds": elapsed,
        "games_per_second": num_games / max(1.0e-9, elapsed),
    }


def main() -> None:
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = root / checkpoint

    device = resolve_device(args.device)
    agent = ActorCriticAgent(checkpoint=checkpoint, device=device, mode="greedy")
    logger = CSVLogger(
        root / args.output,
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
            "baseline_params",
            "elapsed_seconds",
            "games_per_second",
        ],
    )

    opponents = (
        MinimaxAgent(depth=args.minimax_depth, seed=args.seed + 11),
        AlphaBetaAgent(depth=args.alphabeta_depth, seed=args.seed + 22),
        MCTSAgent(
            simulations=args.mcts_simulations,
            rollout_steps=args.mcts_rollout_steps,
            exploration=args.mcts_exploration,
            seed=args.seed + 33,
        ),
    )
    print(f"Evaluating {checkpoint} on {device} for {args.num_games} games per opponent.", flush=True)
    for opponent_agent in opponents:
        params = describe_baseline(opponent_agent)
        print(f"Opponent: {opponent_agent.name} ({params})", flush=True)
        row = evaluate_with_progress(
            agent,
            opponent_agent,
            num_games=args.num_games,
            max_steps=args.max_steps,
            seed=args.seed,
            progress_interval=args.progress_interval,
        )
        row["baseline_params"] = params
        logger.log(row)
        print(row, flush=True)


def describe_baseline(agent: Agent) -> str:
    """Return compact baseline parameter text for CSV logs."""
    if isinstance(agent, MinimaxAgent):
        return f"depth={agent.depth}"
    if isinstance(agent, AlphaBetaAgent):
        return f"depth={agent.depth}"
    if isinstance(agent, MCTSAgent):
        return (
            f"simulations={agent.simulations};"
            f"rollout_steps={agent.rollout_steps};"
            f"exploration={agent.exploration}"
        )
    return ""


if __name__ == "__main__":
    main()
