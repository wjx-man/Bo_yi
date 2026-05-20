"""Command-line evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.agents.actor_critic_agent import ActorCriticAgent
from src.agents.mcts_agent import MCTSAgent
from src.agents.random_agent import RandomAgent
from src.agents.rule_based import RuleBasedAgent
from src.training.evaluator import Evaluator
from src.utils.device import resolve_device
from src.utils.logger import CSVLogger


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an Einstein chess checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--num_games", type=int, default=20)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--opponent", choices=["random", "rule_based", "mcts", "all"], default="all")
    parser.add_argument("--mcts_simulations", type=int, default=80)
    parser.add_argument("--mcts_depth", type=int, default=12)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    device = resolve_device(args.device)
    agent = ActorCriticAgent(checkpoint=root / args.checkpoint, device=device, mode="greedy")
    evaluator = Evaluator()
    logger = CSVLogger(
        root / "logs" / "eval_metrics.csv",
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
    opponents = []
    if args.opponent in {"random", "all"}:
        opponents.append(RandomAgent(seed=1))
    if args.opponent in {"rule_based", "all"}:
        opponents.append(RuleBasedAgent(seed=2))
    if args.opponent in {"mcts", "all"}:
        opponents.append(MCTSAgent(simulations=args.mcts_simulations, max_depth=args.mcts_depth, seed=3))

    for opponent in opponents:
        result = evaluator.evaluate(agent, opponent, num_games=args.num_games)
        logger.log(result.as_dict())
        print(result.as_dict())


if __name__ == "__main__":
    main()
