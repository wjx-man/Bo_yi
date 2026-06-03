#!/usr/bin/env python3
"""最终线上比赛客户端。

这个脚本负责连接比赛服务器、提交开局布局、接收服务器局面、调用项目内部智能体，
再将智能体选择的走法发送回服务器。它不重新实现规则或算法，而是复用训练和本地
评估时使用的同一套 GameSnapshot、Move 和智能体代码。
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import random
import sys
import time
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from einstein_chess.agents import FullNeuralMCTSAgent, NeuralMCTSAgent, PolicyValueAgent  # noqa: E402
from einstein_chess.engine import GameSnapshot, Move, PlayerColor  # noqa: E402
from einstein_chess.online_match_client import (  # noqa: E402
    OnlineMatchClient,
    OnlineMatchProtocolError,
    layout_order_from_mapping,
    parse_state_message,
)
from einstein_chess.players import PlayerAgent  # noqa: E402


DEFAULT_CHECKPOINT = (
    Path("artifacts") / "checkpoints" / "policy_value_v4_full_best_loss.pt"
)


async def main_async(args: argparse.Namespace) -> None:
    """创建智能体、连接服务器，并完成服务器要求的全部对局。"""
    rng = random.Random(args.seed)
    agent = _build_agent(args, rng)

    try:
        # OnlineMatchClient 负责 TCP 连接和协议消息收发。
        client = await OnlineMatchClient.connect(args.host, args.port)
    except (ConnectionError, OSError, OnlineMatchProtocolError) as exc:
        print(f"connect_failed: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    try:
        total_games = int(client.welcome.get("total_games", args.games))
        layout_mode = str(client.welcome.get("layout_mode", "agent"))
        role = client.role
        print(
            f"connected role={role.value} total_games={total_games} "
            f"server_layout={layout_mode} agent={args.agent}",
            flush=True,
        )

        wins = 0
        losses = 0
        draws = 0
        for game_index in range(1, total_games + 1):
            # 每局开始前先按照命令行指定方式提交棋子布局。
            order = _layout_order(args.opening_layout, role, rng)
            await client.send_layout(order)
            print(f"[{game_index}/{total_games}] sent_layout={order}", flush=True)

            winner = await _play_one_game(
                client=client,
                agent=agent,
                game_index=game_index,
                total_games=total_games,
                verbose=args.verbose,
            )
            if winner is role:
                wins += 1
            elif winner is role.opponent:
                losses += 1
            else:
                draws += 1
            print(
                f"[{game_index}/{total_games}] score "
                f"wins={wins} losses={losses} draws={draws}",
                flush=True,
            )
    finally:
        await client.aclose()


async def _play_one_game(
    client: OnlineMatchClient,
    agent: PlayerAgent,
    game_index: int,
    total_games: int,
    verbose: bool,
) -> PlayerColor | None:
    """循环接收服务器消息，直到当前对局结束。"""
    del game_index, total_games
    while True:
        msg = await client.recv_message()
        mtype = msg.get("type")
        if mtype == "game_over":
            winner_raw = msg.get("winner")
            winner = None if winner_raw is None else PlayerColor(str(winner_raw))
            print(
                "game_over "
                f"winner={winner_raw} reason={msg.get('reason')} "
                f"steps={msg.get('final', {}).get('steps')}",
                flush=True,
            )
            return winner
        if mtype == "error":
            print(f"server_error: {msg.get('message')}", file=sys.stderr, flush=True)
            continue
        if mtype != "state":
            if verbose:
                print(f"ignored_message: {msg}", flush=True)
            continue

        # 将网络协议中的 state 消息解析为项目内部可使用的局面视图。
        view = parse_state_message(msg)
        if not view.is_my_turn() or view.legal_moves is None:
            continue

        legal_moves = list(view.legal_moves)
        if not legal_moves:
            await client.send_pass()
            if verbose:
                print(f"turn={view.turn_index} pass", flush=True)
            continue

        # 线上局面转换为 GameSnapshot 后，可以直接复用本地智能体的 choose_move()。
        snapshot = view.to_snapshot()
        started_at = time.perf_counter()
        # 搜索可能耗时，因此放入工作线程，避免阻塞 asyncio 网络循环。
        move = await asyncio.to_thread(agent.choose_move, snapshot, legal_moves)
        elapsed = time.perf_counter() - started_at
        # 客户端再次检查走法是否合法，避免向服务器发送错误动作。
        if move not in legal_moves:
            raise RuntimeError(f"agent returned illegal move: {move!r}")
        await client.send_move_obj(move)
        print(
            f"turn={view.turn_index} dice={view.dice_roll} "
            f"piece={move.piece_number} to={move.to_position} "
            f"think={elapsed:.3f}s",
            flush=True,
        )


def _build_agent(args: argparse.Namespace, rng: random.Random) -> PlayerAgent:
    """根据命令行参数创建纯网络、Root Neural MCTS 或 Full Neural MCTS。"""
    if args.agent == "policy":
        return PolicyValueAgent(
            checkpoint_path=args.checkpoint,
            name="CompetitionPolicy",
            device=args.device,
        )
    if args.agent == "neural-mcts":
        return NeuralMCTSAgent(
            checkpoint_path=args.checkpoint,
            name="CompetitionNeuralMCTS",
            simulations=args.neural_mcts_simulations,
            c_puct=args.c_puct,
            device=args.device,
            rng=rng,
        )
    # 默认创建最终推荐的 Full Neural MCTS 智能体。
    return FullNeuralMCTSAgent(
        checkpoint_path=args.checkpoint,
        name="CompetitionFullNeuralMCTS",
        simulations=args.full_neural_mcts_simulations,
        c_puct=args.c_puct,
        max_depth=args.full_neural_mcts_depth,
        chance_mode=args.chance_mode,
        device=args.device,
        rng=rng,
    )


def _layout_order(
    opening_layout: str,
    role: PlayerColor,
    rng: random.Random,
) -> list[int]:
    """生成提交给服务器的 1-6 号棋子布局顺序。"""
    if opening_layout == "default":
        return [1, 2, 3, 4, 5, 6]
    if opening_layout == "random":
        order = [1, 2, 3, 4, 5, 6]
        rng.shuffle(order)
        return order
    if opening_layout == "agent-default":
        layout = {
            number: position
            for number, position in zip(range(1, 7), role.start_positions)
        }
        return layout_order_from_mapping(layout, role)
    raise ValueError(f"unknown opening layout: {opening_layout}")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """定义线上比赛客户端支持的命令行参数。"""
    parser = argparse.ArgumentParser(
        description="Competition client for the online Einstein Chess match server."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Policy-value checkpoint. Defaults to the final V4_full model.",
    )
    parser.add_argument(
        "--agent",
        choices=("full-neural-mcts", "neural-mcts", "policy"),
        default="full-neural-mcts",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--games", type=int, default=1, help="Fallback if welcome omits total_games.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--opening-layout",
        choices=("default", "random", "agent-default"),
        default="default",
        help="Layout order to submit when the platform uses --layout agent.",
    )
    parser.add_argument("--full-neural-mcts-simulations", type=int, default=80)
    parser.add_argument("--full-neural-mcts-depth", type=int, default=12)
    parser.add_argument("--chance-mode", choices=("sample", "enumerate"), default="sample")
    parser.add_argument("--neural-mcts-simulations", type=int, default=80)
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    """同步入口：检查模型文件后启动异步客户端。"""
    args = _parse_args()
    if not args.checkpoint.is_file():
        print(f"checkpoint_not_found: {args.checkpoint}", file=sys.stderr, flush=True)
        raise SystemExit(1)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()


# 下一步阅读：回到 docs/CODE_PRESENTATION_WALKTHROUGH.md 进行整体复习。
