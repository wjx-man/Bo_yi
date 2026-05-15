#!/usr/bin/env python3
"""
连接在线房间（online_match），用本地训练好的策略 / Neural MCTS 行棋。

示例（连房主 172.18.11.154:8765，用策略网；先连为红、后连为蓝）::

    python scripts/online_model_client.py \\
        --host 172.18.11.154 --port 8765 \\
        --agent policy \\
        --checkpoint artifacts/checkpoints/policy_value_v2_best_loss.pt \\
        --device cuda

Neural MCTS::

    python scripts/online_model_client.py \\
        --host 172.18.11.154 --port 8765 \\
        --agent neural-mcts \\
        --checkpoint artifacts/checkpoints/policy_value_v2_best_loss.pt \\
        --device cuda \\
        --neural-mcts-simulations 80
"""

from __future__ import annotations

import argparse
import asyncio
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from einstein_chess.agents import FullNeuralMCTSAgent, NeuralMCTSAgent, PolicyValueAgent  # noqa: E402
from einstein_chess.engine import GameSnapshot, Move  # noqa: E402
from einstein_chess.online_match_client import (  # noqa: E402
    OnlineMatchClient,
    OnlineMatchProtocolError,
    layout_order_from_mapping,
    parse_state_message,
)
from einstein_chess.players import PlayerAgent  # noqa: E402


async def _choose_move_thread(
    agent: PlayerAgent, snapshot: GameSnapshot, legal: list[Move]
) -> Move:
    return await asyncio.to_thread(agent.choose_move, snapshot, legal)


async def run_agent_session(
    host: str,
    port: int,
    agent: PlayerAgent,
    rng: random.Random,
    layout_mode: str,
) -> None:
    client: OnlineMatchClient | None = None
    try:
        try:
            client = await OnlineMatchClient.connect(host, port)
        except (ConnectionError, OSError, OnlineMatchProtocolError) as exc:
            print(f"连接失败: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(1) from exc

        role = client.role
        print(f"已连接，角色: {role.label} ({role.value})", flush=True)

        if layout_mode == "random":
            layout = agent.choose_layout(role, role.start_positions, rng)
        else:
            layout = {n: pos for n, pos in zip(range(1, 7), role.start_positions)}
        order = layout_order_from_mapping(layout, role)
        await client.send_layout(order)

        while True:
            msg = await client.recv_message()
            mtype = msg.get("type")
            if mtype == "game_over":
                print("终局:", msg, flush=True)
                return
            if mtype == "error":
                print("错误:", msg, flush=True)
                return
            if mtype != "state":
                print("忽略:", msg, flush=True)
                continue

            view = parse_state_message(msg)
            if not view.is_my_turn() or view.legal_moves is None:
                continue

            moves = list(view.legal_moves)
            if not moves:
                await client.send_pass()
                print("让过(pass)", flush=True)
                continue

            snap = view.to_snapshot()
            move = await _choose_move_thread(agent, snap, moves)
            await client.send_move_obj(move)
            print(
                f"走子: 棋子{move.piece_number} {move.from_position}->{move.to_position}",
                flush=True,
            )
    finally:
        if client is not None:
            await client.aclose()


def _build_agent(args: argparse.Namespace, rng: random.Random) -> PlayerAgent:
    if args.agent == "policy":
        return PolicyValueAgent(
            checkpoint_path=args.checkpoint,
            name="OnlinePolicy",
            device=args.device,
        )
    if args.agent == "full-neural-mcts":
        return FullNeuralMCTSAgent(
            checkpoint_path=args.checkpoint,
            name="OnlineFullNeuralMCTS",
            simulations=args.full_neural_mcts_simulations,
            c_puct=args.c_puct,
            max_depth=args.full_neural_mcts_depth,
            chance_mode=args.chance_mode,
            device=args.device,
            rng=rng,
        )
    return NeuralMCTSAgent(
        checkpoint_path=args.checkpoint,
        name="OnlineNeuralMCTS",
        simulations=args.neural_mcts_simulations,
        c_puct=args.c_puct,
        device=args.device,
        rng=rng,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="在线对战 — 使用本地模型客户端")
    parser.add_argument("--host", default="127.0.0.1", help="房主 / 服务器 IP")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--agent",
        choices=("policy", "neural-mcts", "full-neural-mcts"),
        default="policy",
        help="policy=纯策略网络选子；neural-mcts=网络先验+估值的 MCTS",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="例如 artifacts/checkpoints/policy_value_v2_best_loss.pt",
    )
    parser.add_argument("--device", default="cpu", help="cuda 或 cpu")
    parser.add_argument("--neural-mcts-simulations", type=int, default=80)
    parser.add_argument("--full-neural-mcts-simulations", type=int, default=80)
    parser.add_argument("--full-neural-mcts-depth", type=int, default=12)
    parser.add_argument("--chance-mode", choices=("sample", "enumerate"), default="sample")
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--layout",
        choices=("default", "random"),
        default="default",
        help="布阵：默认按编号顺序对应起始格；random=由 PlayerAgent.choose_layout 随机",
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        print(f"找不到 checkpoint 文件: {args.checkpoint}", file=sys.stderr, flush=True)
        raise SystemExit(1)

    rng = random.Random(args.seed)
    agent = _build_agent(args, rng)

    asyncio.run(
        run_agent_session(
            args.host,
            args.port,
            agent,
            rng,
            args.layout,
        )
    )


if __name__ == "__main__":
    main()
