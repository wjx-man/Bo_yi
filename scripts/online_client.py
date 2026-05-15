#!/usr/bin/env python3
"""
连接 `online_match.py` 服务，完成布阵后以随机着法行棋（用于协议联调）。

示例::

    # 终端 1
    python online_match.py --host 127.0.0.1 --port 8765 --seed 0

    # 终端 2（先连为红）
    python scripts/online_client.py --host 127.0.0.1 --port 8765 --layout default

    # 终端 3（后连为蓝）
    python scripts/online_client.py --host 127.0.0.1 --port 8765 --layout default
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

from einstein_chess.online_match_client import (  # noqa: E402
    OnlineMatchClient,
    OnlineMatchProtocolError,
    parse_state_message,
)


async def _play_random(client: OnlineMatchClient, seed: int) -> None:
    rng = random.Random(seed + (0 if client.role.value == "red" else 1))

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
        if not view.legal_moves:
            await client.send_pass()
        else:
            move = rng.choice(view.legal_moves)
            await client.send_move_obj(move)
            print(
                f"走子: 棋子{move.piece_number} -> {move.to_position}",
                flush=True,
            )


async def _async_main(args: argparse.Namespace) -> None:
    try:
        client = await OnlineMatchClient.connect(args.host, args.port)
    except (ConnectionError, OSError, OnlineMatchProtocolError) as exc:
        print(f"连接失败: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(1) from exc

    print(f"已连接，角色: {client.role.value}", flush=True)
    try:
        order = list(range(1, 7))
        if args.layout == "random":
            random.Random(args.seed).shuffle(order)
        await client.send_layout(order)
        await _play_random(client, args.seed)
    finally:
        await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="在线对战 TCP 客户端（随机行棋）")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--layout",
        choices=("default", "random"),
        default="default",
        help="布阵：顺序 1..6 或随机排列",
    )
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
