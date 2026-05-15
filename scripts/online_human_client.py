#!/usr/bin/env python3
"""
真人玩家 — 连接 `online_match.py` 服务（NDJSON/TCP）。

用法（先连为红、后连为蓝）::

    python online_match.py --host 127.0.0.1 --port 8765
    python scripts/online_human_client.py --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from einstein_chess.engine import PlayerColor  # noqa: E402
from einstein_chess.online_match_client import (  # noqa: E402
    OnlineMatchClient,
    OnlineMatchProtocolError,
    OnlineMatchStateView,
    parse_state_message,
)


def _console_utf8() -> None:
    for stream in (sys.stdout, sys.stdin, sys.stderr):
        reconf = getattr(stream, "reconfigure", None)
        if callable(reconf):
            try:
                reconf(encoding="utf-8")
            except Exception:
                pass


def _cell_char(cell: int) -> str:
    if cell == 0:
        return " . "
    if cell > 0:
        return f"R{cell}"
    return f"B{-cell}"


def _format_board(board: tuple[tuple[int, ...], ...]) -> str:
    header = "    " + " ".join(str(c) for c in range(5))
    lines = [header, "  +" + "-" * 19 + "+"]
    for r, row in enumerate(board):
        body = " ".join(_cell_char(v) for v in row)
        lines.append(f"{r} | {body} |")
    lines.append("  +" + "-" * 19 + "+")
    lines.append("  列 0..4 向右增大；行 0..4 向下增大。[0,0] 为红角，[4,4] 为蓝角。")
    return "\n".join(lines)


def _layout_help(role: PlayerColor) -> str:
    slots = role.start_positions
    lines = [
        f"你是【{role.label}】。请在己方 6 个起始格上指定棋子编号 1–6（各用一次）。",
        "按下格顺序依次对应你要放在该格的棋子编号（与服务器 layout.order 一致）：",
    ]
    for i, pos in enumerate(slots, start=1):
        lines.append(f"  第{i}格 (行,列)=({pos[0]},{pos[1]})")
    lines.append("一行输入 6 个整数，空格分隔，例如: 1 2 3 4 5 6")
    lines.append("直接回车表示使用默认顺序 1 2 3 4 5 6。")
    return "\n".join(lines)


async def _ask_line(prompt: str) -> str:
    return (await asyncio.to_thread(input, prompt)).strip()


async def _ask_layout(role: PlayerColor) -> list[int]:
    print(_layout_help(role), flush=True)
    while True:
        raw = await _ask_line("布阵> ")
        if not raw:
            return [1, 2, 3, 4, 5, 6]
        parts = raw.split()
        if len(parts) != 6:
            print("需要恰好 6 个数字。", flush=True)
            continue
        try:
            order = [int(x) for x in parts]
        except ValueError:
            print("请输入整数。", flush=True)
            continue
        if sorted(order) != [1, 2, 3, 4, 5, 6]:
            print("必须 1–6 各出现一次。", flush=True)
            continue
        return order


def _print_state(view: OnlineMatchStateView) -> None:
    print("\n" + "=" * 48, flush=True)
    print(
        f"回合 {view.turn_index} | 行棋方: {view.current_player.label} | "
        f"骰子: {view.dice_roll} | 你是: {view.your_role.label}",
        flush=True,
    )
    print(_format_board(view.board), flush=True)
    if view.winner is not None:
        print(f"胜负已分: {view.winner.label} 胜", flush=True)
        return
    if view.is_my_turn():
        if view.legal_moves is not None:
            print(f"轮到你走。合法着法数: {len(view.legal_moves)}", flush=True)
        else:
            print("轮到你走（等待同步…）", flush=True)
    else:
        print(
            f"对方思考中…（对方合法着法数: {view.legal_move_count}）",
            flush=True,
        )


async def _send_my_move(client: OnlineMatchClient, view: OnlineMatchStateView) -> None:
    assert view.legal_moves is not None
    moves = list(view.legal_moves)
    if not moves:
        print("当前无合法走子，发送让过(pass)。", flush=True)
        await client.send_pass()
        return

    print("\n可选着法（输入序号）：", flush=True)
    for i, m in enumerate(moves):
        fr, to = m.from_position, m.to_position
        print(
            f"  [{i}] 棋子{m.piece_number}  ({fr[0]},{fr[1]}) -> ({to[0]},{to[1]})",
            flush=True,
        )
    while True:
        raw = await _ask_line("走子> ")
        try:
            idx = int(raw)
        except ValueError:
            print("请输入整数序号。", flush=True)
            continue
        if not (0 <= idx < len(moves)):
            print("序号超出范围。", flush=True)
            continue
        await client.send_move_obj(moves[idx])
        return


async def _async_main(args: argparse.Namespace) -> None:
    _console_utf8()
    client: OnlineMatchClient | None = None
    try:
        try:
            client = await OnlineMatchClient.connect(args.host, args.port)
        except (ConnectionError, OSError, OnlineMatchProtocolError) as exc:
            print(f"连接失败: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(1) from exc

        print(f"已连接。你的角色: {client.role.label} ({client.role.value})", flush=True)

        while True:
            order = await _ask_layout(client.role)
            await client.send_layout(order)
            msg = await client.recv_message()
            if msg.get("type") == "error":
                print(f"布阵被拒: {msg.get('message')}", flush=True)
                continue
            if msg.get("type") == "state":
                break
            print(f"意外消息: {msg}", flush=True)
            raise SystemExit(2)

        while True:
            mtype = msg.get("type")
            if mtype == "game_over":
                print("\n对局结束:", msg, flush=True)
                return
            if mtype == "error":
                print(f"错误: {msg.get('message')}", flush=True)
                msg = await client.recv_message()
                continue
            if mtype != "state":
                print(f"未处理消息: {msg}", flush=True)
                msg = await client.recv_message()
                continue

            view = parse_state_message(msg)
            _print_state(view)

            if view.winner is not None:
                return

            if view.is_my_turn() and view.legal_moves is not None:
                while True:
                    await _send_my_move(client, view)
                    reply = await client.recv_message()
                    rtype = reply.get("type")
                    if rtype == "error":
                        print(f"着法被拒: {reply.get('message')}", flush=True)
                        continue
                    if rtype in ("state", "game_over"):
                        msg = reply
                        break
                    print(f"意外回复: {reply}", flush=True)
                    msg = reply
                    break
                continue

            msg = await client.recv_message()
    finally:
        if client is not None:
            await client.aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description="爱恩斯坦棋 — 真人玩家 TCP 客户端")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()
