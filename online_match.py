#!/usr/bin/env python3
"""
爱恩斯坦棋 — 在线对战服务（根目录脚本）

本脚本在本地启动一个 TCP 对战房间，服务端持有唯一棋局 `EinsteinGame`，
所有「是否合法」均通过 `get_legal_moves` / `apply_move` / `pass_turn` 与本项目引擎一致。

================================================================================
一、传输格式：换行分隔 JSON（NDJSON）
================================================================================
每条消息为一行 UTF-8 JSON，以换行符 \\n 结束。禁止使用裸换行出现在字符串值内。

================================================================================
二、棋盘与骰子约定（双方收到的 state 中字段含义一致）
================================================================================
- board: 5x5 二维数组，与 `EinsteinGame.encode_board()` 一致。
  - 0 表示空位
  - 正整数 1..6 表示红方棋子编号
  - 负整数 -1..-6 表示蓝方棋子编号（绝对值为编号）
- current_player: 当前行棋方，\"red\" 或 \"blue\"
- dice_roll: 当前回合骰子点数，整数 1..6（轮到谁走棋，就是谁本回合看到的骰子）
- turn_index: 回合序号（与引擎内一致，从 1 开始）
- winner: 终局前为 null；终局后为 \"red\" / \"blue\"
- your_role: 本条消息接收方在本局中的颜色（\"red\" 或 \"blue\"）

可选字段：
- legal_moves: 仅当 `your_role == current_player` 且棋局未结束时，列出当前方全部合法着法
  （与引擎 `get_legal_moves()` 一致）。对方仅收到 `legal_move_count`，不收到具体列表。
- legal_move_count: 当前行棋方的合法着法数量（双方可见）

着法对象 move 的字段：
- color: \"red\" | \"blue\"
- piece_number: 1..6
- from: [row, col]
- to: [row, col]

坐标约定：row、col 均为 0..4，[0,0] 为红方角，[4,4] 为蓝方角（与引擎一致）。

================================================================================
三、客户端 → 服务端消息
================================================================================
1) 布阵阶段（开局各发一次）
   {\"type\": \"layout\", \"order\": [a,b,c,d,e,f]}
   将六个棋子编号按己方起始格固定顺序映射到位置上。顺序为引擎内
   `PlayerColor.start_positions` 的顺序：
   - 红方六个格依次为：(0,0),(0,1),(0,2),(1,0),(1,1),(2,0)
   - 蓝方六个格依次为：(4,4),(4,3),(4,2),(3,4),(3,3),(2,4)
   `order` 须为 1..6 的一个排列。

2) 行棋阶段（轮到己方时）
   {\"type\": \"move\", \"piece_number\": n, \"to\": [row, col]}
   必须精确匹配当前 `legal_moves` 中某一条（起点由棋盘唯一确定，无需上传）。

   若无着法需让过（引擎要求先判明无合法走子）：
   {\"type\": \"pass\"}

================================================================================
四、服务端 → 客户端消息
================================================================================
- {\"type\": \"welcome\", \"role\": \"red\"|\"blue\", \"protocol_version\": 1}
- {\"type\": \"state\", ...}  # 见第二节
- {\"type\": \"error\", \"message\": \"...\"}  # 非法请求或违规着法
- {\"type\": \"game_over\", \"winner\": \"red\"|\"blue\"|null, \"reason\": \"...\"}

================================================================================
五、启动示例
================================================================================
    python online_match.py --host 0.0.0.0 --port 8765

红方与蓝方各用任意 TCP 客户端（如 nc、自写脚本）连接同一地址；先连为红，后连为蓝。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from typing import Any

from einstein_chess.engine import EinsteinGame, Move, PlayerColor, Position


PROTOCOL_VERSION = 1

RED_START_ORDER: tuple[Position, ...] = PlayerColor.RED.start_positions
BLUE_START_ORDER: tuple[Position, ...] = PlayerColor.BLUE.start_positions


def _str_from_color(c: PlayerColor) -> str:
    return c.value


def _move_to_wire(m: Move) -> dict[str, Any]:
    return {
        "color": m.color.value,
        "piece_number": m.piece_number,
        "from": [m.from_position[0], m.from_position[1]],
        "to": [m.to_position[0], m.to_position[1]],
    }


def _wire_state_for(game: EinsteinGame, viewer: PlayerColor) -> dict[str, Any]:
    legal = game.get_legal_moves()
    show_moves = game.winner is None and viewer == game.current_player
    payload: dict[str, Any] = {
        "type": "state",
        "turn_index": game.turn_index,
        "current_player": _str_from_color(game.current_player),
        "dice_roll": game.dice_roll,
        "board": [list(row) for row in game.encode_board()],
        "winner": None if game.winner is None else _str_from_color(game.winner),
        "your_role": _str_from_color(viewer),
        "legal_move_count": len(legal),
    }
    if show_moves:
        payload["legal_moves"] = [_move_to_wire(m) for m in legal]
    return payload


def _layout_from_order(color: PlayerColor, order: list[int]) -> dict[int, Position]:
    positions = RED_START_ORDER if color is PlayerColor.RED else BLUE_START_ORDER
    if len(order) != 6:
        raise ValueError("order 必须恰好包含 6 个整数")
    if sorted(order) != [1, 2, 3, 4, 5, 6]:
        raise ValueError("order 必须是 1..6 的一个排列")
    return {int(n): positions[i] for i, n in enumerate(order)}


def _find_matching_move(
    game: EinsteinGame, piece_number: int, to: Position
) -> Move | None:
    for m in game.get_legal_moves():
        if m.piece_number == piece_number and m.to_position == to:
            return m
    return None


class _JsonLineProtocol:
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.reader = reader
        self.writer = writer

    async def send_obj(self, obj: dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
        self.writer.write(line.encode("utf-8"))
        await self.writer.drain()

    async def read_obj(self) -> dict[str, Any]:
        raw = await self.reader.readline()
        if not raw:
            raise ConnectionError("连接已关闭")
        line = raw.decode("utf-8").strip()
        if not line:
            raise ValueError("收到空行")
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError("消息必须是 JSON 对象")
        return data


async def _run_match(
    red: _JsonLineProtocol,
    blue: _JsonLineProtocol,
    rng: random.Random,
) -> None:
    layouts: dict[PlayerColor, dict[int, Position]] = {}

    async def recv_layout(proto: _JsonLineProtocol, color: PlayerColor) -> None:
        while True:
            msg = await proto.read_obj()
            if msg.get("type") != "layout":
                await proto.send_obj(
                    {"type": "error", "message": "布阵阶段请发送 {\"type\":\"layout\", \"order\":[...]}"}
                )
                continue
            order = msg.get("order")
            if not isinstance(order, list):
                await proto.send_obj({"type": "error", "message": "layout.order 必须是长度为 6 的数组"})
                continue
            try:
                layout = _layout_from_order(color, [int(x) for x in order])
                game_probe = EinsteinGame(rng=random.Random(0))
                normalized = game_probe._normalize_layout(color, layout)
            except Exception as exc:
                await proto.send_obj({"type": "error", "message": f"布阵不合法: {exc}"})
                continue
            layouts[color] = normalized
            return

    await asyncio.gather(
        recv_layout(red, PlayerColor.RED),
        recv_layout(blue, PlayerColor.BLUE),
    )

    game = EinsteinGame(rng=rng)
    game.reset(
        red_layout=layouts[PlayerColor.RED],
        blue_layout=layouts[PlayerColor.BLUE],
    )

    async def broadcast_state() -> None:
        await asyncio.gather(
            red.send_obj(_wire_state_for(game, PlayerColor.RED)),
            blue.send_obj(_wire_state_for(game, PlayerColor.BLUE)),
        )

    await broadcast_state()

    while game.winner is None:
        color = game.current_player
        active = red if color is PlayerColor.RED else blue

        if not game.get_legal_moves():
            game.pass_turn()
            await asyncio.gather(
                red.send_obj(_wire_state_for(game, PlayerColor.RED)),
                blue.send_obj(_wire_state_for(game, PlayerColor.BLUE)),
            )
            continue

        msg = await active.read_obj()
        mtype = msg.get("type")

        if mtype == "pass":
            try:
                game.pass_turn()
            except Exception as exc:
                await active.send_obj({"type": "error", "message": str(exc)})
                continue
        elif mtype == "move":
            try:
                piece = int(msg["piece_number"])
                to_raw = msg["to"]
                to_pos = (int(to_raw[0]), int(to_raw[1]))
            except Exception:
                await active.send_obj({"type": "error", "message": "move 需要 piece_number 与 to:[row,col]"})
                continue
            chosen = _find_matching_move(game, piece, to_pos)
            if chosen is None:
                await active.send_obj({"type": "error", "message": "不合法着法（与引擎 legal_moves 不一致）"})
                continue
            try:
                game.apply_move(chosen)
            except Exception as exc:
                await active.send_obj({"type": "error", "message": str(exc)})
                continue
        else:
            await active.send_obj({"type": "error", "message": "未知消息类型；请发 move 或 pass"})
            continue

        await broadcast_state()

    reason = "goal_or_capture"
    win = game.winner
    over = {
        "type": "game_over",
        "winner": None if win is None else win.value,
        "reason": reason,
        "final": {
            "board": [list(row) for row in game.encode_board()],
            "turn_index": game.turn_index,
        },
    }
    await asyncio.gather(red.send_obj(over), blue.send_obj(over))


async def _handle_client(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    lobby: "MatchLobby",
) -> None:
    addr = writer.get_extra_info("peername")
    proto = _JsonLineProtocol(reader, writer)
    try:
        await lobby.attach(proto, addr)
    except Exception as exc:
        try:
            await proto.send_obj({"type": "error", "message": str(exc)})
        except Exception:
            pass
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass


class MatchLobby:
    """先连入者为红方；第二个连入者与红方配对并开始一局。"""

    def __init__(self, rng: random.Random) -> None:
        self._rng = rng
        self._red_proto: _JsonLineProtocol | None = None
        self._red_session_done: asyncio.Future[None] | None = None
        self._mutex = asyncio.Lock()

    async def attach(self, proto: _JsonLineProtocol, addr: object) -> None:
        session_done: asyncio.Future[None] | None = None
        async with self._mutex:
            if self._red_proto is None:
                self._red_proto = proto
                loop = asyncio.get_running_loop()
                self._red_session_done = loop.create_future()
                session_done = self._red_session_done
                await proto.send_obj(
                    {
                        "type": "welcome",
                        "role": "red",
                        "protocol_version": PROTOCOL_VERSION,
                        "peer": str(addr),
                    }
                )
            else:
                red = self._red_proto
                self._red_proto = None
                done_fut = self._red_session_done
                self._red_session_done = None
                await proto.send_obj(
                    {
                        "type": "welcome",
                        "role": "blue",
                        "protocol_version": PROTOCOL_VERSION,
                        "peer": str(addr),
                    }
                )
                try:
                    await _run_match(red, proto, self._rng)
                finally:
                    if done_fut is not None and not done_fut.done():
                        done_fut.set_result(None)
                return

        if session_done is not None:
            await session_done


async def _main_async(host: str, port: int, seed: int | None) -> None:
    rng = random.Random(seed)
    lobby = MatchLobby(rng)
    server = await asyncio.start_server(
        lambda r, w: asyncio.create_task(_handle_client(r, w, lobby)),
        host=host,
        port=port,
    )
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(f"爱恩斯坦棋在线服务已启动: {addrs}", file=sys.stderr)
    print("先接入者为红方，第二个接入者为蓝方。", file=sys.stderr)
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="爱恩斯坦棋 TCP 在线对战服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    parser.add_argument("--seed", type=int, default=None, help="随机数种子（骰子可复现）")
    args = parser.parse_args()
    try:
        asyncio.run(_main_async(args.host, args.port, args.seed))
    except KeyboardInterrupt:
        print("\n已停止服务。", file=sys.stderr)


if __name__ == "__main__":
    main()
