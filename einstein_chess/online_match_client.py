"""
爱恩斯坦棋在线对战 — TCP 客户端接口（与根目录 `online_match.py` 协议一致）。

协议摘要：换行分隔 JSON（NDJSON），UTF-8。详见 `online_match.py` 文档字符串。
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .engine import GameSnapshot, Move, PlayerColor, Position

PROTOCOL_VERSION = 1


class OnlineMatchProtocolError(RuntimeError):
    """服务端首包非 welcome 或消息格式不符合约定时抛出。"""


@dataclass(frozen=True)
class OnlineMatchStateView:
    """解析后的 `{"type":"state", ...}` 视图，便于 UI 或智能体使用。"""

    turn_index: int
    current_player: PlayerColor
    dice_roll: int
    board: tuple[tuple[int, ...], ...]
    winner: PlayerColor | None
    your_role: PlayerColor
    legal_move_count: int
    legal_moves: tuple[Move, ...] | None

    def is_my_turn(self) -> bool:
        return self.winner is None and self.your_role is self.current_player

    def to_snapshot(self) -> GameSnapshot:
        """仅在己方回合且服务端下发了 `legal_moves` 时与线上一致；否则 `legal_moves` 为空元组。"""
        moves = self.legal_moves if self.legal_moves is not None else ()
        return GameSnapshot(
            board=self.board,
            current_player=self.current_player,
            dice_roll=self.dice_roll,
            winner=self.winner,
            legal_moves=moves,
            turn_index=self.turn_index,
        )


def wire_move_to_move(data: Mapping[str, Any]) -> Move:
    color = PlayerColor(str(data["color"]))
    piece_number = int(data["piece_number"])
    from_raw = data["from"]
    to_raw = data["to"]
    fr = (int(from_raw[0]), int(from_raw[1]))
    to = (int(to_raw[0]), int(to_raw[1]))
    return Move(
        color=color,
        piece_number=piece_number,
        from_position=fr,
        to_position=to,
    )


def move_to_wire_dict(move: Move) -> dict[str, Any]:
    return {
        "color": move.color.value,
        "piece_number": move.piece_number,
        "from": [move.from_position[0], move.from_position[1]],
        "to": [move.to_position[0], move.to_position[1]],
    }


def layout_order_from_mapping(
    layout: Mapping[int, Position], color: PlayerColor
) -> list[int]:
    """将 `number -> position` 转为服务端 `layout.order`（按该方 `start_positions` 顺序）。"""
    order: list[int] = []
    for pos in color.start_positions:
        for number, p in layout.items():
            if p == pos:
                order.append(int(number))
                break
        else:
            raise ValueError(f"layout 缺少起始格 {pos} 上的棋子")
    if sorted(order) != [1, 2, 3, 4, 5, 6]:
        raise ValueError("layout 必须为 1..6 各占据一个起始格")
    return order


def parse_state_message(msg: Mapping[str, Any]) -> OnlineMatchStateView:
    if msg.get("type") != "state":
        raise ValueError("不是 state 消息")
    current = PlayerColor(str(msg["current_player"]))
    your = PlayerColor(str(msg["your_role"]))
    winner_raw = msg.get("winner")
    winner = None if winner_raw is None else PlayerColor(str(winner_raw))
    board_rows = msg.get("board")
    if not isinstance(board_rows, list):
        raise ValueError("state.board 必须是二维数组")
    board = tuple(tuple(int(x) for x in row) for row in board_rows)
    legal_raw = msg.get("legal_moves")
    legal: tuple[Move, ...] | None
    if legal_raw is None:
        legal = None
    else:
        if not isinstance(legal_raw, list):
            raise ValueError("state.legal_moves 必须是数组或省略")
        legal = tuple(wire_move_to_move(m) for m in legal_raw)
    return OnlineMatchStateView(
        turn_index=int(msg["turn_index"]),
        current_player=current,
        dice_roll=int(msg["dice_roll"]),
        board=board,
        winner=winner,
        your_role=your,
        legal_move_count=int(msg["legal_move_count"]),
        legal_moves=legal,
    )


class OnlineMatchClient:
    """
    已连接并完成 `welcome` 握手后的会话。

    用法::

        client = await OnlineMatchClient.connect("127.0.0.1", 8765)
        try:
            await client.send_layout([1, 2, 3, 4, 5, 6])
            msg = await client.recv_message()
            ...
        finally:
            await client.aclose()
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        welcome: dict[str, Any],
    ) -> None:
        self._reader = reader
        self._writer = writer
        self.welcome = welcome
        if welcome.get("type") != "welcome":
            raise OnlineMatchProtocolError("首条消息必须是 welcome")
        pv = welcome.get("protocol_version")
        if pv is not None and int(pv) != PROTOCOL_VERSION:
            raise OnlineMatchProtocolError(
                f"不支持的 protocol_version: {pv!r}（客户端期望 {PROTOCOL_VERSION}）"
            )
        self.role = PlayerColor(str(welcome["role"]))

    @classmethod
    async def connect(cls, host: str, port: int) -> OnlineMatchClient:
        reader, writer = await asyncio.open_connection(host, port)
        try:
            first = await cls._read_obj(reader)
        except BaseException:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            raise
        if first.get("type") != "welcome":
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass
            raise OnlineMatchProtocolError(f"期望 welcome，收到: {first.get('type')!r}")
        return cls(reader, writer, first)

    @staticmethod
    async def _read_obj(reader: asyncio.StreamReader) -> dict[str, Any]:
        raw = await reader.readline()
        if not raw:
            raise ConnectionError("连接已关闭")
        line = raw.decode("utf-8").strip()
        if not line:
            raise ValueError("收到空行")
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError("消息必须是 JSON 对象")
        return data

    async def send_obj(self, obj: dict[str, Any]) -> None:
        payload = json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n"
        self._writer.write(payload.encode("utf-8"))
        await self._writer.drain()

    async def recv_message(self) -> dict[str, Any]:
        return await self._read_obj(self._reader)

    async def send_layout(self, order: Sequence[int]) -> None:
        await self.send_obj({"type": "layout", "order": [int(x) for x in order]})

    async def send_move(self, piece_number: int, to: Position) -> None:
        row, col = to
        await self.send_obj(
            {"type": "move", "piece_number": int(piece_number), "to": [int(row), int(col)]}
        )

    async def send_move_obj(self, move: Move) -> None:
        await self.send_move(move.piece_number, move.to_position)

    async def send_pass(self) -> None:
        await self.send_obj({"type": "pass"})

    async def aclose(self) -> None:
        self._writer.close()
        try:
            await self._writer.wait_closed()
        except Exception:
            pass
