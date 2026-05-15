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
- behaviour_count: 本局已经公开广播过的行为数量（双方可见）
- last_behaviour: 上一个行为（双方可见）；开局首个 state 为 null。
  字段包含 turn_index/color/dice_roll/legal_move_count/action/move。

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
- {\"type\": \"game_over\", \"winner\": \"red\"|\"blue\"|null, \"reason\": \"...\", \"behaviours\": [...]}

================================================================================
五、启动示例
================================================================================
    python online_match.py --host 0.0.0.0 --port 8765

红方与蓝方各用任意 TCP 客户端（如 nc、自写脚本）连接同一地址；先连为红，后连为蓝。
"""

from __future__ import annotations

import argparse
import asyncio
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np

from einstein_chess.engine import EinsteinGame, Move, PlayerColor, Position
from einstein_chess.training.action_codec import ACTION_SIZE, move_to_action_id
from einstein_chess.training.state_encoder import encode_state


PROTOCOL_VERSION = 1
LAYOUT_MODES = ("default", "random", "agent")

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


@dataclass(frozen=True)
class OnlineMatchResult:
    game_index: int
    winner: PlayerColor | None
    reason: str
    turn_index: int
    step_count: int
    final_board: tuple[tuple[int, ...], ...]
    red_layout: dict[int, Position]
    blue_layout: dict[int, Position]
    behaviours: tuple[OnlineBehaviour, ...]
    samples: tuple[OnlineTrainingSample, ...]


@dataclass(frozen=True)
class OnlineBehaviour:
    turn_index: int
    color: PlayerColor
    dice_roll: int
    legal_move_count: int
    action: str
    move: Move | None


@dataclass(frozen=True)
class OnlineTrainingSample:
    state: np.ndarray
    policy: np.ndarray
    player: PlayerColor
    dice_roll: int
    action_id: int
    turn_index: int
    game_id: int
    winner: PlayerColor | None


def _wire_state_for(
    game: EinsteinGame,
    viewer: PlayerColor,
    *,
    game_index: int = 1,
    total_games: int = 1,
    layout_mode: str = "agent",
    behaviours: Sequence[OnlineBehaviour] = (),
) -> dict[str, Any]:
    legal = game.get_legal_moves()
    show_moves = game.winner is None and viewer == game.current_player
    payload: dict[str, Any] = {
        "type": "state",
        "game_index": game_index,
        "total_games": total_games,
        "layout_mode": layout_mode,
        "turn_index": game.turn_index,
        "current_player": _str_from_color(game.current_player),
        "dice_roll": game.dice_roll,
        "board": [list(row) for row in game.encode_board()],
        "winner": None if game.winner is None else _str_from_color(game.winner),
        "your_role": _str_from_color(viewer),
        "legal_move_count": len(legal),
        "behaviour_count": len(behaviours),
        "last_behaviour": None
        if not behaviours
        else _behaviour_to_wire(behaviours[-1]),
    }
    if show_moves:
        payload["legal_moves"] = [_move_to_wire(m) for m in legal]
    return payload


def _behaviour_to_wire(behaviour: OnlineBehaviour) -> dict[str, Any]:
    return {
        "turn_index": behaviour.turn_index,
        "color": behaviour.color.value,
        "dice_roll": behaviour.dice_roll,
        "legal_move_count": behaviour.legal_move_count,
        "action": behaviour.action,
        "move": None
        if behaviour.move is None
        else _move_to_wire(behaviour.move),
    }


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
    *,
    layout_mode: str = "agent",
    game_index: int = 1,
    total_games: int = 1,
    max_turns: int = 1000,
    move_timeout: float | None = None,
) -> OnlineMatchResult:
    layouts: dict[PlayerColor, dict[int, Position]] = {}

    async def recv_layout(proto: _JsonLineProtocol, color: PlayerColor) -> None:
        while True:
            msg = await proto.read_obj()
            if msg.get("type") != "layout":
                await proto.send_obj(
                    {"type": "error", "message": "布阵阶段请发送 {\"type\":\"layout\", \"order\":[...]}"}
                )
                continue
            if layout_mode != "agent":
                return
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

    if layout_mode == "default":
        layouts = {
            PlayerColor.RED: EinsteinGame.default_layout_for(PlayerColor.RED),
            PlayerColor.BLUE: EinsteinGame.default_layout_for(PlayerColor.BLUE),
        }
    elif layout_mode == "random":
        layouts = {
            PlayerColor.RED: EinsteinGame.random_layout_for(PlayerColor.RED, rng=rng),
            PlayerColor.BLUE: EinsteinGame.random_layout_for(PlayerColor.BLUE, rng=rng),
        }

    game = EinsteinGame(rng=rng)
    game.reset(
        red_layout=layouts[PlayerColor.RED],
        blue_layout=layouts[PlayerColor.BLUE],
    )
    step_count = 0
    behaviours: list[OnlineBehaviour] = []
    pending_samples: list[dict[str, Any]] = []

    async def broadcast_state() -> None:
        await asyncio.gather(
            red.send_obj(
                _wire_state_for(
                    game,
                    PlayerColor.RED,
                    game_index=game_index,
                    total_games=total_games,
                    layout_mode=layout_mode,
                    behaviours=behaviours,
                )
            ),
            blue.send_obj(
                _wire_state_for(
                    game,
                    PlayerColor.BLUE,
                    game_index=game_index,
                    total_games=total_games,
                    layout_mode=layout_mode,
                    behaviours=behaviours,
                )
            ),
        )

    async def finish(
        winner: PlayerColor | None,
        reason: str,
    ) -> OnlineMatchResult:
        win = winner
        over = {
            "type": "game_over",
            "game_index": game_index,
            "total_games": total_games,
            "winner": None if win is None else win.value,
            "reason": reason,
            "final": {
                "board": [list(row) for row in game.encode_board()],
                "turn_index": game.turn_index,
                "steps": step_count,
            },
            "behaviours": [_behaviour_to_wire(behaviour) for behaviour in behaviours],
        }
        await asyncio.gather(red.send_obj(over), blue.send_obj(over), return_exceptions=True)
        return OnlineMatchResult(
            game_index=game_index,
            winner=win,
            reason=reason,
            turn_index=game.turn_index,
            step_count=step_count,
            final_board=game.encode_board(),
            red_layout=dict(layouts[PlayerColor.RED]),
            blue_layout=dict(layouts[PlayerColor.BLUE]),
            behaviours=tuple(behaviours),
            samples=tuple(_finalize_training_samples(pending_samples, win)),
        )

    await broadcast_state()

    while game.winner is None:
        if step_count >= max_turns:
            return await finish(None, "max_turns_exceeded")

        color = game.current_player
        active = red if color is PlayerColor.RED else blue
        legal_moves = game.get_legal_moves()

        try:
            if move_timeout is None:
                msg = await active.read_obj()
            else:
                msg = await asyncio.wait_for(active.read_obj(), timeout=move_timeout)
        except TimeoutError:
            await active.send_obj({"type": "error", "message": "本回合思考超时"})
            return await finish(color.opponent, "move_timeout")

        mtype = msg.get("type")

        if not legal_moves:
            if mtype != "pass":
                await active.send_obj({"type": "error", "message": "当前无合法着法，只能发送 pass"})
                return await finish(color.opponent, "illegal_move")
            turn_index = game.turn_index
            dice_roll = game.dice_roll
            try:
                game.pass_turn()
            except Exception as exc:
                await active.send_obj({"type": "error", "message": str(exc)})
                return await finish(color.opponent, f"agent_error:{type(exc).__name__}")
            step_count += 1
            behaviours.append(
                OnlineBehaviour(
                    turn_index=turn_index,
                    color=color,
                    dice_roll=dice_roll,
                    legal_move_count=0,
                    action="pass",
                    move=None,
                )
            )
        elif mtype == "move":
            try:
                piece = int(msg["piece_number"])
                to_raw = msg["to"]
                to_pos = (int(to_raw[0]), int(to_raw[1]))
            except Exception:
                await active.send_obj({"type": "error", "message": "move 需要 piece_number 与 to:[row,col]"})
                return await finish(color.opponent, "protocol_error")
            chosen = _find_matching_move(game, piece, to_pos)
            if chosen is None:
                await active.send_obj({"type": "error", "message": "不合法着法（与引擎 legal_moves 不一致）"})
                return await finish(color.opponent, "illegal_move")
            chosen_turn_index = game.turn_index
            chosen_dice_roll = game.dice_roll
            snapshot = game.snapshot()
            pending_samples.append(
                {
                    "state": encode_state(snapshot),
                    "policy": _one_hot_policy(move_to_action_id(chosen)),
                    "player": color,
                    "dice_roll": chosen_dice_roll,
                    "action_id": move_to_action_id(chosen),
                    "turn_index": chosen_turn_index,
                    "game_id": game_index,
                }
            )
            try:
                game.apply_move(chosen)
            except Exception as exc:
                await active.send_obj({"type": "error", "message": str(exc)})
                return await finish(color.opponent, f"agent_error:{type(exc).__name__}")
            behaviours.append(
                OnlineBehaviour(
                    turn_index=chosen_turn_index,
                    color=color,
                    dice_roll=chosen_dice_roll,
                    legal_move_count=len(legal_moves),
                    action="move",
                    move=chosen,
                )
            )
            step_count += 1
        else:
            await active.send_obj({"type": "error", "message": "未知消息类型；请发 move 或 pass"})
            return await finish(color.opponent, "protocol_error")

        if game.winner is not None:
            return await finish(game.winner, _win_reason(game.move_history[-1]))
        await broadcast_state()

    return await finish(game.winner, "finished")


def _win_reason(move: Move) -> str:
    if move.to_position == move.color.goal:
        return "goal"
    return "capture_all"


def _one_hot_policy(action_id: int) -> np.ndarray:
    policy = np.zeros((ACTION_SIZE,), dtype=np.float32)
    policy[action_id] = 1.0
    return policy


def _finalize_training_samples(
    pending_samples: Sequence[dict[str, Any]],
    winner: PlayerColor | None,
) -> list[OnlineTrainingSample]:
    samples: list[OnlineTrainingSample] = []
    for sample in pending_samples:
        samples.append(
            OnlineTrainingSample(
                state=sample["state"],
                policy=sample["policy"],
                player=sample["player"],
                dice_roll=sample["dice_roll"],
                action_id=sample["action_id"],
                turn_index=sample["turn_index"],
                game_id=sample["game_id"],
                winner=winner,
            )
        )
    return samples


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

    def __init__(
        self,
        rng: random.Random,
        *,
        games: int = 1,
        layout_mode: str = "agent",
        max_turns: int = 1000,
        move_timeout: float | None = None,
        output: Path | None = None,
        record_jsonl: Path | None = None,
        dataset_output: Path | None = None,
    ) -> None:
        self._rng = rng
        self._games = games
        self._layout_mode = layout_mode
        self._max_turns = max_turns
        self._move_timeout = move_timeout
        self._output = output
        self._record_jsonl = record_jsonl
        self._dataset_output = dataset_output
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
                        "total_games": self._games,
                        "layout_mode": self._layout_mode,
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
                        "total_games": self._games,
                        "layout_mode": self._layout_mode,
                        "peer": str(addr),
                    }
                )
                try:
                    await self._run_series(red, proto)
                finally:
                    if done_fut is not None and not done_fut.done():
                        done_fut.set_result(None)
                return

        if session_done is not None:
            await session_done

    async def _run_series(
        self,
        red: _JsonLineProtocol,
        blue: _JsonLineProtocol,
    ) -> None:
        results: list[OnlineMatchResult] = []
        for game_index in range(1, self._games + 1):
            try:
                result = await _run_match(
                    red,
                    blue,
                    self._rng,
                    layout_mode=self._layout_mode,
                    game_index=game_index,
                    total_games=self._games,
                    max_turns=self._max_turns,
                    move_timeout=self._move_timeout,
                )
            except ConnectionError as exc:
                print(f"第 {game_index} 局连接中断: {exc}", file=sys.stderr, flush=True)
                break
            results.append(result)
            winner = "none" if result.winner is None else result.winner.value
            print(
                f"[{game_index}/{self._games}] winner={winner} "
                f"reason={result.reason} steps={result.step_count}",
                file=sys.stderr,
                flush=True,
            )

        if results:
            _print_series_summary(results)
            if self._output is not None:
                _write_results_csv(self._output, self._layout_mode, results)
            if self._record_jsonl is not None:
                _write_records_jsonl(self._record_jsonl, self._layout_mode, results)
            if self._dataset_output is not None:
                _write_training_dataset_npz(self._dataset_output, results)


def _print_series_summary(results: list[OnlineMatchResult]) -> None:
    red_wins = sum(1 for result in results if result.winner is PlayerColor.RED)
    blue_wins = sum(1 for result in results if result.winner is PlayerColor.BLUE)
    draws = sum(1 for result in results if result.winner is None)
    total_steps = sum(result.step_count for result in results)
    total = len(results)
    reasons: dict[str, int] = {}
    for result in results:
        reasons[result.reason] = reasons.get(result.reason, 0) + 1
    print("=" * 56, file=sys.stderr)
    print(f"completed_games: {total}", file=sys.stderr)
    print(f"red_wins: {red_wins}", file=sys.stderr)
    print(f"blue_wins: {blue_wins}", file=sys.stderr)
    print(f"draws_or_unfinished: {draws}", file=sys.stderr)
    print(f"red_win_rate: {red_wins / total:.4f}", file=sys.stderr)
    print(f"blue_win_rate: {blue_wins / total:.4f}", file=sys.stderr)
    print(f"average_steps: {total_steps / total:.2f}", file=sys.stderr)
    print(f"reasons: {reasons}", file=sys.stderr)


def _write_results_csv(
    output_path: Path,
    layout_mode: str,
    results: list[OnlineMatchResult],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        fieldnames = (
            "game_index",
            "layout_mode",
            "winner",
            "reason",
            "turn_index",
            "steps",
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "game_index": result.game_index,
                    "layout_mode": layout_mode,
                    "winner": "none" if result.winner is None else result.winner.value,
                    "reason": result.reason,
                    "turn_index": result.turn_index,
                    "steps": result.step_count,
                }
            )
    print(f"saved_results: {output_path}", file=sys.stderr, flush=True)


def _write_records_jsonl(
    output_path: Path,
    layout_mode: str,
    results: list[OnlineMatchResult],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for result in results:
            record = {
                "game_index": result.game_index,
                "layout_mode": layout_mode,
                "winner": None if result.winner is None else result.winner.value,
                "reason": result.reason,
                "turn_index": result.turn_index,
                "steps": result.step_count,
                "red_layout": _layout_order_for(PlayerColor.RED, result.red_layout),
                "blue_layout": _layout_order_for(PlayerColor.BLUE, result.blue_layout),
                "final_board": [list(row) for row in result.final_board],
                "behaviours": [
                    _behaviour_to_wire(behaviour)
                    for behaviour in result.behaviours
                ],
            }
            file.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
    print(f"saved_records_jsonl: {output_path}", file=sys.stderr, flush=True)


def _write_training_dataset_npz(
    output_path: Path,
    results: list[OnlineMatchResult],
) -> None:
    samples = [sample for result in results for sample in result.samples]
    if not samples:
        print("skip_dataset_output: no move samples were recorded", file=sys.stderr)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        states=np.stack([sample.state for sample in samples]).astype(np.float32),
        policies=np.stack([sample.policy for sample in samples]).astype(np.float32),
        values=np.asarray([_sample_value(sample) for sample in samples], dtype=np.float32),
        players=np.asarray([_player_code(sample.player) for sample in samples], dtype=np.int8),
        dice_rolls=np.asarray([sample.dice_roll for sample in samples], dtype=np.int8),
        action_ids=np.asarray([sample.action_id for sample in samples], dtype=np.int8),
        game_ids=np.asarray([sample.game_id for sample in samples], dtype=np.int32),
        turn_indices=np.asarray([sample.turn_index for sample in samples], dtype=np.int32),
        winners=np.asarray([_player_code(sample.winner) for sample in samples], dtype=np.int8),
    )
    print(
        f"saved_training_dataset: {output_path} samples={len(samples)}",
        file=sys.stderr,
        flush=True,
    )


def _layout_order_for(
    color: PlayerColor,
    layout: dict[int, Position],
) -> list[int]:
    order: list[int] = []
    for position in color.start_positions:
        for number, piece_position in layout.items():
            if piece_position == position:
                order.append(int(number))
                break
        else:
            raise ValueError(f"layout missing position {position}")
    return order


def _sample_value(sample: OnlineTrainingSample) -> float:
    if sample.winner is None:
        return 0.0
    return 1.0 if sample.player is sample.winner else -1.0


def _player_code(player: PlayerColor | None) -> int:
    if player is PlayerColor.RED:
        return 1
    if player is PlayerColor.BLUE:
        return -1
    return 0


async def _main_async(
    host: str,
    port: int,
    seed: int | None,
    *,
    games: int = 1,
    layout_mode: str = "agent",
    max_turns: int = 1000,
    move_timeout: float | None = None,
    output: Path | None = None,
    record_jsonl: Path | None = None,
    dataset_output: Path | None = None,
) -> None:
    rng = random.Random(seed)
    lobby = MatchLobby(
        rng,
        games=games,
        layout_mode=layout_mode,
        max_turns=max_turns,
        move_timeout=move_timeout,
        output=output,
        record_jsonl=record_jsonl,
        dataset_output=dataset_output,
    )
    server = await asyncio.start_server(
        lambda r, w: asyncio.create_task(_handle_client(r, w, lobby)),
        host=host,
        port=port,
    )
    addrs = ", ".join(str(sock.getsockname()) for sock in server.sockets or [])
    print(f"爱恩斯坦棋在线服务已启动: {addrs}", file=sys.stderr)
    print("先接入者为红方，第二个接入者为蓝方。", file=sys.stderr)
    print(
        f"计划局数: {games}；开局模式: {layout_mode}；"
        f"最大回合: {max_turns}；单步超时: {move_timeout or '不限'}",
        file=sys.stderr,
    )
    async with server:
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="爱恩斯坦棋 TCP 在线对战服务")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址")
    parser.add_argument("--port", type=int, default=8765, help="监听端口")
    parser.add_argument("--seed", type=int, default=None, help="随机数种子（骰子可复现）")
    parser.add_argument("--games", type=int, default=1, help="连续对战局数，例如 100")
    parser.add_argument(
        "--layout",
        choices=LAYOUT_MODES,
        default="agent",
        help="开局模式：default=平台默认；random=平台随机；agent=双方客户端自定义",
    )
    parser.add_argument("--max-turns", type=int, default=1000, help="单局最大步数，超过判未完成")
    parser.add_argument(
        "--move-timeout",
        type=float,
        default=None,
        help="单步最大等待秒数，超时该方负；默认不限制",
    )
    parser.add_argument("--output", type=Path, default=None, help="可选：保存每局结果 CSV")
    parser.add_argument(
        "--record-jsonl",
        type=Path,
        default=None,
        help="可选：保存完整原始对局记录 JSONL（一行一局，含布局和 behaviours）",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=None,
        help="可选：保存可训练 NPZ 数据集（one-hot 策略标签，兼容训练脚本）",
    )
    args = parser.parse_args()
    if args.games <= 0:
        parser.error("--games must be positive")
    if args.max_turns <= 0:
        parser.error("--max-turns must be positive")
    if args.move_timeout is not None and args.move_timeout <= 0:
        parser.error("--move-timeout must be positive")
    try:
        asyncio.run(
            _main_async(
                args.host,
                args.port,
                args.seed,
                games=args.games,
                layout_mode=args.layout,
                max_turns=args.max_turns,
                move_timeout=args.move_timeout,
                output=args.output,
                record_jsonl=args.record_jsonl,
                dataset_output=args.dataset_output,
            )
        )
    except KeyboardInterrupt:
        print("\n已停止服务。", file=sys.stderr)


if __name__ == "__main__":
    main()
