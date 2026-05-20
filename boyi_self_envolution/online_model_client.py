#!/usr/bin/env python3
"""Persistent JSONL client for the Einstein chess online judge.

The judge owns dice rolls, legal move generation, rule checking, game updates,
and multi-game scheduling. This client only mirrors each state into the local
EinsteinChessEnv, asks the existing model-backed Agent for an action, and sends
one move chosen from the judge-provided legal_moves list.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable, Iterable


PROTOCOL_VERSION = 1
ROOT = Path(__file__).resolve().parent
AI_ROOT = ROOT / "einstein_chess_ai"

ai_path = str(AI_ROOT)
if ai_path not in sys.path:
    sys.path.insert(0, ai_path)

from src.agents.base import Agent  # noqa: E402
from src.env.env import EinsteinChessEnv  # noqa: E402
from src.env.rules import ACTION_SIZE, BLUE, DELTAS, PIECE_IDS, RED, action_to_id, id_to_action  # noqa: E402


LogFn = Callable[[str], None]
WireMove = dict[str, Any]
WireState = dict[str, Any]


class JsonLineConnection:
    """Small UTF-8 JSON-per-line helper for the judge protocol."""

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
            raise ConnectionError("Server closed the connection.")
        data = json.loads(raw.decode("utf-8").strip())
        if not isinstance(data, dict):
            raise ValueError("Protocol message must be a JSON object.")
        return data

    async def close(self) -> None:
        self.writer.close()
        await self.writer.wait_closed()


def _resolve_checkpoint(checkpoint: str | Path | None) -> Path | None:
    if checkpoint is None:
        return None
    path = Path(checkpoint)
    if path.is_absolute():
        return path
    if (ROOT / path).exists():
        return ROOT / path
    return AI_ROOT / path


def make_local_model_agent(
    checkpoint: str | Path | None = ROOT / "weights" / "model.pt",
    *,
    device: str = "auto",
    mode: str = "greedy",
    temperature: float = 1.0,
) -> Agent:
    """Create the existing ActorCriticAgent for online play."""
    from src.agents.actor_critic_agent import ActorCriticAgent

    return ActorCriticAgent(
        checkpoint=_resolve_checkpoint(checkpoint),
        device=device,
        mode=mode,
        temperature=temperature,
    )


def _normalize_layout_order(order: Iterable[int]) -> list[int]:
    normalized = [int(x) for x in order]
    if len(normalized) != 6 or sorted(normalized) != [1, 2, 3, 4, 5, 6]:
        raise ValueError("layout_order must be a permutation of 1..6")
    return normalized


def _state_to_env(state: WireState) -> EinsteinChessEnv:
    """Mirror a judge state packet into EinsteinChessEnv."""
    env = EinsteinChessEnv()
    positions = {
        RED: {pid: None for pid in PIECE_IDS},
        BLUE: {pid: None for pid in PIECE_IDS},
    }

    for row_index, row in enumerate(state["board"]):
        for col_index, cell in enumerate(row):
            value = int(cell)
            if value > 0:
                positions[RED][value] = (row_index, col_index)
            elif value < 0:
                positions[BLUE][-value] = (row_index, col_index)

    env.positions = positions
    env.current_player = str(state["current_player"])
    env.dice = int(state["dice_roll"])
    env.winner = state.get("winner")
    env.step_count = max(0, int(state.get("turn_index", 1)) - 1)
    env.last_move = None
    return env


def _wire_move_to_action(move: WireMove) -> int | None:
    color = str(move["color"])
    piece_id = int(move["piece_number"])
    from_pos = tuple(int(x) for x in move["from"])
    to_pos = tuple(int(x) for x in move["to"])
    delta = (to_pos[0] - from_pos[0], to_pos[1] - from_pos[1])
    try:
        direction = DELTAS[color].index(delta)
    except (KeyError, ValueError):
        return None
    return action_to_id(piece_id, direction)


def _action_to_wire_move(action: int, legal_moves: Iterable[WireMove]) -> WireMove | None:
    for move in legal_moves:
        if _wire_move_to_action(move) == int(action):
            return move
    return None


def _best_protocol_move(agent: Agent, env: EinsteinChessEnv, legal_moves: list[WireMove]) -> WireMove:
    """Select an agent action and map it to one of the judge-supplied moves."""
    if not legal_moves:
        raise ValueError("No legal moves are available in the state packet.")

    chosen_action = int(agent.select_action(env))
    chosen_move = _action_to_wire_move(chosen_action, legal_moves)
    if chosen_move is not None:
        return chosen_move

    legal_actions = [
        action
        for action in (_wire_move_to_action(move) for move in legal_moves)
        if action is not None
    ]
    if not legal_actions:
        return legal_moves[0]

    diagnostics = getattr(agent, "last_diagnostics", None)
    probabilities = getattr(diagnostics, "probabilities", None)
    if probabilities is not None and len(probabilities) >= ACTION_SIZE:
        fallback_action = max(legal_actions, key=lambda action: float(probabilities[action]))
    else:
        fallback_action = legal_actions[0]

    return _action_to_wire_move(fallback_action, legal_moves) or legal_moves[0]


def _move_payload(move: WireMove) -> dict[str, Any]:
    return {
        "type": "move",
        "piece_number": int(move["piece_number"]),
        "to": [int(move["to"][0]), int(move["to"][1])],
    }


async def play_local_model_series(
    host: str = "127.0.0.1",
    port: int = 8765,
    *,
    games: int | None = None,
    checkpoint: str | Path | None = ROOT / "weights" / "model.pt",
    agent: Agent | None = None,
    layout_order: list[int] | tuple[int, ...] = (1, 2, 3, 4, 5, 6),
    device: str = "auto",
    mode: str = "greedy",
    temperature: float = 1.0,
    log: LogFn | None = print,
) -> list[dict[str, Any]]:
    """Connect once and play a persistent multi-game match."""
    layout = _normalize_layout_order(layout_order)
    shared_agent = agent or make_local_model_agent(
        checkpoint=checkpoint,
        device=device,
        mode=mode,
        temperature=temperature,
    )

    def emit(message: str) -> None:
        if log is not None:
            log(message)

    emit(f"Connecting to {host}:{port} ...")
    reader, writer = await asyncio.open_connection(host, port)
    conn = JsonLineConnection(reader, writer)
    role: str | None = None
    results: list[dict[str, Any]] = []

    try:
        welcome = await conn.read_obj()
        if welcome.get("type") != "welcome":
            raise RuntimeError(f"Expected welcome packet, got: {welcome}")
        if int(welcome.get("protocol_version", -1)) != PROTOCOL_VERSION:
            raise RuntimeError(f"Unsupported protocol version: {welcome}")

        role = str(welcome["role"])
        server_total = int(welcome.get("total_games") or 1)
        target_games = server_total if games is None else min(int(games), server_total)
        layout_mode = welcome.get("layout_mode", "unknown")
        emit(
            f"Connected as {role}. total_games={server_total} "
            f"target_games={target_games} layout_mode={layout_mode}"
        )

        for local_game_index in range(1, target_games + 1):
            moves_sent = 0
            emit(f"=== game {local_game_index}/{target_games}: sending layout {layout} ===")
            await conn.send_obj({"type": "layout", "order": layout})

            while True:
                msg = await conn.read_obj()
                msg_type = msg.get("type")

                if msg_type == "game_over":
                    msg["role"] = role
                    msg["moves_sent"] = moves_sent
                    results.append(msg)
                    emit(
                        f"game_over game={msg.get('game_index', local_game_index)} "
                        f"winner={msg.get('winner')} reason={msg.get('reason')} "
                        f"moves_sent={moves_sent}"
                    )
                    break

                if msg_type == "error":
                    emit(f"server error: {msg.get('message')}")
                    continue

                if msg_type != "state":
                    emit(f"Ignoring packet: {msg}")
                    continue

                if msg.get("winner") is not None:
                    continue
                if msg.get("your_role") != role or msg.get("current_player") != role:
                    continue

                legal_moves_raw = msg.get("legal_moves")
                if legal_moves_raw is None:
                    continue

                legal_moves = list(legal_moves_raw)
                if not legal_moves:
                    emit(
                        f"pass game={msg.get('game_index')} "
                        f"turn={msg.get('turn_index')} dice={msg.get('dice_roll')}"
                    )
                    await conn.send_obj({"type": "pass"})
                    continue

                env = _state_to_env(msg)
                move = _best_protocol_move(shared_agent, env, legal_moves)
                action_id = _wire_move_to_action(move)
                piece_id, direction = id_to_action(0 if action_id is None else action_id)
                emit(
                    f"move game={msg.get('game_index')} "
                    f"turn={msg.get('turn_index')} dice={msg.get('dice_roll')} "
                    f"piece={piece_id} direction={direction} to={move['to']}"
                )
                await conn.send_obj(_move_payload(move))
                moves_sent += 1

        wins = sum(1 for row in results if row.get("winner") == role)
        played = len(results)
        if played:
            emit(f"Series complete. games={played} wins={wins} win_rate={wins / played:.3f}")
        return results
    finally:
        await conn.close()


async def play_local_model_against_opponent(
    host: str = "127.0.0.1",
    port: int = 8765,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility wrapper that plays one game on the persistent protocol."""
    kwargs["games"] = 1
    results = await play_local_model_series(host, port, **kwargs)
    if not results:
        raise RuntimeError("No game_over packet was received.")
    return results[0]


def run_local_model_series(
    host: str = "127.0.0.1",
    port: int = 8765,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Synchronous wrapper around play_local_model_series()."""
    return asyncio.run(play_local_model_series(host, port, **kwargs))


def run_local_model_match(
    host: str = "127.0.0.1",
    port: int = 8765,
    **kwargs: Any,
) -> dict[str, Any]:
    """Synchronous wrapper that plays one game."""
    return asyncio.run(play_local_model_against_opponent(host, port, **kwargs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Play the Einstein chess online judge with a local model.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", default=str(ROOT / "weights" / "model.pt"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--layout",
        default="1,2,3,4,5,6",
        help="Comma-separated piece order for the six starting cells.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=None,
        help="Optional maximum games to play. Default uses welcome.total_games.",
    )
    args = parser.parse_args()
    layout = [int(x.strip()) for x in args.layout.split(",") if x.strip()]
    run_local_model_series(
        host=args.host,
        port=args.port,
        games=args.games,
        checkpoint=args.checkpoint,
        layout_order=layout,
        device=args.device,
        mode=args.mode,
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
