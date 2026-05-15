"""Tests for `einstein_chess.online_match_client` (wire helpers + live TCP smoke)."""

from __future__ import annotations

import asyncio
import random
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from einstein_chess.engine import EinsteinGame, Move, PlayerColor
from einstein_chess.online_match_client import (
    OnlineMatchClient,
    layout_order_from_mapping,
    move_to_wire_dict,
    parse_state_message,
    wire_move_to_move,
)


def test_layout_order_from_default_red() -> None:
    layout = EinsteinGame.default_layout_for(PlayerColor.RED)
    assert layout_order_from_mapping(layout, PlayerColor.RED) == [1, 2, 3, 4, 5, 6]


def test_wire_move_roundtrip() -> None:
    m = Move(
        color=PlayerColor.RED,
        piece_number=3,
        from_position=(0, 0),
        to_position=(1, 0),
    )
    w = move_to_wire_dict(m)
    assert wire_move_to_move(w) == m


def test_parse_state_message() -> None:
    msg = {
        "type": "state",
        "turn_index": 2,
        "current_player": "red",
        "dice_roll": 4,
        "board": [[0, 0, 0, 0, 0]] * 5,
        "winner": None,
        "your_role": "red",
        "legal_move_count": 1,
        "legal_moves": [
            {
                "color": "red",
                "piece_number": 4,
                "from": [0, 2],
                "to": [1, 2],
            }
        ],
    }
    view = parse_state_message(msg)
    assert view.is_my_turn()
    assert view.legal_moves is not None and len(view.legal_moves) == 1


async def _bot_loop(client: OnlineMatchClient, seed: int) -> None:
    rng = random.Random(seed)
    while True:
        msg = await client.recv_message()
        mtype = msg.get("type")
        if mtype == "game_over":
            return
        if mtype == "error":
            raise AssertionError(msg)
        if mtype != "state":
            continue
        view = parse_state_message(msg)
        if not view.is_my_turn() or view.legal_moves is None:
            continue
        if not view.legal_moves:
            await client.send_pass()
        else:
            await client.send_move_obj(rng.choice(list(view.legal_moves)))


async def _run_two_bots(port: int) -> None:
    red = await OnlineMatchClient.connect("127.0.0.1", port)
    blue = await OnlineMatchClient.connect("127.0.0.1", port)
    try:
        await asyncio.gather(
            red.send_layout([1, 2, 3, 4, 5, 6]),
            blue.send_layout([1, 2, 3, 4, 5, 6]),
        )
        await asyncio.gather(_bot_loop(red, 11), _bot_loop(blue, 22))
    finally:
        await asyncio.gather(red.aclose(), blue.aclose(), return_exceptions=True)


def test_online_match_server_two_random_bots() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    proc = subprocess.Popen(
        [
            sys.executable,
            str(ROOT / "online_match.py"),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--seed",
            "42",
        ],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.35)
        asyncio.run(_run_two_bots(port))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=3)
