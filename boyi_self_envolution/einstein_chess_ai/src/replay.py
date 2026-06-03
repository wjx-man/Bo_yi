"""在终端中按步骤回放自我博弈保存的棋局记录。"""

from __future__ import annotations

import argparse
import time

from src.utils.serialization import load_json


def print_board(board: list[list[int]]) -> None:
    """打印带颜色和编号的棋盘：正数为红方，负数为蓝方。"""
    for row in board:
        print(" ".join(" . " if v == 0 else (f"R{v}" if v > 0 else f"B{-v}") for v in row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved game record in the terminal.")
    parser.add_argument("--record", required=True)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    record = load_json(args.record)
    # 棋局记录包含初始棋盘和每一步走完后的 board_after。
    print(f"game={record['game_id']} first={record['first_player']}")
    print_board(record["initial_board"])
    for move in record["moves"]:
        # delay 控制两步之间的停顿时间，使终端输出更容易观察。
        time.sleep(args.delay)
        print(f"\nstep={move['step']} player={move['player']} dice={move['dice']} action={move['action']}")
        print_board(move["board_after"])
    print(f"winner={record['winner']} reason={record['win_reason']}")


if __name__ == "__main__":
    main()

