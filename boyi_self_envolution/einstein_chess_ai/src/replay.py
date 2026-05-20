"""Text replay for saved game records."""

from __future__ import annotations

import argparse
import time

from src.utils.serialization import load_json


def print_board(board: list[list[int]]) -> None:
    """Print a signed board."""
    for row in board:
        print(" ".join(" . " if v == 0 else (f"R{v}" if v > 0 else f"B{-v}") for v in row))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved game record in the terminal.")
    parser.add_argument("--record", required=True)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    record = load_json(args.record)
    print(f"game={record['game_id']} first={record['first_player']}")
    print_board(record["initial_board"])
    for move in record["moves"]:
        time.sleep(args.delay)
        print(f"\nstep={move['step']} player={move['player']} dice={move['dice']} action={move['action']}")
        print_board(move["board_after"])
    print(f"winner={record['winner']} reason={record['win_reason']}")


if __name__ == "__main__":
    main()

