from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..engine import GameSnapshot, Move, PlayerColor


ACTION_SIZE = 18
_DIRECTIONS_PER_PIECE = 3
_ACTION_DELTAS = {
    PlayerColor.RED: ((0, 1), (1, 0), (1, 1)),
    PlayerColor.BLUE: ((0, -1), (-1, 0), (-1, -1)),
}


def move_to_action_id(move: Move) -> int:
    delta = (
        move.to_position[0] - move.from_position[0],
        move.to_position[1] - move.from_position[1],
    )
    try:
        direction_index = _ACTION_DELTAS[move.color].index(delta)
    except ValueError as exc:
        raise ValueError(f"Move direction is invalid for {move.color.value}.") from exc
    return (move.piece_number - 1) * _DIRECTIONS_PER_PIECE + direction_index


def action_id_to_move(action_id: int, snapshot: GameSnapshot) -> Move:
    if not 0 <= action_id < ACTION_SIZE:
        raise ValueError("action_id must be in [0, 17].")

    piece_number = action_id // _DIRECTIONS_PER_PIECE + 1
    direction_index = action_id % _DIRECTIONS_PER_PIECE
    color = snapshot.current_player
    delta_row, delta_col = _ACTION_DELTAS[color][direction_index]

    from_position = _find_piece(snapshot, color, piece_number)
    if from_position is None:
        raise ValueError("The requested piece is not on the board.")

    candidate = Move(
        color=color,
        piece_number=piece_number,
        from_position=from_position,
        to_position=(from_position[0] + delta_row, from_position[1] + delta_col),
    )
    if candidate not in snapshot.legal_moves:
        raise ValueError("The requested action is not legal in this snapshot.")
    return candidate


def legal_action_mask(snapshot: GameSnapshot) -> np.ndarray:
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for move in snapshot.legal_moves:
        mask[move_to_action_id(move)] = 1.0
    return mask


def policy_dict_to_vector(policy: Mapping[Move, float]) -> np.ndarray:
    vector = np.zeros(ACTION_SIZE, dtype=np.float32)
    for move, probability in policy.items():
        vector[move_to_action_id(move)] += float(probability)

    total = float(vector.sum())
    if total > 0:
        vector /= total
    return vector


def policy_from_visit_counts(visit_counts: Mapping[Move, int]) -> np.ndarray:
    total_visits = sum(visit_counts.values())
    if total_visits <= 0:
        return policy_dict_to_vector({move: 0.0 for move in visit_counts})
    return policy_dict_to_vector(
        {move: visits / total_visits for move, visits in visit_counts.items()}
    )


def _find_piece(
    snapshot: GameSnapshot, color: PlayerColor, piece_number: int
) -> tuple[int, int] | None:
    encoded_piece = piece_number if color is PlayerColor.RED else -piece_number
    for row_index, row in enumerate(snapshot.board):
        for col_index, value in enumerate(row):
            if value == encoded_piece:
                return (row_index, col_index)
    return None
