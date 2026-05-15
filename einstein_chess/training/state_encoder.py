from __future__ import annotations

import numpy as np

from ..engine import EinsteinGame, GameSnapshot, PlayerColor


STATE_CHANNELS = 15
BOARD_SIZE = EinsteinGame.BOARD_SIZE
RED_OFFSET = 0
BLUE_OFFSET = 6
CURRENT_PLAYER_CHANNEL = 12
DICE_CHANNEL = 13
LEGAL_SOURCE_CHANNEL = 14


def encode_state(snapshot: GameSnapshot) -> np.ndarray:
    state = np.zeros((STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    for row_index, row in enumerate(snapshot.board):
        for col_index, encoded_piece in enumerate(row):
            if encoded_piece > 0:
                channel = RED_OFFSET + encoded_piece - 1
                state[channel, row_index, col_index] = 1.0
            elif encoded_piece < 0:
                channel = BLUE_OFFSET + abs(encoded_piece) - 1
                state[channel, row_index, col_index] = 1.0

    player_value = 1.0 if snapshot.current_player is PlayerColor.RED else -1.0
    state[CURRENT_PLAYER_CHANNEL, :, :] = player_value
    state[DICE_CHANNEL, :, :] = snapshot.dice_roll / 6.0

    for move in snapshot.legal_moves:
        row, col = move.from_position
        state[LEGAL_SOURCE_CHANNEL, row, col] = 1.0

    return state
