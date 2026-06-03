"""将游戏局面编码为神经网络输入。

神经网络不能直接理解 Piece、Move 或棋盘对象，因此这里把一个局面转换为
15 张 5 × 5 的数字棋盘，也就是形状为 (15, 5, 5) 的浮点数组。
"""

from __future__ import annotations

import numpy as np

from ..engine import EinsteinGame, GameSnapshot, PlayerColor


# 15 个通道相当于 15 张重叠的透明棋盘。
STATE_CHANNELS = 15
BOARD_SIZE = EinsteinGame.BOARD_SIZE
# 通道 0-5：红方 1-6 号棋子；通道 6-11：蓝方 1-6 号棋子。
RED_OFFSET = 0
BLUE_OFFSET = 6
# 通道 12：当前行动方；通道 13：骰子；通道 14：合法候选棋子位置。
CURRENT_PLAYER_CHANNEL = 12
DICE_CHANNEL = 13
LEGAL_SOURCE_CHANNEL = 14


def encode_state(snapshot: GameSnapshot) -> np.ndarray:
    """把一个局面快照转换为 (15, 5, 5) 的网络输入。"""
    state = np.zeros((STATE_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    # 每一枚编号棋子使用独立通道，网络才能区分骰子所对应的棋子编号。
    for row_index, row in enumerate(snapshot.board):
        for col_index, encoded_piece in enumerate(row):
            if encoded_piece > 0:
                channel = RED_OFFSET + encoded_piece - 1
                state[channel, row_index, col_index] = 1.0
            elif encoded_piece < 0:
                channel = BLUE_OFFSET + abs(encoded_piece) - 1
                state[channel, row_index, col_index] = 1.0

    # 同一个棋盘轮到红方或蓝方行动时意义不同，因此需要显式记录当前玩家。
    player_value = 1.0 if snapshot.current_player is PlayerColor.RED else -1.0
    state[CURRENT_PLAYER_CHANNEL, :, :] = player_value
    # 将骰子除以 6，缩放到较稳定的 0-1 范围。
    state[DICE_CHANNEL, :, :] = snapshot.dice_roll / 6.0

    # 标记本回合允许移动的棋子来源位置，帮助网络聚焦当前候选棋子。
    for move in snapshot.legal_moves:
        row, col = move.from_position
        state[LEGAL_SOURCE_CHANNEL, row, col] = 1.0

    return state


# 下一步阅读：einstein_chess/training/action_codec.py
