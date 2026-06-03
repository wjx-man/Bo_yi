"""在游戏走法 Move 与神经网络 18 维动作空间之间进行转换。

双方各有 6 枚编号棋子，每枚棋子都有 3 个前进方向，因此网络固定输出：

6 个棋子编号 × 3 个方向 = 18 个动作分数
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..engine import GameSnapshot, Move, PlayerColor


ACTION_SIZE = 18
_DIRECTIONS_PER_PIECE = 3
# 同一个 direction_index 对红蓝双方表示相对前进方向，而不是相同绝对坐标方向。
_ACTION_DELTAS = {
    PlayerColor.RED: ((0, 1), (1, 0), (1, 1)),
    PlayerColor.BLUE: ((0, -1), (-1, 0), (-1, -1)),
}


def move_to_action_id(move: Move) -> int:
    """将一条具体走法转换为 0-17 的动作编号。"""
    delta = (
        move.to_position[0] - move.from_position[0],
        move.to_position[1] - move.from_position[1],
    )
    try:
        direction_index = _ACTION_DELTAS[move.color].index(delta)
    except ValueError as exc:
        raise ValueError(f"Move direction is invalid for {move.color.value}.") from exc
    # 例如 3 号棋子第 2 个方向：action_id = (3 - 1) * 3 + 2 = 8。
    return (move.piece_number - 1) * _DIRECTIONS_PER_PIECE + direction_index


def action_id_to_move(action_id: int, snapshot: GameSnapshot) -> Move:
    """将动作编号还原为当前局面中的具体 Move，并检查它是否合法。"""
    if not 0 <= action_id < ACTION_SIZE:
        raise ValueError("action_id must be in [0, 17].")

    # 整除得到棋子编号，取余得到该棋子的方向编号。
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
    """生成 18 维合法动作掩码：合法位置为 1，非法位置为 0。"""
    mask = np.zeros(ACTION_SIZE, dtype=np.float32)
    for move in snapshot.legal_moves:
        mask[move_to_action_id(move)] = 1.0
    return mask


def policy_dict_to_vector(policy: Mapping[Move, float]) -> np.ndarray:
    """将“Move -> 概率”字典转换为网络训练使用的 18 维策略向量。"""
    vector = np.zeros(ACTION_SIZE, dtype=np.float32)
    for move, probability in policy.items():
        vector[move_to_action_id(move)] += float(probability)

    # 再次归一化，确保所有动作概率之和为 1。
    total = float(vector.sum())
    if total > 0:
        vector /= total
    return vector


def policy_from_visit_counts(visit_counts: Mapping[Move, int]) -> np.ndarray:
    """直接将搜索访问次数转换为 18 维策略答案。"""
    total_visits = sum(visit_counts.values())
    if total_visits <= 0:
        return policy_dict_to_vector({move: 0.0 for move in visit_counts})
    return policy_dict_to_vector(
        {move: visits / total_visits for move, visits in visit_counts.items()}
    )


def _find_piece(
    snapshot: GameSnapshot, color: PlayerColor, piece_number: int
) -> tuple[int, int] | None:
    """在整数编码棋盘中寻找指定颜色和编号的棋子。"""
    encoded_piece = piece_number if color is PlayerColor.RED else -piece_number
    for row_index, row in enumerate(snapshot.board):
        for col_index, value in enumerate(row):
            if value == encoded_piece:
                return (row_index, col_index)
    return None


# 下一步阅读：einstein_chess/training/model.py
