"""验证骰子候选棋子和双方移动方向等基础规则。"""

from src.env.env import EinsteinChessEnv
from src.env.rules import BLUE, RED, action_to_id


def test_initial_board_is_legal():
    # 固定开局中双方应各有 6 枚棋子，且不能出现位置重复。
    env = EinsteinChessEnv(seed=1)
    env.reset(init_mode="fixed", first_player=RED)
    assert len(env.alive_pieces(RED)) == 6
    assert len(env.alive_pieces(BLUE)) == 6
    assert len({p for p in env.positions[RED].values() if p is not None}) == 6
    assert len({p for p in env.positions[BLUE].values() if p is not None}) == 6


def test_red_move_directions_are_legal():
    # 红方在棋盘内部时应有向下、向右、右下三个方向。
    env = EinsteinChessEnv(seed=1)
    env.reset(init_mode="fixed", first_player=RED)
    env.positions[RED] = {i: None for i in range(1, 7)}
    env.positions[BLUE] = {i: None for i in range(1, 7)}
    env.positions[RED][1] = (1, 1)
    env.positions[BLUE][1] = (4, 4)
    env.dice = 1
    assert set(env.legal_actions()) == {action_to_id(1, 0), action_to_id(1, 1), action_to_id(1, 2)}


def test_blue_move_directions_are_legal():
    # 蓝方使用镜像后的向上、向左、左上三个方向，动作编号仍统一为 0、1、2。
    env = EinsteinChessEnv(seed=1)
    env.reset(init_mode="fixed", first_player=BLUE)
    env.positions[RED] = {i: None for i in range(1, 7)}
    env.positions[BLUE] = {i: None for i in range(1, 7)}
    env.positions[BLUE][1] = (3, 3)
    env.positions[RED][1] = (0, 0)
    env.dice = 1
    assert set(env.legal_actions()) == {action_to_id(1, 0), action_to_id(1, 1), action_to_id(1, 2)}


def test_exact_dice_only_selects_that_alive_piece():
    # 骰子对应编号仍存活时，只允许该编号棋子行动。
    env = EinsteinChessEnv(seed=1)
    env.reset(init_mode="fixed", first_player=RED)
    env.dice = 3
    assert env.get_candidate_pieces() == [3]
    assert all(action // 3 + 1 == 3 for action in env.legal_actions())


def test_missing_dice_uses_nearest_lower_and_higher_independently():
    # 骰子对应编号被吃掉时，应分别选择最近的较小编号和较大编号。
    env = EinsteinChessEnv(seed=1)
    env.reset(init_mode="fixed", first_player=RED)
    env.positions[RED][4] = None
    env.positions[RED][5] = None
    env.dice = 4
    assert env.get_candidate_pieces() == [3, 6]

