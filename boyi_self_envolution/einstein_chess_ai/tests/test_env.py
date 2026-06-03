"""验证强化学习环境执行动作后的吃子、奖励和终局状态。"""

from src.env.env import EinsteinChessEnv
from src.env.rules import BLUE, RED, action_to_id


def clear(env):
    """清空默认棋盘，便于构造只包含少量棋子的精确测试局面。"""
    env.positions = {
        RED: {i: None for i in range(1, 7)},
        BLUE: {i: None for i in range(1, 7)},
    }
    env.winner = None
    env.win_reason = None
    env.step_count = 0


def test_capture_opponent_piece():
    # 红方 1 号棋向右移动后应覆盖并移除蓝方 1 号棋。
    env = EinsteinChessEnv(seed=1)
    env.reset(first_player=RED)
    clear(env)
    env.current_player = RED
    env.dice = 1
    env.positions[RED][1] = (0, 0)
    env.positions[BLUE][1] = (0, 1)
    env.step(action_to_id(1, 0))
    assert env.positions[BLUE][1] is None
    assert env.positions[RED][1] == (0, 1)


def test_capture_own_piece():
    # 爱恩斯坦棋允许进入己方棋子所在格，因此己方棋子也会被移除。
    env = EinsteinChessEnv(seed=1)
    env.reset(first_player=RED)
    clear(env)
    env.current_player = RED
    env.dice = 1
    env.positions[RED][1] = (0, 0)
    env.positions[RED][2] = (0, 1)
    env.positions[BLUE][1] = (4, 4)
    env.step(action_to_id(1, 0))
    assert env.positions[RED][2] is None
    assert env.positions[RED][1] == (0, 1)


def test_red_reaches_goal_and_wins():
    # 红方到达右下目标角后，环境应返回正奖励并结束棋局。
    env = EinsteinChessEnv(seed=1)
    env.reset(first_player=RED)
    clear(env)
    env.current_player = RED
    env.dice = 6
    env.positions[RED][6] = (3, 3)
    env.positions[BLUE][1] = (0, 0)
    _, reward, done, _ = env.step(action_to_id(6, 2))
    assert done
    assert reward == 1.0
    assert env.winner == RED
    assert env.win_reason == "reach_corner"


def test_blue_reaches_goal_and_wins():
    # 蓝方到达左上目标角时也应使用相同的奖励与终局逻辑。
    env = EinsteinChessEnv(seed=1)
    env.reset(first_player=BLUE)
    clear(env)
    env.current_player = BLUE
    env.dice = 6
    env.positions[BLUE][6] = (1, 1)
    env.positions[RED][1] = (4, 4)
    _, reward, done, _ = env.step(action_to_id(6, 2))
    assert done
    assert reward == 1.0
    assert env.winner == BLUE
    assert env.win_reason == "reach_corner"


def test_capture_all_wins():
    # 吃掉对方最后一枚棋子是另一种获胜条件。
    env = EinsteinChessEnv(seed=1)
    env.reset(first_player=RED)
    clear(env)
    env.current_player = RED
    env.dice = 1
    env.positions[RED][1] = (0, 0)
    env.positions[BLUE][1] = (0, 1)
    env.step(action_to_id(1, 0))
    assert env.winner == RED
    assert env.win_reason == "capture_all"

