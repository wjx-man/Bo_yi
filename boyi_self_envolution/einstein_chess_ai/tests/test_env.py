from src.env.env import EinsteinChessEnv
from src.env.rules import BLUE, RED, action_to_id


def clear(env):
    env.positions = {
        RED: {i: None for i in range(1, 7)},
        BLUE: {i: None for i in range(1, 7)},
    }
    env.winner = None
    env.win_reason = None
    env.step_count = 0


def test_capture_opponent_piece():
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

