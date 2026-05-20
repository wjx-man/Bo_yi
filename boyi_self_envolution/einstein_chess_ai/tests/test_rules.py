from src.env.env import EinsteinChessEnv
from src.env.rules import BLUE, RED, action_to_id


def test_initial_board_is_legal():
    env = EinsteinChessEnv(seed=1)
    env.reset(init_mode="fixed", first_player=RED)
    assert len(env.alive_pieces(RED)) == 6
    assert len(env.alive_pieces(BLUE)) == 6
    assert len({p for p in env.positions[RED].values() if p is not None}) == 6
    assert len({p for p in env.positions[BLUE].values() if p is not None}) == 6


def test_red_move_directions_are_legal():
    env = EinsteinChessEnv(seed=1)
    env.reset(init_mode="fixed", first_player=RED)
    env.positions[RED] = {i: None for i in range(1, 7)}
    env.positions[BLUE] = {i: None for i in range(1, 7)}
    env.positions[RED][1] = (1, 1)
    env.positions[BLUE][1] = (4, 4)
    env.dice = 1
    assert set(env.legal_actions()) == {action_to_id(1, 0), action_to_id(1, 1), action_to_id(1, 2)}


def test_blue_move_directions_are_legal():
    env = EinsteinChessEnv(seed=1)
    env.reset(init_mode="fixed", first_player=BLUE)
    env.positions[RED] = {i: None for i in range(1, 7)}
    env.positions[BLUE] = {i: None for i in range(1, 7)}
    env.positions[BLUE][1] = (3, 3)
    env.positions[RED][1] = (0, 0)
    env.dice = 1
    assert set(env.legal_actions()) == {action_to_id(1, 0), action_to_id(1, 1), action_to_id(1, 2)}


def test_exact_dice_only_selects_that_alive_piece():
    env = EinsteinChessEnv(seed=1)
    env.reset(init_mode="fixed", first_player=RED)
    env.dice = 3
    assert env.get_candidate_pieces() == [3]
    assert all(action // 3 + 1 == 3 for action in env.legal_actions())


def test_missing_dice_uses_nearest_lower_and_higher_independently():
    env = EinsteinChessEnv(seed=1)
    env.reset(init_mode="fixed", first_player=RED)
    env.positions[RED][4] = None
    env.positions[RED][5] = None
    env.dice = 4
    assert env.get_candidate_pieces() == [3, 6]

