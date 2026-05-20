from src.env.env import EinsteinChessEnv
from src.env.rules import BLUE, RED


def test_encoder_shape_and_blue_mirroring():
    env = EinsteinChessEnv(seed=1)
    env.reset(first_player=BLUE, init_mode="fixed")
    env.dice = 1
    state = env.get_encoded_state()
    assert state.shape == (18, 5, 5)
    # Blue piece 1 starts at real (4,4), mirrored to current-player plane (0,0).
    assert state[0, 0, 0] == 1.0
    # Red opponent piece 1 at real (0,0), mirrored to opponent plane (4,4).
    assert state[6, 4, 4] == 1.0
    assert state[12].sum() == 25.0

