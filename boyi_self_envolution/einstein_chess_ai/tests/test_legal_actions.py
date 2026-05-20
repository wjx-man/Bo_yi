from src.env.env import EinsteinChessEnv


def test_legal_action_mask_shape_is_18():
    env = EinsteinChessEnv(seed=1)
    env.reset(first_player="red")
    mask = env.legal_action_mask()
    assert mask.shape == (18,)
    assert mask.sum() >= 1

