"""验证固定动作空间对应的合法动作掩码。"""

from src.env.env import EinsteinChessEnv


def test_legal_action_mask_shape_is_18():
    env = EinsteinChessEnv(seed=1)
    env.reset(first_player="red")
    mask = env.legal_action_mask()
    # 18 = 6 个棋子 * 3 个移动方向；初始局面至少存在一个合法动作。
    assert mask.shape == (18,)
    assert mask.sum() >= 1

