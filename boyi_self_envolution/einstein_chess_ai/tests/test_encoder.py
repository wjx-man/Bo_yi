"""验证棋盘状态编码的形状，以及蓝方视角镜像是否正确。"""

from src.env.env import EinsteinChessEnv
from src.env.rules import BLUE, RED


def test_encoder_shape_and_blue_mirroring():
    env = EinsteinChessEnv(seed=1)
    env.reset(first_player=BLUE, init_mode="fixed")
    env.dice = 1
    state = env.get_encoded_state()
    # 模型固定接收 18 个 5x5 特征平面。
    assert state.shape == (18, 5, 5)
    # 蓝方 1 号棋真实位置是 (4,4)，镜像到当前玩家平面的 (0,0)。
    assert state[0, 0, 0] == 1.0
    # 红方 1 号棋是真实位置 (0,0)，镜像到对手平面的 (4,4)。
    assert state[6, 4, 4] == 1.0
    # 骰子为 1 时，对应骰子平面应在 25 个格子中全部置 1。
    assert state[12].sum() == 25.0

