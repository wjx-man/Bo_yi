"""验证经验回放池能够保存 transition 并按 batch 形状采样。"""

import numpy as np

from src.training.replay_buffer import TransitionReplayBuffer


def test_replay_buffer_add_and_sample():
    buffer = TransitionReplayBuffer(capacity=4, seed=1)
    # 构造一条与自我博弈输出字段一致的最小 transition。
    transition = {
        "state": np.zeros((18, 5, 5), dtype=np.float32),
        "legal_mask": np.ones(18, dtype=np.float32),
        "action": 0,
        "reward": 0.0,
        "next_state": np.zeros((18, 5, 5), dtype=np.float32),
        "next_legal_mask": np.ones(18, dtype=np.float32),
        "done": False,
        "old_log_prob": -1.0,
        "old_value": 0.0,
    }
    buffer.add(transition)
    batch = buffer.sample(1)
    # 采样后 state 增加 batch 维度，标量 action 也堆叠为一维数组。
    assert len(buffer) == 1
    assert batch["state"].shape == (1, 18, 5, 5)
    assert batch["action"].shape == (1,)

