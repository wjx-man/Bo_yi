import numpy as np

from src.training.replay_buffer import TransitionReplayBuffer


def test_replay_buffer_add_and_sample():
    buffer = TransitionReplayBuffer(capacity=4, seed=1)
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
    assert len(buffer) == 1
    assert batch["state"].shape == (1, 18, 5, 5)
    assert batch["action"].shape == (1,)

