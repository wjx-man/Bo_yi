"""强化学习经验回放池。"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np


class TransitionReplayBuffer:
    """固定容量的经验回放池。

    回放池保存自我博弈产生的历史 transition。训练时随机采样，能够打乱
    相邻棋步之间的强相关性；容量满后，deque 会自动丢弃最旧的数据。
    """

    def __init__(self, capacity: int = 100_000, seed: int | None = None) -> None:
        self.capacity = int(capacity)
        self.buffer: deque[dict[str, Any]] = deque(maxlen=self.capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: dict[str, Any]) -> None:
        """加入一条 transition。"""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """随机采样一个 batch，并将相同字段堆叠为 NumPy 数组。"""
        if not self.buffer:
            raise ValueError("Cannot sample from an empty replay buffer")
        batch = self.rng.sample(list(self.buffer), min(int(batch_size), len(self.buffer)))
        # 模型一次处理整个 batch，因此 state 等字段需要从多条记录堆叠起来。
        return {
            "state": np.stack([item["state"] for item in batch]).astype(np.float32),
            "legal_mask": np.stack([item["legal_mask"] for item in batch]).astype(np.float32),
            "action": np.asarray([item["action"] for item in batch], dtype=np.int64),
            "reward": np.asarray([item["reward"] for item in batch], dtype=np.float32),
            "next_state": np.stack([item["next_state"] for item in batch]).astype(np.float32),
            "next_legal_mask": np.stack([item["next_legal_mask"] for item in batch]).astype(np.float32),
            "done": np.asarray([item["done"] for item in batch], dtype=np.float32),
            "old_log_prob": np.asarray([item["old_log_prob"] for item in batch], dtype=np.float32),
            "old_value": np.asarray([item.get("old_value", 0.0) for item in batch], dtype=np.float32),
        }

    def state_dict(self) -> dict[str, Any]:
        """Return serializable buffer state."""
        return {"capacity": self.capacity, "buffer": list(self.buffer)}

    @classmethod
    def from_state_dict(cls, data: dict[str, Any]) -> "TransitionReplayBuffer":
        """Restore a buffer from state."""
        obj = cls(data["capacity"])
        obj.buffer.extend(data["buffer"])
        return obj

