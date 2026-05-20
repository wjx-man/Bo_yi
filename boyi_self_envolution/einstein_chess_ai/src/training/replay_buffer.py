"""Experience replay buffer."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np


class TransitionReplayBuffer:
    """Fixed-size replay buffer for transition dictionaries."""

    def __init__(self, capacity: int = 100_000, seed: int | None = None) -> None:
        self.capacity = int(capacity)
        self.buffer: deque[dict[str, Any]] = deque(maxlen=self.capacity)
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: dict[str, Any]) -> None:
        """Add one transition."""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> dict[str, Any]:
        """Sample a batch and stack array fields."""
        if not self.buffer:
            raise ValueError("Cannot sample from an empty replay buffer")
        batch = self.rng.sample(list(self.buffer), min(int(batch_size), len(self.buffer)))
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

