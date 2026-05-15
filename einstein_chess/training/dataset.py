from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .action_codec import ACTION_SIZE
from .state_encoder import STATE_CHANNELS


class NPZSelfPlayDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        with np.load(self.path) as data:
            self.states = data["states"].astype(np.float32)
            self.policies = data["policies"].astype(np.float32)
            self.values = data["values"].astype(np.float32)

        self._validate()

    def __len__(self) -> int:
        return int(self.states.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = torch.from_numpy(self.states[index])
        policy = torch.from_numpy(self.policies[index])
        value = torch.tensor(self.values[index], dtype=torch.float32)
        return state, policy, value

    def _validate(self) -> None:
        sample_count = self.states.shape[0]
        if self.states.shape != (sample_count, STATE_CHANNELS, 5, 5):
            raise ValueError(
                f"states must have shape (N, {STATE_CHANNELS}, 5, 5), got {self.states.shape}."
            )
        if self.policies.shape != (sample_count, ACTION_SIZE):
            raise ValueError(
                f"policies must have shape (N, {ACTION_SIZE}), got {self.policies.shape}."
            )
        if self.values.shape != (sample_count,):
            raise ValueError(f"values must have shape (N,), got {self.values.shape}.")
        if sample_count == 0:
            raise ValueError("Dataset is empty.")
