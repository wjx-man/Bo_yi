from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from ..engine import GameSnapshot, Move
from ..players import PlayerAgent
from ..training.action_codec import ACTION_SIZE, action_id_to_move, legal_action_mask
from ..training.model import PolicyValueNet
from ..training.state_encoder import encode_state


class PolicyValueAgent(PlayerAgent):
    mode = "policy_value"

    def __init__(
        self,
        checkpoint_path: str | Path,
        name: str = "PolicyValue",
        device: str = "cpu",
    ) -> None:
        super().__init__(name=name)
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)
        self.model = self._load_model(self.checkpoint_path)
        self.model.to(self.device)
        self.model.eval()
        self.last_value: float = 0.0
        self.last_policy: np.ndarray = np.zeros(ACTION_SIZE, dtype=np.float32)

    def choose_move(
        self, snapshot: GameSnapshot, legal_moves: Sequence[Move]
    ) -> Move:
        if not legal_moves:
            raise ValueError("No legal moves are available.")

        state = torch.from_numpy(encode_state(snapshot)).unsqueeze(0).to(self.device)
        mask = legal_action_mask(snapshot)
        with torch.no_grad():
            policy_logits, value = self.model(state)
            logits = policy_logits.squeeze(0).detach().cpu().numpy()

        masked_logits = np.full(ACTION_SIZE, -np.inf, dtype=np.float32)
        masked_logits[mask > 0] = logits[mask > 0]
        best_action_id = int(np.argmax(masked_logits))
        self.last_policy = _softmax_masked(logits, mask)
        self.last_value = float(value.squeeze(0).detach().cpu())
        return action_id_to_move(best_action_id, snapshot)

    def _load_model(self, checkpoint_path: Path) -> PolicyValueNet:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        config = checkpoint.get("config", {})
        hidden_channels = int(config.get("hidden_channels", 64))
        residual_blocks = int(config.get("residual_blocks", 0))
        model = PolicyValueNet(
            hidden_channels=hidden_channels,
            residual_blocks=residual_blocks,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        return model


def _softmax_masked(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    valid_logits = np.full(ACTION_SIZE, -np.inf, dtype=np.float32)
    valid_logits[mask > 0] = logits[mask > 0]
    max_logit = float(np.max(valid_logits[mask > 0]))
    exp_logits = np.zeros(ACTION_SIZE, dtype=np.float32)
    exp_logits[mask > 0] = np.exp(valid_logits[mask > 0] - max_logit)
    total = float(exp_logits.sum())
    if total > 0:
        exp_logits /= total
    return exp_logits
