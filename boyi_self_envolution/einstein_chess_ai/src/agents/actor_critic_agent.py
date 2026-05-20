"""Actor-Critic model-backed agent."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.env.env import EinsteinChessEnv
from src.models.actor_critic import ActorCriticNet, masked_distribution
from src.utils.device import resolve_device

from .base import Agent


@dataclass
class ActionDiagnostics:
    """Policy diagnostics useful for GUI and game records."""

    action: int
    log_prob: float
    value: float
    probabilities: list[float]


class ActorCriticAgent(Agent):
    """Select actions with a PyTorch Actor-Critic network."""

    name = "actor_critic"

    def __init__(
        self,
        model: ActorCriticNet | None = None,
        checkpoint: str | Path | None = None,
        device: str | torch.device = "auto",
        mode: str = "greedy",
        temperature: float = 1.0,
    ) -> None:
        self.device = resolve_device(device)
        self.model = model or ActorCriticNet()
        self.model.to(self.device)
        self.mode = mode
        self.temperature = max(1.0e-6, float(temperature))
        self.last_diagnostics: ActionDiagnostics | None = None
        if checkpoint is not None:
            self.load(checkpoint)

    def load(self, checkpoint: str | Path) -> None:
        """Load model weights from a checkpoint."""
        data = torch.load(checkpoint, map_location=self.device)
        state = data.get("model_state_dict", data)
        self.model.load_state_dict(state)

    @torch.no_grad()
    def evaluate_policy(self, env: EinsteinChessEnv) -> ActionDiagnostics:
        """Return selected action plus policy/value diagnostics."""
        self.model.eval()
        state = torch.from_numpy(env.get_encoded_state()).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(env.legal_action_mask()).unsqueeze(0).to(self.device)
        logits, value = self.model(state)
        logits = logits / self.temperature
        dist = masked_distribution(logits, mask)
        if self.mode == "sample":
            action_tensor = dist.sample()
        else:
            action_tensor = torch.argmax(dist.probs, dim=-1)
        log_prob = dist.log_prob(action_tensor).item()
        probs = dist.probs.squeeze(0).detach().cpu().numpy().astype(float)
        diagnostics = ActionDiagnostics(
            action=int(action_tensor.item()),
            log_prob=float(log_prob),
            value=float(value.item()),
            probabilities=np.asarray(probs, dtype=float).tolist(),
        )
        self.last_diagnostics = diagnostics
        return diagnostics

    def select_action(self, env: EinsteinChessEnv) -> int:
        return self.evaluate_policy(env).action
