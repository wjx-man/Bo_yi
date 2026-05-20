"""PyTorch Actor-Critic network for Einstein chess."""

from __future__ import annotations

import torch
from torch import nn

from .residual import ResidualBlock


class ActorCriticNet(nn.Module):
    """Lightweight Actor-Critic model for 5x5 board states."""

    def __init__(self, in_channels: int = 18, num_res_blocks: int = 2) -> None:
        super().__init__()
        trunk_layers: list[nn.Module] = [
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]
        trunk_layers.extend(ResidualBlock(64) for _ in range(num_res_blocks))
        trunk_layers.extend(
            [
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Flatten(),
                nn.Linear(128 * 5 * 5, 256),
                nn.ReLU(inplace=True),
            ]
        )
        self.trunk = nn.Sequential(*trunk_layers)
        self.actor = nn.Linear(256, 18)
        self.critic = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Tanh())

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (actor_logits, value)."""
        features = self.trunk(state.float())
        return self.actor(features), self.critic(features)


def masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Set illegal action logits to a large negative value."""
    mask = legal_mask.bool()
    return logits.masked_fill(~mask, -1.0e9)


def masked_distribution(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.distributions.Categorical:
    """Create a categorical distribution over legal actions only."""
    return torch.distributions.Categorical(logits=masked_logits(logits, legal_mask))

