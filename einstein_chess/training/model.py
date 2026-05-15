from __future__ import annotations

import torch
from torch import nn

from .action_codec import ACTION_SIZE
from .state_encoder import STATE_CHANNELS


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.ReLU()

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.activation(states + self.layers(states))


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        in_channels: int = STATE_CHANNELS,
        hidden_channels: int = 64,
        action_size: int = ACTION_SIZE,
        residual_blocks: int = 0,
    ) -> None:
        super().__init__()
        if residual_blocks < 0:
            raise ValueError("residual_blocks must be non-negative.")
        self.residual_blocks = residual_blocks
        if residual_blocks == 0:
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
            )
        else:
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
                *[ResidualBlock(hidden_channels) for _ in range(residual_blocks)],
            )
        flattened_size = hidden_channels * 5 * 5
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(states)
        policy_logits = self.policy_head(features)
        values = self.value_head(features).squeeze(-1)
        return policy_logits, values


def policy_value_loss(
    policy_logits: torch.Tensor,
    predicted_values: torch.Tensor,
    target_policies: torch.Tensor,
    target_values: torch.Tensor,
    value_loss_weight: float = 1.0,
    l2_weight: float = 0.0,
    model: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
    value_loss = nn.functional.mse_loss(predicted_values, target_values)
    total_loss = policy_loss + value_loss_weight * value_loss

    if l2_weight and model is not None:
        l2_loss = torch.zeros((), device=policy_logits.device)
        for parameter in model.parameters():
            l2_loss = l2_loss + parameter.pow(2).sum()
        total_loss = total_loss + l2_weight * l2_loss

    return total_loss, policy_loss, value_loss
