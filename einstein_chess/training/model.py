"""策略价值神经网络。

输入：一个形状为 (15, 5, 5) 的局面状态。
输出：
1. policy_logits：18 个动作分数，回答“当前应该怎么走”。
2. value：一个 [-1, 1] 范围内的局面价值，回答“当前行动方是否有利”。
"""

from __future__ import annotations

import torch
from torch import nn

from .action_codec import ACTION_SIZE
from .state_encoder import STATE_CHANNELS


class ResidualBlock(nn.Module):
    """ResNet V4 试验使用的残差块；最终 V4 Full 没有采用该结构。"""

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
        """将输入与卷积结果相加，形成残差连接。"""
        return self.activation(states + self.layers(states))


class PolicyValueNet(nn.Module):
    """共享棋盘特征提取主干，并分出策略头和价值头。"""

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
            # 最终 V4 Full 使用三层普通 CNN 主干。
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
            # 该分支用于曾经尝试过、但实战效果较差的 ResNet V4。
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.ReLU(),
                *[ResidualBlock(hidden_channels) for _ in range(residual_blocks)],
            )
        # 64 张 5 × 5 特征图会被展平为 1600 个数字。
        flattened_size = hidden_channels * 5 * 5
        # 策略头输出 18 个动作 logits；它们还不是概率。
        self.policy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
        # 价值头输出当前行动方的局面价值，Tanh 将结果限制在 [-1, 1]。
        self.value_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """对一批局面同时输出动作分数和局面价值。"""
        # 策略头和价值头共享同一份棋盘特征，分别学习不同任务。
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
    """计算策略误差、价值误差和总误差。"""
    # 策略头学习搜索器给出的访问次数分布，而不是只学习最终执行动作。
    log_probs = torch.log_softmax(policy_logits, dim=1)
    policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
    # 价值头学习整局自我对弈的最终胜负标签：+1、-1 或 0。
    value_loss = nn.functional.mse_loss(predicted_values, target_values)
    total_loss = policy_loss + value_loss_weight * value_loss

    # 可选的 L2 正则用于限制参数过度增大。
    if l2_weight and model is not None:
        l2_loss = torch.zeros((), device=policy_logits.device)
        for parameter in model.parameters():
            l2_loss = l2_loss + parameter.pow(2).sum()
        total_loss = total_loss + l2_weight * l2_loss

    return total_loss, policy_loss, value_loss


# 下一步阅读：einstein_chess/agents/neural_mcts.py
