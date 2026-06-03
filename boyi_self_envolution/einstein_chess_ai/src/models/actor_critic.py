"""爱恩斯坦棋的 PyTorch Actor-Critic 网络。"""

from __future__ import annotations

import torch
from torch import nn

from .residual import ResidualBlock


class ActorCriticNet(nn.Module):
    """用于 5x5 棋盘的轻量级 Actor-Critic 网络。

    输入状态形状为 [batch, 18, 5, 5]。共享主干先提取棋盘特征，
    Actor 输出 18 个动作的分数，Critic 输出当前玩家视角的局面价值。
    """

    def __init__(self, in_channels: int = 18, num_res_blocks: int = 2) -> None:
        super().__init__()
        # 共享主干：卷积层保留棋盘的空间关系，残差块增强特征提取能力。
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
        # 18 = 6 个棋子 * 每个棋子的 3 个移动方向。
        # Actor 输出的是 logits（未归一化分数），后续再转换为动作概率。
        self.actor = nn.Linear(256, 18)
        # Critic 输出一个 [-1, 1] 范围内的标量，表示当前局面对当前玩家的价值。
        self.critic = nn.Sequential(nn.Linear(256, 64), nn.ReLU(inplace=True), nn.Linear(64, 1), nn.Tanh())

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """执行前向传播，返回 (动作分数, 局面价值)。"""
        features = self.trunk(state.float())
        return self.actor(features), self.critic(features)


def masked_logits(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """将非法动作分数设为极小值，使其 softmax 概率接近 0。"""
    mask = legal_mask.bool()
    return logits.masked_fill(~mask, -1.0e9)


def masked_distribution(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.distributions.Categorical:
    """只在合法动作上创建离散概率分布。"""
    return torch.distributions.Categorical(logits=masked_logits(logits, legal_mask))

