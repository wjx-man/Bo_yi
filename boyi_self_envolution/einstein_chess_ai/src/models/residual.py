"""用于 5x5 棋盘网络的小型残差块。"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """两层卷积组成的残差块，用于稳定较深网络的训练。"""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 跳跃连接保留原始特征：输出不是单纯的 F(x)，而是 x + F(x)。
        return self.relu(x + self.net(x))

