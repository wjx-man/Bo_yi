"""由 Actor-Critic 神经网络驱动的智能体。"""

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
    """一次决策的附加信息，训练、GUI 和棋局记录都会使用。"""

    action: int
    # 选择该动作时的对数概率，PPO 风格损失需要用它与新策略比较。
    log_prob: float
    # Critic 对当前局面的价值预测。
    value: float
    # 18 个动作的完整概率，便于界面展示和调试。
    probabilities: list[float]


class ActorCriticAgent(Agent):
    """使用 Actor-Critic 网络选择动作的智能体。"""

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
        """从检查点加载模型参数。"""
        data = torch.load(checkpoint, map_location=self.device)
        state = data.get("model_state_dict", data)
        self.model.load_state_dict(state)

    @torch.no_grad()
    def evaluate_policy(self, env: EinsteinChessEnv) -> ActionDiagnostics:
        """执行一次只推理、不计算梯度的前向传播。"""
        self.model.eval()
        # 环境状态原本是 [18, 5, 5]，unsqueeze(0) 增加 batch 维度。
        state = torch.from_numpy(env.get_encoded_state()).unsqueeze(0).to(self.device)
        mask = torch.from_numpy(env.legal_action_mask()).unsqueeze(0).to(self.device)
        logits, value = self.model(state)
        # 温度越高，动作分布越平缓；温度越低，越倾向于最高分动作。
        logits = logits / self.temperature
        dist = masked_distribution(logits, mask)
        if self.mode == "sample":
            # 自我博弈训练时使用采样，保留探索能力。
            action_tensor = dist.sample()
        else:
            # 评估和实际对战时使用贪心选择，保证决策稳定。
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
