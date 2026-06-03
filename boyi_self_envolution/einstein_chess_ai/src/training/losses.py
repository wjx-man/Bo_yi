"""Actor-Critic 与 PPO 风格损失函数。"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.actor_critic import masked_distribution


def compute_actor_critic_loss(
    model,
    batch: dict,
    gamma: float = 0.95,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    ppo_clip_eps: float = 0.2,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, dict[str, float]]:
    """计算 PPO 风格的 Actor-Critic 损失。

    Actor 学习提高好动作的概率，Critic 学习预测局面价值，熵奖励用于保留探索。
    由于数据来自经验回放池，这里更准确地说是 PPO 风格实现，而非严格 on-policy PPO。
    """
    device = torch.device(device)
    # 将回放池中的 NumPy 数据转换为 PyTorch Tensor，并移动到训练设备。
    states = torch.as_tensor(batch["state"], dtype=torch.float32, device=device)
    masks = torch.as_tensor(batch["legal_mask"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["action"], dtype=torch.int64, device=device)
    rewards = torch.as_tensor(batch["reward"], dtype=torch.float32, device=device)
    next_states = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=device)
    next_masks = torch.as_tensor(batch["next_legal_mask"], dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch["done"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(batch["old_log_prob"], dtype=torch.float32, device=device)

    # 当前模型的前向传播：同时得到动作分数和当前状态价值。
    logits, values = model(states)
    values = values.squeeze(-1)
    dist = masked_distribution(logits, masks)
    # 只取 batch 中实际执行过的动作概率，用于比较新旧策略。
    new_log_probs = dist.log_prob(actions)
    # 熵越大表示策略越分散，加入熵奖励可以避免过早停止探索。
    entropy = dist.entropy().mean()

    with torch.no_grad():
        # TD 目标只作为监督信号，不需要对 next_state 分支反向传播。
        _, next_values = model(next_states)
        next_values = next_values.squeeze(-1)
        # 下一状态是从对手视角编码的：对手价值越高，当前玩家价值越低，因此使用减号。
        # 终局时 dones=1，未来价值项会被清零，目标只剩本步奖励。
        targets = rewards - gamma * next_values * (1.0 - dones)
        # 优势函数衡量“实际目标价值”比“模型原先预期价值”好多少。
        advantages = targets - values
        if advantages.numel() > 1:
            # 标准化优势可以减小 batch 尺度差异，让策略更新更稳定。
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)

    # ratio = 新策略执行该动作的概率 / 旧策略执行该动作的概率。
    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    # PPO 裁剪限制单次策略更新幅度，避免动作概率突然变化过大。
    clipped = torch.clamp(ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    # Critic 使用均方误差，让预测价值逐渐接近 TD 目标。
    value_loss = F.mse_loss(values, targets)
    # 减去 entropy 是因为优化器最小化 loss，而我们希望最大化策略熵。
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    metrics = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "value_mean": float(values.mean().detach().cpu()),
    }
    return loss, metrics

