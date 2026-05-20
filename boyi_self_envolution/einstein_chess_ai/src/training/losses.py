"""Actor-Critic/PPO-style losses."""

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
    """Compute PPO-style replay loss with current-player value perspective."""
    device = torch.device(device)
    states = torch.as_tensor(batch["state"], dtype=torch.float32, device=device)
    masks = torch.as_tensor(batch["legal_mask"], dtype=torch.float32, device=device)
    actions = torch.as_tensor(batch["action"], dtype=torch.int64, device=device)
    rewards = torch.as_tensor(batch["reward"], dtype=torch.float32, device=device)
    next_states = torch.as_tensor(batch["next_state"], dtype=torch.float32, device=device)
    next_masks = torch.as_tensor(batch["next_legal_mask"], dtype=torch.float32, device=device)
    dones = torch.as_tensor(batch["done"], dtype=torch.float32, device=device)
    old_log_probs = torch.as_tensor(batch["old_log_prob"], dtype=torch.float32, device=device)

    logits, values = model(states)
    values = values.squeeze(-1)
    dist = masked_distribution(logits, masks)
    new_log_probs = dist.log_prob(actions)
    entropy = dist.entropy().mean()

    with torch.no_grad():
        _, next_values = model(next_states)
        next_values = next_values.squeeze(-1)
        # Next state is encoded from the opponent's perspective, hence the sign flip.
        targets = rewards - gamma * next_values * (1.0 - dones)
        advantages = targets - values
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1.0e-8)

    ratio = torch.exp(new_log_probs - old_log_probs)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - ppo_clip_eps, 1.0 + ppo_clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()
    value_loss = F.mse_loss(values, targets)
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    metrics = {
        "loss": float(loss.detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "value_mean": float(values.mean().detach().cpu()),
    }
    return loss, metrics

