"""验证 Actor-Critic 网络前向传播的输出形状。"""

import pytest

torch = pytest.importorskip("torch")

from src.models.actor_critic import ActorCriticNet


def test_actor_critic_forward_shapes():
    # batch=2，每个样本是 18 个 5x5 状态平面。
    model = ActorCriticNet()
    logits, value = model(torch.zeros(2, 18, 5, 5))
    # Actor 为每个样本输出 18 个动作分数，Critic 输出 1 个局面价值。
    assert logits.shape == (2, 18)
    assert value.shape == (2, 1)
