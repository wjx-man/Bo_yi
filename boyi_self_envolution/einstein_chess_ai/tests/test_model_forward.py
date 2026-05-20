import pytest

torch = pytest.importorskip("torch")

from src.models.actor_critic import ActorCriticNet


def test_actor_critic_forward_shapes():
    model = ActorCriticNet()
    logits, value = model(torch.zeros(2, 18, 5, 5))
    assert logits.shape == (2, 18)
    assert value.shape == (2, 1)
