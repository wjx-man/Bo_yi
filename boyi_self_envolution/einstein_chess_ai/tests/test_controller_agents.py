import pytest

pytest.importorskip("torch")

from src.agents.actor_critic_agent import ActorCriticAgent
from src.agents.base import HumanAgent
from src.agents.random_agent import RandomAgent
from src.agents.rule_based import RuleBasedAgent
from src.agents.search import AlphaBetaAgent, MCTSAgent, MinimaxAgent
from src.gui.controller import make_agent


def test_make_agent_supports_tk_agent_choices():
    params = {
        "device": "cpu",
        "actor_mode": "greedy",
        "temperature": 1.0,
        "minimax_depth": 1,
        "alphabeta_depth": 1,
        "mcts_simulations": 2,
        "mcts_rollout_steps": 2,
        "mcts_exploration": 1.0,
        "seed": 1,
    }
    expected = {
        "Human": HumanAgent,
        "Random": RandomAgent,
        "RuleBased": RuleBasedAgent,
        "ActorCritic": ActorCriticAgent,
        "Minimax": MinimaxAgent,
        "AlphaBeta": AlphaBetaAgent,
        "MCTS": MCTSAgent,
    }
    for kind, agent_type in expected.items():
        assert isinstance(make_agent(kind, params=params), agent_type)
