"""验证传统搜索智能体始终返回环境认可的合法动作。"""

from src.agents.search import AlphaBetaAgent, MCTSAgent, MinimaxAgent
from src.env.env import EinsteinChessEnv


def test_search_agents_return_legal_actions():
    # 测试使用浅搜索和少量模拟，只检查接口正确性而不比较棋力。
    env = EinsteinChessEnv(seed=7, max_steps=60)
    env.reset(seed=7)
    for agent in (
        MinimaxAgent(depth=1, seed=1),
        AlphaBetaAgent(depth=1, seed=2),
        MCTSAgent(simulations=4, rollout_steps=6, seed=3),
    ):
        action = agent.select_action(env)
        assert action in env.legal_actions()
