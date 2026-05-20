"""GUI controller that keeps widgets thin."""

from __future__ import annotations

from src.agents.actor_critic_agent import ActorCriticAgent
from src.agents.base import Agent, HumanAgent
from src.agents.random_agent import RandomAgent
from src.agents.rule_based import RuleBasedAgent
from src.agents.search import AlphaBetaAgent, MCTSAgent, MinimaxAgent
from src.env.env import EinsteinChessEnv
from src.env.rules import DELTAS, BLUE, RED, id_to_action
from src.utils.device import resolve_device


class GameController:
    """Coordinate environment and agents for the GUI."""

    def __init__(self) -> None:
        self.env = EinsteinChessEnv()
        self.agents: dict[str, Agent] = {RED: HumanAgent(), BLUE: RuleBasedAgent()}
        self.selected_source: tuple[int, int] | None = None

    def new_game(self, red_agent: Agent, blue_agent: Agent) -> None:
        """Start a new game with selected agents."""
        self.env.reset()
        self.agents = {RED: red_agent, BLUE: blue_agent}
        self.selected_source = None

    def legal_targets(self) -> set[tuple[int, int]]:
        """Return target squares for current legal actions."""
        targets = set()
        player = self.env.current_player
        for action in self.env.legal_actions():
            piece_id, direction = id_to_action(action)
            pos = self.env.positions[player][piece_id]
            if pos is None:
                continue
            dr, dc = DELTAS[player][direction]
            targets.add((pos[0] + dr, pos[1] + dc))
        return targets

    def handle_cell_click(self, row: int, col: int) -> bool:
        """Process a human click. Returns True if a move was made."""
        if self.env.is_terminal() or not isinstance(self.agents[self.env.current_player], HumanAgent):
            return False
        player = self.env.current_player
        clicked = (row, col)
        if self.selected_source is None:
            if clicked in self.env.positions[player].values():
                self.selected_source = clicked
            return False
        for action in self.env.legal_actions():
            piece_id, direction = id_to_action(action)
            pos = self.env.positions[player][piece_id]
            if pos != self.selected_source:
                continue
            dr, dc = DELTAS[player][direction]
            if (pos[0] + dr, pos[1] + dc) == clicked:
                self.env.step(action)
                self.selected_source = None
                return True
        self.selected_source = None
        return False

    def step_ai_if_needed(self) -> bool:
        """Run one AI move when current player is controlled by AI."""
        if self.env.is_terminal():
            return False
        agent = self.agents[self.env.current_player]
        if isinstance(agent, HumanAgent):
            return False
        self.env.step(agent.select_action(self.env))
        return True


def make_agent(kind: str, checkpoint: str | None = None, params: dict | None = None) -> Agent:
    """Create an agent from a GUI selection."""
    params = params or {}
    if kind == "Human":
        return HumanAgent()
    if kind == "Random":
        return RandomAgent(seed=params.get("seed"))
    if kind == "RuleBased":
        return RuleBasedAgent(seed=params.get("seed"))
    if kind == "ActorCritic":
        device = resolve_device(params.get("device", "auto"))
        mode = str(params.get("actor_mode", "greedy"))
        temperature = float(params.get("temperature", 1.0))
        return (
            ActorCriticAgent(checkpoint=checkpoint, device=device, mode=mode, temperature=temperature)
            if checkpoint
            else ActorCriticAgent(device=device, mode=mode, temperature=temperature)
        )
    if kind == "Minimax":
        return MinimaxAgent(depth=int(params.get("minimax_depth", 2)), seed=params.get("seed"))
    if kind == "AlphaBeta":
        return AlphaBetaAgent(depth=int(params.get("alphabeta_depth", 3)), seed=params.get("seed"))
    if kind == "MCTS":
        return MCTSAgent(
            simulations=int(params.get("mcts_simulations", 128)),
            rollout_steps=int(params.get("mcts_rollout_steps", 80)),
            exploration=float(params.get("mcts_exploration", 1.4)),
            seed=params.get("seed"),
        )
    raise ValueError(f"Unknown agent kind: {kind}")
