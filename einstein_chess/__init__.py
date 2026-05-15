from .engine import EinsteinGame, GameSnapshot, Move, Piece, PlayerColor
from .agents import (
    FullNeuralMCTSAgent,
    FullNeuralMCTSSearchStats,
    MCTSAgent,
    MCTSSearchStats,
    NeuralMCTSAgent,
    NeuralMCTSSearchStats,
    PolicyValueAgent,
)
from .match import MatchResult, MatchRunner, MatchStep
from .players import HumanPlayer, PlayerAgent, RandomAIPlayer
from .training import ACTION_SIZE, STATE_CHANNELS

__all__ = [
    "ACTION_SIZE",
    "EinsteinGame",
    "FullNeuralMCTSAgent",
    "FullNeuralMCTSSearchStats",
    "GameSnapshot",
    "HumanPlayer",
    "MCTSAgent",
    "MCTSSearchStats",
    "MatchResult",
    "MatchRunner",
    "MatchStep",
    "NeuralMCTSAgent",
    "NeuralMCTSSearchStats",
    "Move",
    "Piece",
    "PlayerAgent",
    "PlayerColor",
    "PolicyValueAgent",
    "RandomAIPlayer",
    "STATE_CHANNELS",
]

try:
    from .online_match_client import (
        OnlineMatchClient,
        OnlineMatchProtocolError,
        OnlineMatchStateView,
        layout_order_from_mapping,
        move_to_wire_dict,
        parse_state_message,
        wire_move_to_move,
    )
except ModuleNotFoundError:
    pass
else:
    __all__.extend(
        [
            "OnlineMatchClient",
            "OnlineMatchProtocolError",
            "OnlineMatchStateView",
            "layout_order_from_mapping",
            "move_to_wire_dict",
            "parse_state_message",
            "wire_move_to_move",
        ]
    )
