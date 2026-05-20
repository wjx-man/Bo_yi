"""Replay state holder used by the main window."""

from __future__ import annotations

from src.utils.serialization import load_json


class ReplayController:
    """Load and step through saved game records."""

    def __init__(self) -> None:
        self.record = None
        self.index = -1

    def load(self, path: str) -> None:
        """Load a game record."""
        self.record = load_json(path)
        self.index = -1

    def current_board(self):
        """Return the current replay board."""
        if self.record is None:
            return None
        if self.index < 0:
            return self.record["initial_board"]
        return self.record["moves"][self.index]["board_after"]

    def next(self):
        """Advance one move."""
        if self.record is None:
            return None
        self.index = min(self.index + 1, len(self.record["moves"]) - 1)
        return self.current_board()

    def previous(self):
        """Go back one move."""
        if self.record is None:
            return None
        self.index = max(self.index - 1, -1)
        return self.current_board()

