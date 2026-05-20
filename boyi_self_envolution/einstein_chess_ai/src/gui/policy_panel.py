"""Policy probability table for Actor-Critic diagnostics."""

from __future__ import annotations

from PySide6.QtWidgets import QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from src.env.rules import DIRECTION_NAMES, id_to_action


class PolicyPanel(QWidget):
    """Display 18 actor probabilities."""

    def __init__(self) -> None:
        super().__init__()
        self.table = QTableWidget(18, 3)
        self.table.setHorizontalHeaderLabels(["Action", "Move", "Prob"])
        for action in range(18):
            piece, direction = id_to_action(action)
            self.table.setItem(action, 0, QTableWidgetItem(str(action)))
            self.table.setItem(action, 1, QTableWidgetItem(f"{piece}-{DIRECTION_NAMES[direction]}"))
            self.table.setItem(action, 2, QTableWidgetItem("0.000"))
        layout = QVBoxLayout(self)
        layout.addWidget(self.table)

    def set_probabilities(self, probs: list[float] | None) -> None:
        """Update displayed probabilities."""
        probs = probs or [0.0] * 18
        for idx, prob in enumerate(probs[:18]):
            self.table.setItem(idx, 2, QTableWidgetItem(f"{prob:.3f}"))

