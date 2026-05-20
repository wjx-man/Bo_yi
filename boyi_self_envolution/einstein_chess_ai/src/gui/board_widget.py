"""5x5 board widget."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget


class BoardWidget(QWidget):
    """Draw and interact with a 5x5 Einstein chess board."""

    cell_clicked = Signal(int, int)

    def __init__(self) -> None:
        super().__init__()
        self.board = [[0 for _ in range(5)] for _ in range(5)]
        self.legal_targets: set[tuple[int, int]] = set()
        self.last_from: tuple[int, int] | None = None
        self.last_to: tuple[int, int] | None = None
        self.setMinimumSize(420, 420)

    def set_position(self, board, legal_targets=None, last_move=None) -> None:
        """Update board state."""
        self.board = board
        self.legal_targets = set(legal_targets or [])
        self.last_from = None if last_move is None else last_move.get("from_pos")
        self.last_to = None if last_move is None else last_move.get("to_pos")
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        size = min(self.width(), self.height())
        margin_x = (self.width() - size) / 2
        margin_y = (self.height() - size) / 2
        cell = size / 5
        col = int((event.position().x() - margin_x) // cell)
        row = int((event.position().y() - margin_y) // cell)
        if 0 <= row < 5 and 0 <= col < 5:
            self.cell_clicked.emit(row, col)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        size = min(self.width(), self.height()) - 12
        x0 = (self.width() - size) / 2
        y0 = (self.height() - size) / 2
        cell = size / 5
        painter.fillRect(self.rect(), QColor("#f7f7f2"))
        for row in range(5):
            for col in range(5):
                x = x0 + col * cell
                y = y0 + row * cell
                color = QColor("#ffffff") if (row + col) % 2 == 0 else QColor("#e8ecef")
                if (row, col) in self.legal_targets:
                    color = QColor("#d7f4d0")
                if (row, col) == self.last_from:
                    color = QColor("#ffe0b2")
                if (row, col) == self.last_to:
                    color = QColor("#fff176")
                painter.fillRect(int(x), int(y), int(cell), int(cell), color)
                painter.setPen(QPen(QColor("#263238"), 1))
                painter.drawRect(int(x), int(y), int(cell), int(cell))
                value = self.board[row][col]
                if value:
                    painter.setBrush(QColor("#c62828") if value > 0 else QColor("#1565c0"))
                    painter.setPen(Qt.NoPen)
                    painter.drawRoundedRect(int(x + 8), int(y + 8), int(cell - 16), int(cell - 16), 6, 6)
                    painter.setPen(QColor("white"))
                    painter.setFont(QFont("Arial", max(12, int(cell * 0.24)), QFont.Bold))
                    label = f"R{value}" if value > 0 else f"B{-value}"
                    painter.drawText(int(x), int(y), int(cell), int(cell), Qt.AlignCenter, label)

