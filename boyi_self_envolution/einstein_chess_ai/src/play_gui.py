"""Launch the PySide6 GUI."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1100, 720)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

