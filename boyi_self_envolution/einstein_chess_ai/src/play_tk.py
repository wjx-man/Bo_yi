"""Launch the Tkinter GUI."""

from __future__ import annotations

from src.gui.tk_window import TkEinsteinApp


def main() -> None:
    app = TkEinsteinApp()
    app.mainloop()


if __name__ == "__main__":
    main()
