from einstein_chess.engine import PlayerColor
from einstein_chess.players import HumanPlayer
from einstein_chess.ui import EinsteinChessApp


def main() -> None:
    players = {
        PlayerColor.RED: HumanPlayer("\u7ea2\u65b9"),
        PlayerColor.BLUE: HumanPlayer("\u84dd\u65b9"),
    }
    app = EinsteinChessApp(players=players)
    app.mainloop()


if __name__ == "__main__":
    main()

