"""Main PySide6 application window."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.agents.actor_critic_agent import ActorCriticAgent
from src.env.rules import BLUE, RED

from .board_widget import BoardWidget
from .controller import GameController, make_agent
from .policy_panel import PolicyPanel
from .replay_widget import ReplayController


class MainWindow(QMainWindow):
    """Einstein chess simulator, AI battle, and replay window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Einstein Chess AI")
        self.controller = GameController()
        self.replay = ReplayController()
        self.checkpoint_path: str | None = None
        self.auto_timer = QTimer(self)
        self.auto_timer.timeout.connect(self._auto_tick)

        self.board = BoardWidget()
        self.board.cell_clicked.connect(self._cell_clicked)
        self.policy_panel = PolicyPanel()
        self.status = QLabel()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Human vs Human", "Human vs AI", "AI vs AI", "Replay"])
        self.red_combo = QComboBox()
        self.blue_combo = QComboBox()
        for combo in (self.red_combo, self.blue_combo):
            combo.addItems(["Human", "Random", "RuleBased", "ActorCritic"])
        self.blue_combo.setCurrentText("RuleBased")
        self.speed = QSlider()
        self.speed.setMinimum(100)
        self.speed.setMaximum(1500)
        self.speed.setValue(500)

        new_button = QPushButton("New Game")
        new_button.clicked.connect(self._new_game)
        load_model_button = QPushButton("Load Model")
        load_model_button.clicked.connect(self._load_model)
        load_replay_button = QPushButton("Load Replay")
        load_replay_button.clicked.connect(self._load_replay)
        prev_button = QPushButton("Prev")
        prev_button.clicked.connect(self._replay_prev)
        next_button = QPushButton("Next")
        next_button.clicked.connect(self._replay_next)
        play_button = QPushButton("Play/Pause")
        play_button.clicked.connect(self._toggle_auto)

        controls = QVBoxLayout()
        for widget in (
            QLabel("Mode"),
            self.mode_combo,
            QLabel("Red Agent"),
            self.red_combo,
            QLabel("Blue Agent"),
            self.blue_combo,
            new_button,
            load_model_button,
            load_replay_button,
            prev_button,
            next_button,
            play_button,
            QLabel("Replay speed"),
            self.speed,
            self.status,
        ):
            controls.addWidget(widget)
        controls.addStretch(1)

        main = QHBoxLayout()
        main.addWidget(self.board, stretch=3)
        side = QVBoxLayout()
        side.addLayout(controls)
        side.addWidget(self.policy_panel, stretch=1)
        main.addLayout(side, stretch=2)
        root = QWidget()
        root.setLayout(main)
        self.setCentralWidget(root)
        self._refresh()

    def _new_game(self) -> None:
        red = make_agent(self.red_combo.currentText(), self.checkpoint_path)
        blue = make_agent(self.blue_combo.currentText(), self.checkpoint_path)
        self.controller.new_game(red, blue)
        self.mode_combo.setCurrentText(self._infer_mode())
        self._refresh()
        self._auto_tick()

    def _infer_mode(self) -> str:
        if self.red_combo.currentText() == "Human" and self.blue_combo.currentText() == "Human":
            return "Human vs Human"
        if self.red_combo.currentText() != "Human" and self.blue_combo.currentText() != "Human":
            return "AI vs AI"
        return "Human vs AI"

    def _load_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load checkpoint", "checkpoints", "PyTorch (*.pt)")
        if path:
            self.checkpoint_path = path

    def _load_replay(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load game record", "data/game_records", "JSON (*.json)")
        if path:
            self.replay.load(path)
            self.mode_combo.setCurrentText("Replay")
            self.board.set_position(self.replay.current_board())
            self.status.setText("Replay loaded")

    def _cell_clicked(self, row: int, col: int) -> None:
        if self.mode_combo.currentText() == "Replay":
            return
        moved = self.controller.handle_cell_click(row, col)
        self._refresh()
        if moved:
            self._auto_tick()

    def _auto_tick(self) -> None:
        if self.mode_combo.currentText() == "Replay":
            if self.auto_timer.isActive():
                self._replay_next()
            return
        moved = self.controller.step_ai_if_needed()
        if moved:
            self._refresh()
            QTimer.singleShot(150, self._auto_tick)

    def _toggle_auto(self) -> None:
        if self.auto_timer.isActive():
            self.auto_timer.stop()
        else:
            self.auto_timer.start(self.speed.value())
            self._auto_tick()

    def _replay_next(self) -> None:
        board = self.replay.next()
        if board is not None:
            self.board.set_position(board)
            self.status.setText(f"Replay step {self.replay.index}")

    def _replay_prev(self) -> None:
        board = self.replay.previous()
        if board is not None:
            self.board.set_position(board)
            self.status.setText(f"Replay step {self.replay.index}")

    def _refresh(self) -> None:
        env = self.controller.env
        self.board.set_position(env.get_observation()["board"], self.controller.legal_targets(), env.last_move)
        text = (
            f"Player: {env.current_player}  Dice: {env.dice}  "
            f"Candidates: {env.get_candidate_pieces()}  "
            f"Red: {env.clock.remaining[RED]:.0f}s Blue: {env.clock.remaining[BLUE]:.0f}s"
        )
        if env.winner:
            text += f"  Winner: {env.winner} ({env.win_reason})"
        self.status.setText(text)
        agent = self.controller.agents.get(env.current_player)
        if isinstance(agent, ActorCriticAgent) and agent.last_diagnostics:
            self.policy_panel.set_probabilities(agent.last_diagnostics.probabilities)

