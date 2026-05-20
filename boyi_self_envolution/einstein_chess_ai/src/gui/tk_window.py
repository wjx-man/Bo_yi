"""Tkinter application window for Einstein chess."""

from __future__ import annotations

from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.agents.actor_critic_agent import ActorCriticAgent
from src.env.rules import BLUE, DIRECTION_NAMES, RED, id_to_action

from .controller import GameController, make_agent


AGENT_KINDS = ("Human", "Random", "RuleBased", "ActorCritic", "Minimax", "AlphaBeta", "MCTS")
CELL_COUNT = 5


class TkEinsteinApp(tk.Tk):
    """Dependency-light Tkinter UI for playing against a trained model."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Einstein Chess AI - Tkinter")
        self.minsize(980, 650)
        self.controller = GameController()
        self.project_root = Path(__file__).resolve().parents[2]
        default_checkpoint = self.project_root / "checkpoints" / "model.pt"
        self.checkpoint_path: str | None = str(default_checkpoint) if default_checkpoint.exists() else None
        self.square = 96
        self.board_size = self.square * CELL_COUNT
        self.auto_job: str | None = None

        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill=tk.BOTH, expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=0)
        root.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(root, width=self.board_size, height=self.board_size, bg="#f7f7f2", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self._cell_clicked)

        side = ttk.Frame(root, padding=(14, 0, 0, 0))
        side.grid(row=0, column=1, sticky="ns")

        self.mode_var = tk.StringVar(value="Human vs AI")
        self.red_var = tk.StringVar(value="Human")
        self.blue_var = tk.StringVar(value="ActorCritic")
        self.speed_var = tk.IntVar(value=350)
        self.seed_var = tk.StringVar(value="")
        self.device_var = tk.StringVar(value="auto")
        self.actor_mode_var = tk.StringVar(value="greedy")
        self.temperature_var = tk.StringVar(value="1.0")
        self.minimax_depth_var = tk.StringVar(value="2")
        self.alphabeta_depth_var = tk.StringVar(value="3")
        self.mcts_simulations_var = tk.StringVar(value="128")
        self.mcts_rollout_steps_var = tk.StringVar(value="80")
        self.mcts_exploration_var = tk.StringVar(value="1.4")
        self.checkpoint_var = tk.StringVar(value=self.checkpoint_path or "No model loaded")
        self.status_var = tk.StringVar()

        self._add_labeled_combo(side, "Mode", self.mode_var, ("Human vs Human", "Human vs AI", "AI vs AI"))
        self._add_labeled_combo(side, "Red Agent", self.red_var, AGENT_KINDS)
        self._add_labeled_combo(side, "Blue Agent", self.blue_var, AGENT_KINDS)

        ttk.Button(side, text="New Game", command=self._new_game).pack(fill=tk.X, pady=(12, 4))
        ttk.Button(side, text="Load Model", command=self._load_model).pack(fill=tk.X, pady=4)
        ttk.Button(side, text="AI Step", command=self._auto_tick).pack(fill=tk.X, pady=4)
        ttk.Button(side, text="Play/Pause AI", command=self._toggle_auto).pack(fill=tk.X, pady=4)

        ttk.Label(side, text="AI speed").pack(anchor="w", pady=(12, 0))
        ttk.Scale(side, from_=100, to=1500, variable=self.speed_var, orient=tk.HORIZONTAL).pack(fill=tk.X)
        self._build_param_panel(side)
        ttk.Label(side, textvariable=self.checkpoint_var, wraplength=330).pack(fill=tk.X, pady=(12, 6))
        ttk.Label(side, textvariable=self.status_var, wraplength=330).pack(fill=tk.X, pady=(6, 12))

        ttk.Label(side, text="Policy probabilities").pack(anchor="w")
        self.policy = ttk.Treeview(side, columns=("action", "move", "prob"), show="headings", height=18)
        self.policy.heading("action", text="Action")
        self.policy.heading("move", text="Move")
        self.policy.heading("prob", text="Prob")
        self.policy.column("action", width=58, anchor="center")
        self.policy.column("move", width=92, anchor="center")
        self.policy.column("prob", width=72, anchor="e")
        for action in range(18):
            piece, direction = id_to_action(action)
            self.policy.insert("", "end", iid=str(action), values=(action, f"{piece}-{DIRECTION_NAMES[direction]}", "0.000"))
        self.policy.pack(fill=tk.BOTH, expand=True)

    def _add_labeled_combo(self, parent: ttk.Frame, label: str, variable: tk.StringVar, values: tuple[str, ...]) -> None:
        ttk.Label(parent, text=label).pack(anchor="w", pady=(6, 0))
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.pack(fill=tk.X)

    def _build_param_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Agent parameters", padding=8)
        frame.pack(fill=tk.X, pady=(12, 0))
        for col in range(2):
            frame.columnconfigure(col, weight=1)

        self._add_param_entry(frame, "Seed", self.seed_var, 0, 0)
        self._add_labeled_combo_grid(frame, "Device", self.device_var, ("auto", "cuda", "cpu"), 0, 1)
        self._add_labeled_combo_grid(frame, "Model mode", self.actor_mode_var, ("greedy", "sample"), 1, 0)
        self._add_param_entry(frame, "Temperature", self.temperature_var, 1, 1)
        self._add_param_entry(frame, "Minimax depth", self.minimax_depth_var, 2, 0)
        self._add_param_entry(frame, "AlphaBeta depth", self.alphabeta_depth_var, 2, 1)
        self._add_param_entry(frame, "MCTS sims", self.mcts_simulations_var, 3, 0)
        self._add_param_entry(frame, "MCTS rollout", self.mcts_rollout_steps_var, 3, 1)
        self._add_param_entry(frame, "MCTS explore", self.mcts_exploration_var, 4, 0)

    def _add_param_entry(self, parent: ttk.Frame, label: str, variable: tk.StringVar, row: int, col: int) -> None:
        box = ttk.Frame(parent)
        box.grid(row=row, column=col, sticky="ew", padx=3, pady=3)
        ttk.Label(box, text=label).pack(anchor="w")
        ttk.Entry(box, textvariable=variable, width=12).pack(fill=tk.X)

    def _add_labeled_combo_grid(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        values: tuple[str, ...],
        row: int,
        col: int,
    ) -> None:
        box = ttk.Frame(parent)
        box.grid(row=row, column=col, sticky="ew", padx=3, pady=3)
        ttk.Label(box, text=label).pack(anchor="w")
        ttk.Combobox(box, textvariable=variable, values=values, state="readonly", width=10).pack(fill=tk.X)

    def _new_game(self) -> None:
        try:
            params = self._agent_params()
            red = make_agent(self.red_var.get(), self.checkpoint_path, params)
            blue = make_agent(self.blue_var.get(), self.checkpoint_path, params)
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Cannot start game", str(exc))
            return
        self.controller.new_game(red, blue)
        self.mode_var.set(self._infer_mode())
        self._refresh()
        self._schedule_ai()

    def _agent_params(self) -> dict:
        seed = self._optional_int(self.seed_var.get(), "Seed")
        return {
            "seed": seed,
            "device": self.device_var.get(),
            "actor_mode": self.actor_mode_var.get(),
            "temperature": self._float_value(self.temperature_var.get(), "Temperature", minimum=1.0e-6),
            "minimax_depth": self._int_value(self.minimax_depth_var.get(), "Minimax depth", minimum=1),
            "alphabeta_depth": self._int_value(self.alphabeta_depth_var.get(), "AlphaBeta depth", minimum=1),
            "mcts_simulations": self._int_value(self.mcts_simulations_var.get(), "MCTS sims", minimum=1),
            "mcts_rollout_steps": self._int_value(self.mcts_rollout_steps_var.get(), "MCTS rollout", minimum=1),
            "mcts_exploration": self._float_value(self.mcts_exploration_var.get(), "MCTS explore", minimum=0.0),
        }

    def _optional_int(self, text: str, label: str) -> int | None:
        text = text.strip()
        if not text:
            return None
        return self._int_value(text, label)

    def _int_value(self, text: str, label: str, minimum: int | None = None) -> int:
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer") from exc
        if minimum is not None and value < minimum:
            raise ValueError(f"{label} must be >= {minimum}")
        return value

    def _float_value(self, text: str, label: str, minimum: float | None = None) -> float:
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number") from exc
        if minimum is not None and value < minimum:
            raise ValueError(f"{label} must be >= {minimum:g}")
        return value

    def _infer_mode(self) -> str:
        red_is_human = self.red_var.get() == "Human"
        blue_is_human = self.blue_var.get() == "Human"
        if red_is_human and blue_is_human:
            return "Human vs Human"
        if not red_is_human and not blue_is_human:
            return "AI vs AI"
        return "Human vs AI"

    def _load_model(self) -> None:
        initial_dir = self.project_root / "checkpoints"
        path = filedialog.askopenfilename(
            title="Load checkpoint",
            initialdir=str(initial_dir),
            filetypes=(("PyTorch checkpoints", "*.pt"), ("All files", "*.*")),
        )
        if path:
            self.checkpoint_path = path
            self.checkpoint_var.set(path)

    def _cell_clicked(self, event: tk.Event) -> None:
        row = int(event.y // self.square)
        col = int(event.x // self.square)
        if not (0 <= row < CELL_COUNT and 0 <= col < CELL_COUNT):
            return
        moved = self.controller.handle_cell_click(row, col)
        self._refresh()
        if moved:
            self._schedule_ai()

    def _toggle_auto(self) -> None:
        if self.auto_job is not None:
            self.after_cancel(self.auto_job)
            self.auto_job = None
        else:
            self._schedule_ai()

    def _schedule_ai(self) -> None:
        if self.auto_job is None:
            self.auto_job = self.after(int(self.speed_var.get()), self._auto_tick)

    def _auto_tick(self) -> None:
        self.auto_job = None
        moved = self.controller.step_ai_if_needed()
        if moved:
            self._refresh()
            if not self.controller.env.is_terminal():
                self._schedule_ai()

    def _refresh(self) -> None:
        env = self.controller.env
        obs = env.get_observation()
        self._draw_board(obs["board"], self.controller.legal_targets(), env.last_move)
        text = (
            f"Player: {env.current_player}  Dice: {env.dice}\n"
            f"Candidates: {env.get_candidate_pieces()}\n"
            f"Red: {env.clock.remaining[RED]:.0f}s  Blue: {env.clock.remaining[BLUE]:.0f}s"
        )
        if env.winner:
            text += f"\nWinner: {env.winner} ({env.win_reason})"
        self.status_var.set(text)

        agent = self.controller.agents.get(env.current_player)
        if isinstance(agent, ActorCriticAgent) and agent.last_diagnostics:
            self._set_policy(agent.last_diagnostics.probabilities)

    def _draw_board(self, board, legal_targets=None, last_move=None) -> None:
        legal_targets = set(legal_targets or [])
        last_from = None if last_move is None else last_move.get("from_pos")
        last_to = None if last_move is None else last_move.get("to_pos")
        self.canvas.delete("all")

        for row in range(CELL_COUNT):
            for col in range(CELL_COUNT):
                x0 = col * self.square
                y0 = row * self.square
                x1 = x0 + self.square
                y1 = y0 + self.square
                color = "#ffffff" if (row + col) % 2 == 0 else "#e8ecef"
                if (row, col) in legal_targets:
                    color = "#d7f4d0"
                if (row, col) == last_from:
                    color = "#ffe0b2"
                if (row, col) == last_to:
                    color = "#fff176"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="#263238")

                value = board[row][col]
                if value:
                    fill = "#c62828" if value > 0 else "#1565c0"
                    label = f"R{value}" if value > 0 else f"B{-value}"
                    pad = 12
                    self.canvas.create_rectangle(x0 + pad, y0 + pad, x1 - pad, y1 - pad, fill=fill, outline="")
                    self.canvas.create_text((x0 + x1) / 2, (y0 + y1) / 2, text=label, fill="white", font=("Arial", 22, "bold"))

    def _set_policy(self, probs: list[float] | None) -> None:
        probs = probs or [0.0] * 18
        for idx, prob in enumerate(probs[:18]):
            piece, direction = id_to_action(idx)
            self.policy.item(str(idx), values=(idx, f"{piece}-{DIRECTION_NAMES[direction]}", f"{prob:.3f}"))
