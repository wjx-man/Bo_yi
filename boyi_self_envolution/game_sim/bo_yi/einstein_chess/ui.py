from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Mapping

from .engine import EinsteinGame, Move, PlayerColor, Position
from .players import HumanPlayer, PlayerAgent


class LayoutDialog(tk.Toplevel):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title("\u624b\u52a8\u5e03\u5c40")
        self.resizable(False, False)
        self.transient(master)
        self.grab_set()

        self.result: tuple[dict[int, Position], dict[int, Position]] | None = None
        self.vars: dict[PlayerColor, list[tk.StringVar]] = {
            PlayerColor.RED: [],
            PlayerColor.BLUE: [],
        }

        self._build_widgets()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.wait_visibility()
        self.focus_set()
        self.wait_window()

    def _build_widgets(self) -> None:
        body = ttk.Frame(self, padding=14)
        body.grid(row=0, column=0, sticky="nsew")

        ttk.Label(
            body,
            text=(
                "\u4e3a\u53cc\u65b9\u51fa\u53d1\u533a\u7684\u6bcf\u4e2a\u68cb\u4f4d"
                "\u6307\u5b9a 1-6 \u53f7\u68cb\u5b50\uff0c\u6bcf\u4e2a\u7f16\u53f7"
                "\u5fc5\u987b\u6070\u597d\u4f7f\u7528\u4e00\u6b21\u3002"
            ),
            wraplength=460,
            justify="left",
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        self._build_color_panel(body, PlayerColor.RED, 0)
        self._build_color_panel(body, PlayerColor.BLUE, 1)

        button_row = ttk.Frame(body)
        button_row.grid(row=2, column=0, columnspan=2, sticky="e", pady=(14, 0))
        ttk.Button(
            button_row,
            text="\u53d6\u6d88",
            command=self._cancel,
        ).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(
            button_row,
            text="\u786e\u8ba4\u5e03\u5c40",
            command=self._confirm,
        ).grid(row=0, column=1)

    def _build_color_panel(
        self, parent: ttk.Frame, color: PlayerColor, column: int
    ) -> None:
        frame = ttk.LabelFrame(
            parent,
            text=f"{color.label}\u51fa\u53d1\u533a",
            padding=10,
        )
        frame.grid(row=1, column=column, sticky="nsew", padx=(0 if column == 0 else 10, 0), pady=(12, 0))

        for index, position in enumerate(color.start_positions):
            ttk.Label(frame, text=self._format_position(position), width=6).grid(
                row=index, column=0, sticky="w", pady=(0, 4)
            )
            var = tk.StringVar(value=str(index + 1))
            self.vars[color].append(var)
            ttk.Combobox(
                frame,
                textvariable=var,
                values=("1", "2", "3", "4", "5", "6"),
                state="readonly",
                width=6,
            ).grid(row=index, column=1, sticky="w", pady=(0, 4))

    def _confirm(self) -> None:
        red_layout = self._read_layout(PlayerColor.RED)
        if red_layout is None:
            return
        blue_layout = self._read_layout(PlayerColor.BLUE)
        if blue_layout is None:
            return
        self.result = (red_layout, blue_layout)
        self.destroy()

    def _read_layout(self, color: PlayerColor) -> dict[int, Position] | None:
        numbers = [int(var.get()) for var in self.vars[color]]
        expected = list(EinsteinGame.PIECE_NUMBERS)
        if sorted(numbers) != expected:
            messagebox.showerror(
                "\u5e03\u5c40\u9519\u8bef",
                f"{color.label}\u5fc5\u987b\u6070\u597d\u4f7f\u7528 1-6 \u53f7\u5404\u4e00\u6b21\u3002",
                parent=self,
            )
            return None
        return {
            number: position
            for number, position in zip(numbers, color.start_positions)
        }

    def _cancel(self) -> None:
        self.result = None
        self.destroy()

    def _format_position(self, position: Position) -> str:
        row, col = position
        return f"{chr(ord('A') + col)}{row + 1}"


class EinsteinChessApp(tk.Tk):
    CELL_SIZE = 96
    BOARD_MARGIN = 18
    BOARD_PIXEL = EinsteinGame.BOARD_SIZE * CELL_SIZE + BOARD_MARGIN * 2

    def __init__(self, players: Mapping[PlayerColor, PlayerAgent] | None = None) -> None:
        super().__init__()
        self.title("\u7231\u6069\u65af\u5766\u68cb")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.players = dict(
            players
            or {
                PlayerColor.RED: HumanPlayer("\u7ea2\u65b9"),
                PlayerColor.BLUE: HumanPlayer("\u84dd\u65b9"),
            }
        )
        self.game = EinsteinGame()
        self.selected_position: Position | None = None
        self.highlighted_moves: list[Move] = []
        self.event_log: list[str] = []
        self.pending_job: str | None = None

        self.mode_var = tk.StringVar()
        self.turn_var = tk.StringVar()
        self.dice_var = tk.StringVar()
        self.candidate_var = tk.StringVar()
        self.result_var = tk.StringVar()
        self.status_var = tk.StringVar()

        self._build_widgets()
        self._update_mode_text()
        self.new_game(random_layout=False)

    def _build_widgets(self) -> None:
        self.configure(bg="#f3efe3")
        container = ttk.Frame(self, padding=16)
        container.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(
            container,
            width=self.BOARD_PIXEL,
            height=self.BOARD_PIXEL,
            bg="#efe5d0",
            highlightthickness=0,
        )
        self.canvas.grid(row=0, column=0, sticky="n")
        self.canvas.bind("<Button-1>", self._on_board_click)

        sidebar = ttk.Frame(container, padding=(20, 0, 0, 0))
        sidebar.grid(row=0, column=1, sticky="ns")

        ttk.Label(
            sidebar,
            text="\u7231\u6069\u65af\u5766\u68cb",
            font=("Microsoft YaHei UI", 20, "bold"),
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            sidebar,
            textvariable=self.mode_var,
            font=("Microsoft YaHei UI", 10),
        ).grid(row=1, column=0, sticky="w", pady=(4, 12))

        info_frame = ttk.LabelFrame(
            sidebar,
            text="\u5bf9\u5c40\u4fe1\u606f",
            padding=12,
        )
        info_frame.grid(row=2, column=0, sticky="ew")

        ttk.Label(info_frame, textvariable=self.turn_var).grid(row=0, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.dice_var).grid(
            row=1, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Label(info_frame, textvariable=self.candidate_var, wraplength=280).grid(
            row=2, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Label(info_frame, textvariable=self.result_var).grid(
            row=3, column=0, sticky="w", pady=(6, 0)
        )

        action_frame = ttk.LabelFrame(
            sidebar,
            text="\u64cd\u4f5c",
            padding=12,
        )
        action_frame.grid(row=3, column=0, sticky="ew", pady=(12, 0))

        ttk.Button(
            action_frame,
            text="\u65b0\u5c40\uff08\u9ed8\u8ba4\u5e03\u5c40\uff09",
            command=lambda: self.new_game(random_layout=False),
        ).grid(row=0, column=0, sticky="ew")
        ttk.Button(
            action_frame,
            text="\u65b0\u5c40\uff08\u968f\u673a\u5e03\u5c40\uff09",
            command=lambda: self.new_game(random_layout=True),
        ).grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(
            action_frame,
            text="\u65b0\u5c40\uff08\u624b\u52a8\u5e03\u5c40\uff09",
            command=lambda: self.new_game(random_layout=False, manual_layout=True),
        ).grid(row=2, column=0, sticky="ew", pady=(8, 0))

        tip_frame = ttk.LabelFrame(
            sidebar,
            text="\u72b6\u6001\u63d0\u793a",
            padding=12,
        )
        tip_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(
            tip_frame,
            textvariable=self.status_var,
            wraplength=280,
            justify="left",
        ).grid(row=0, column=0, sticky="w")

        history_frame = ttk.LabelFrame(
            sidebar,
            text="\u56de\u5408\u8bb0\u5f55",
            padding=12,
        )
        history_frame.grid(row=5, column=0, sticky="nsew", pady=(12, 0))
        sidebar.rowconfigure(5, weight=1)

        self.history_list = tk.Listbox(
            history_frame,
            width=34,
            height=13,
            font=("Consolas", 10),
            activestyle="none",
        )
        self.history_list.grid(row=0, column=0, sticky="nsew")

        rule_frame = ttk.LabelFrame(
            sidebar,
            text="\u89c4\u5219\u901f\u89c8",
            padding=12,
        )
        rule_frame.grid(row=6, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(
            rule_frame,
            justify="left",
            wraplength=280,
            text=(
                "1. \u7ea2\u65b9\u5411\u53f3/\u4e0b/\u53f3\u4e0b\u8d70\uff0c"
                "\u84dd\u65b9\u53cd\u5411\u3002\n"
                "2. \u63b7\u5230\u51e0\u53f7\u5c31\u4f18\u5148\u8d70\u51e0\u53f7\u3002\n"
                "3. \u82e5\u8be5\u7f16\u53f7\u5df2\u88ab\u5403\uff0c"
                "\u53ef\u8d70\u6700\u8fd1\u7f16\u53f7\u3002\n"
                "4. \u8d70\u5230\u5bf9\u89d2\u7ec8\u70b9\u6216"
                "\u5403\u5149\u5bf9\u624b\u5373\u80dc\u3002"
            ),
        ).grid(row=0, column=0, sticky="w")

    def new_game(self, random_layout: bool, manual_layout: bool = False) -> None:
        if manual_layout:
            dialog = LayoutDialog(self)
            if dialog.result is None:
                return
            red_layout, blue_layout = dialog.result
        else:
            red_layout = self._build_layout(PlayerColor.RED, random_layout)
            blue_layout = self._build_layout(PlayerColor.BLUE, random_layout)

        self._cancel_pending_job()
        self.selected_position = None
        self.highlighted_moves = []
        self.event_log = []

        self.game.reset(red_layout=red_layout, blue_layout=blue_layout)

        if manual_layout:
            opening_text = "\u5df2\u5f00\u59cb\u624b\u52a8\u5e03\u5c40\u65b0\u5bf9\u5c40\u3002"
        elif random_layout:
            opening_text = "\u5df2\u5f00\u59cb\u968f\u673a\u5e03\u5c40\u65b0\u5bf9\u5c40\u3002"
        else:
            opening_text = "\u5df2\u5f00\u59cb\u65b0\u5bf9\u5c40\u3002"
        self._push_log(opening_text)
        self._refresh_view()
        self.after(120, self._continue_turn)

    def _build_layout(
        self, color: PlayerColor, random_layout: bool
    ) -> dict[int, Position]:
        if random_layout:
            return EinsteinGame.random_layout_for(color, rng=self.game.rng)
        agent = self.players[color]
        return agent.choose_layout(color, color.start_positions, self.game.rng)

    def _continue_turn(self) -> None:
        self._cancel_pending_job()
        if self.game.winner is not None:
            self.status_var.set(
                f"{self.game.winner.label}\u5df2\u83b7\u80dc\uff0c"
                "\u70b9\u51fb\u6309\u94ae\u53ef\u91cd\u65b0\u5f00\u5c40\u3002"
            )
            self._refresh_view()
            return

        legal_moves = self.game.get_legal_moves()
        if not legal_moves:
            message = (
                f"{self.game.current_player.label}\u63b7\u5230 "
                f"{self.game.dice_roll}\uff0c"
                "\u4f46\u5f53\u524d\u6ca1\u6709\u5408\u6cd5\u6b65\uff0c"
                "\u7cfb\u7edf\u81ea\u52a8\u8df3\u8fc7\u3002"
            )
            self.status_var.set(message)
            self._push_log(message)
            self._refresh_view()
            self.pending_job = self.after(900, self._pass_turn)
            return

        agent = self.players[self.game.current_player]
        if agent.is_human:
            self.status_var.set(
                f"{self.game.current_player.label}"
                "\u8bf7\u5148\u70b9\u51fb\u53ef\u8d70\u68cb\u5b50\uff0c"
                "\u518d\u70b9\u51fb\u76ee\u6807\u683c\u3002"
            )
            self._refresh_view()
            return

        self.status_var.set(
            f"{self.game.current_player.label}\u601d\u8003\u4e2d..."
        )
        self._refresh_view()
        self.pending_job = self.after(300, self._play_ai_turn)

    def _pass_turn(self) -> None:
        self.pending_job = None
        self.game.pass_turn()
        self.selected_position = None
        self.highlighted_moves = []
        self._refresh_view()
        self._continue_turn()

    def _play_ai_turn(self) -> None:
        self.pending_job = None
        agent = self.players[self.game.current_player]
        legal_moves = self.game.get_legal_moves()
        move = agent.choose_move(self.game.snapshot(), legal_moves)
        self._execute_move(move)

    def _execute_move(self, move: Move) -> None:
        self.game.apply_move(move)
        self.selected_position = None
        self.highlighted_moves = []
        self._push_log(self._format_move(move))
        self._refresh_view()
        self._continue_turn()

    def _on_board_click(self, event: tk.Event) -> None:
        if self.game.winner is not None:
            return

        current_agent = self.players[self.game.current_player]
        if not current_agent.is_human:
            return

        position = self._canvas_to_position(event.x, event.y)
        if position is None:
            return

        move = self._match_highlighted_move(position)
        if move is not None:
            self._execute_move(move)
            return

        piece = self.game.get_piece(position)
        if piece is None:
            self.selected_position = None
            self.highlighted_moves = []
            self.status_var.set("\u8be5\u4f4d\u7f6e\u6ca1\u6709\u53ef\u9009\u68cb\u5b50\u3002")
            self._refresh_view()
            return

        if piece.color is not self.game.current_player:
            self.selected_position = None
            self.highlighted_moves = []
            self.status_var.set(
                "\u5f53\u524d\u53ea\u80fd\u64cd\u4f5c\u5df1\u65b9\u68cb\u5b50\u3002"
            )
            self._refresh_view()
            return

        candidate_numbers = self.game.get_candidate_numbers(
            self.game.current_player, self.game.dice_roll
        )
        if piece.number not in candidate_numbers:
            self.selected_position = None
            self.highlighted_moves = []
            self.status_var.set(
                "\u8be5\u68cb\u5b50\u4e0d\u7b26\u5408\u672c\u56de\u5408\u9ab0\u5b50\u8981\u6c42\u3002"
            )
            self._refresh_view()
            return

        self.selected_position = position
        self.highlighted_moves = [
            move
            for move in self.game.get_legal_moves()
            if move.from_position == position
        ]
        if not self.highlighted_moves:
            self.status_var.set(
                "\u8be5\u68cb\u5b50\u672c\u56de\u5408\u6ca1\u6709\u5408\u6cd5\u843d\u70b9\u3002"
            )
        else:
            self.status_var.set(
                "\u5df2\u9009\u4e2d\u68cb\u5b50\uff0c"
                "\u8bf7\u70b9\u51fb\u9ad8\u4eae\u76ee\u6807\u683c\u5b8c\u6210\u8d70\u5b50\u3002"
            )
        self._refresh_view()

    def _match_highlighted_move(self, position: Position) -> Move | None:
        for move in self.highlighted_moves:
            if move.to_position == position:
                return move
        return None

    def _refresh_view(self) -> None:
        current_color = self.game.current_player
        candidate_numbers = self.game.get_candidate_numbers(
            current_color, self.game.dice_roll
        )
        candidate_text = "\u3001".join(str(number) for number in candidate_numbers)
        if not candidate_text:
            candidate_text = "\u65e0"

        self.turn_var.set(f"\u5f53\u524d\u56de\u5408\uff1a{current_color.label}")
        self.dice_var.set(f"\u9ab0\u5b50\u7ed3\u679c\uff1a{self.game.dice_roll}")
        self.candidate_var.set(
            f"\u672c\u56de\u5408\u53ef\u52a8\u7f16\u53f7\uff1a{candidate_text}"
        )
        if self.game.winner is None:
            self.result_var.set("\u80dc\u8d1f\u72b6\u6001\uff1a\u5bf9\u5c40\u4e2d")
        else:
            self.result_var.set(
                f"\u80dc\u8d1f\u72b6\u6001\uff1a{self.game.winner.label}\u83b7\u80dc"
            )

        self._draw_board()
        self._refresh_history()

    def _refresh_history(self) -> None:
        self.history_list.delete(0, tk.END)
        visible_events = self.event_log[-14:]
        for entry in visible_events:
            self.history_list.insert(tk.END, entry)
        if visible_events:
            self.history_list.yview_moveto(1.0)

    def _draw_board(self) -> None:
        self.canvas.delete("all")
        light = "#f5ddb2"
        dark = "#dfc08e"
        current_candidates = set(
            self.game.get_candidate_numbers(self.game.current_player, self.game.dice_roll)
        )
        last_move = self.game.move_history[-1] if self.game.move_history else None

        for row in range(EinsteinGame.BOARD_SIZE):
            for col in range(EinsteinGame.BOARD_SIZE):
                x1, y1, x2, y2 = self._cell_bbox((row, col))
                fill = light if (row + col) % 2 == 0 else dark
                self.canvas.create_rectangle(
                    x1, y1, x2, y2, fill=fill, outline="#8d6e54", width=2
                )

                if (row, col) == (0, 0):
                    self.canvas.create_text(
                        x1 + 8,
                        y1 + 10,
                        text="\u7ea2\u8d77/\u84dd\u7ec8",
                        anchor="nw",
                        fill="#6c4f3d",
                        font=("Microsoft YaHei UI", 8),
                    )
                if (row, col) == (4, 4):
                    self.canvas.create_text(
                        x1 + 8,
                        y1 + 10,
                        text="\u84dd\u8d77/\u7ea2\u7ec8",
                        anchor="nw",
                        fill="#6c4f3d",
                        font=("Microsoft YaHei UI", 8),
                    )

                if last_move is not None and (
                    (row, col) == last_move.from_position
                    or (row, col) == last_move.to_position
                ):
                    self.canvas.create_rectangle(
                        x1 + 4,
                        y1 + 4,
                        x2 - 4,
                        y2 - 4,
                        outline="#2e7d32",
                        width=3,
                    )

                for move in self.highlighted_moves:
                    if move.to_position == (row, col):
                        self.canvas.create_oval(
                            x1 + 34,
                            y1 + 34,
                            x2 - 34,
                            y2 - 34,
                            fill="#2e7d32",
                            outline="",
                        )

                piece = self.game.get_piece((row, col))
                if piece is not None:
                    outline_color = "#4b3b2a"
                    outline_width = 2
                    if (
                        piece.color is self.game.current_player
                        and piece.number in current_candidates
                    ):
                        outline_color = "#ffcc33"
                        outline_width = 4
                    if self.selected_position == (row, col):
                        outline_color = "#1b5e20"
                        outline_width = 5

                    piece_fill = (
                        "#d64b4b" if piece.color is PlayerColor.RED else "#4c7ad9"
                    )
                    self.canvas.create_rectangle(
                        x1 + 16,
                        y1 + 16,
                        x2 - 16,
                        y2 - 16,
                        fill=piece_fill,
                        outline=outline_color,
                        width=outline_width,
                    )
                    self.canvas.create_text(
                        (x1 + x2) / 2,
                        (y1 + y2) / 2,
                        text=str(piece.number),
                        fill="white",
                        font=("Microsoft YaHei UI", 24, "bold"),
                    )

                for move in self.highlighted_moves:
                    if move.to_position == (row, col):
                        self.canvas.create_rectangle(
                            x1 + 8,
                            y1 + 8,
                            x2 - 8,
                            y2 - 8,
                            outline="#2e7d32",
                            width=4,
                        )

                self.canvas.create_text(
                    x1 + 8,
                    y2 - 8,
                    text=self._format_position((row, col)),
                    anchor="sw",
                    fill="#5d4836",
                    font=("Consolas", 9),
                )

    def _canvas_to_position(self, x: int, y: int) -> Position | None:
        x -= self.BOARD_MARGIN
        y -= self.BOARD_MARGIN
        if x < 0 or y < 0:
            return None
        col = x // self.CELL_SIZE
        row = y // self.CELL_SIZE
        if row >= EinsteinGame.BOARD_SIZE or col >= EinsteinGame.BOARD_SIZE:
            return None
        return (row, col)

    def _cell_bbox(self, position: Position) -> tuple[int, int, int, int]:
        row, col = position
        x1 = self.BOARD_MARGIN + col * self.CELL_SIZE
        y1 = self.BOARD_MARGIN + row * self.CELL_SIZE
        x2 = x1 + self.CELL_SIZE
        y2 = y1 + self.CELL_SIZE
        return x1, y1, x2, y2

    def _format_position(self, position: Position) -> str:
        row, col = position
        return f"{chr(ord('A') + col)}{row + 1}"

    def _format_move(self, move: Move) -> str:
        return (
            f"{len(self.event_log):>2}. "
            f"{move.color.short_label}{move.piece_number} "
            f"{self._format_position(move.from_position)} -> "
            f"{self._format_position(move.to_position)}"
        )

    def _push_log(self, message: str) -> None:
        self.event_log.append(message)

    def _update_mode_text(self) -> None:
        red_human = self.players[PlayerColor.RED].is_human
        blue_human = self.players[PlayerColor.BLUE].is_human
        if red_human and blue_human:
            mode_name = "\u4eba\u4eba\u5bf9\u6218"
        elif red_human or blue_human:
            mode_name = "\u4eba\u673a\u5bf9\u6218"
        else:
            mode_name = "\u673a\u673a\u5bf9\u6218"
        self.mode_var.set(f"\u5f53\u524d\u6a21\u5f0f\uff1a{mode_name}")

    def _cancel_pending_job(self) -> None:
        if self.pending_job is not None:
            self.after_cancel(self.pending_job)
            self.pending_job = None

    def _on_close(self) -> None:
        self._cancel_pending_job()
        self.destroy()
