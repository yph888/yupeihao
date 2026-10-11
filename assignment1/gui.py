# -*- coding: utf-8 -*-
"""gui.py -- 广义五子棋图形界面（tkinter，标准库自带，无第三方依赖）。

功能
----
* 三种对战模式：人人 / 人机 / 机机，界面上一键切换；
* 棋盘大小 N 与连珠数 K 可在界面上配置（含常见预设），胜负与和棋判定
  完全来自 ``gomoku_game.GomokuGame``，与 CLI、裁判程序同源；
* 鼠标点击落子，悬停高亮，最后一手用红色小方块标记；
* AI 在后台线程里思考，界面不会卡死，并显示每步耗时；
* 支持悔棋、重开、以及"机机对战"连续开局。

依赖：仅 Python 标准库（tkinter）。
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_cli import AIPlayer, load_ai_class, resolve_player
from gomoku_game import (BLACK, EMPTY, WHITE, GomokuGame, Move, opponent)

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- 配色

COLOR_BG = "#e8b96a"          # 木色棋盘
COLOR_LINE = "#5a3a12"
COLOR_BLACK = "#101010"
COLOR_WHITE = "#f7f7f7"
COLOR_LAST = "#d62828"
COLOR_HOVER = "#3a6ea5"
COLOR_STAR = "#5a3a12"

MODE_PVP, MODE_PVE, MODE_EVE = "人人对战", "人机对战", "机机对战"

#: (显示文本, N, K) 预设
PRESETS: List[Tuple[str, int, int]] = [
    ("标准 15x15 五连", 15, 5),
    ("13x13 五连", 13, 5),
    ("11x11 五连", 11, 5),
    ("9x9 四连", 9, 4),
    ("7x7 四连", 7, 4),
    ("5x5 三连", 5, 3),
    ("自定义", -1, -1),
]


class GomokuGUI:
    """主窗口。"""

    def __init__(self, root: tk.Tk, ai_path: Optional[str] = None,
                 opponent_path: Optional[str] = None) -> None:
        self.root = root
        self.root.title("广义五子棋 (m,n,k)-Game — BME1322 Assignment 1")
        self.root.resizable(False, False)

        # 黑方 AI：默认本作业的评赛 AI
        self.ai_path = ai_path or os.path.join(HERE, "gomoku_ai.py")
        # 白方 AI（机机对战）：默认同样是评赛 AI，双方同强度自我对战。
        # 注意：默认**不是** random_ai.py —— 随机 AI 只作为"弱基线对手"选项，
        # 若默认使用它，机机对战会出现"白棋完全不防守"的观感。
        self.opponent_path = opponent_path or os.path.join(HERE, "gomoku_ai.py")

        self.board_size = 15
        self.win_length = 5
        self.time_limit = 5.0
        self.game = GomokuGame(self.board_size, self.win_length)
        self.mode = MODE_PVE
        self.human_color = BLACK
        self.ai_player: Optional[AIPlayer] = None
        self.ai2_player: Optional[AIPlayer] = None
        self.busy = False                      # AI 正在思考
        self.hover: Optional[Move] = None
        self.move_log: List[str] = []
        self._result_queue: "queue.Queue" = queue.Queue()
        self.cell = 44
        self.margin = 44

        self._build_widgets()
        self._layout_board()
        self._new_game()

    # -------------------------------------------------- 界面构建
    def _build_widgets(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(outer, background=COLOR_BG,
                                highlightthickness=1,
                                highlightbackground="#8a6a3a")
        self.canvas.grid(row=0, column=0, rowspan=3, sticky="nsew")
        self.canvas.bind("<Button-1>", self._on_click)
        self.canvas.bind("<Motion>", self._on_motion)
        self.canvas.bind("<Leave>", self._on_leave)

        panel = ttk.Frame(outer, padding=(10, 0, 0, 0))
        panel.grid(row=0, column=1, sticky="n")

        # ---- 模式 ----
        box_mode = ttk.LabelFrame(panel, text="对战模式", padding=6)
        box_mode.pack(fill="x", pady=(0, 6))
        self.var_mode = tk.StringVar(value=self.mode)
        for text in (MODE_PVP, MODE_PVE, MODE_EVE):
            ttk.Radiobutton(box_mode, text=text, value=text,
                            variable=self.var_mode,
                            command=self._on_mode_change).pack(anchor="w")

        # ---- 配置 ----
        box_cfg = ttk.LabelFrame(panel, text="棋盘配置 (N x N, K 连)", padding=6)
        box_cfg.pack(fill="x", pady=(0, 6))
        self.var_preset = tk.StringVar(value=PRESETS[0][0])
        combo = ttk.Combobox(box_cfg, textvariable=self.var_preset, width=18,
                             state="readonly",
                             values=[p[0] for p in PRESETS])
        combo.pack(anchor="w")
        combo.bind("<<ComboboxSelected>>", self._on_preset)

        row_n = ttk.Frame(box_cfg)
        row_n.pack(anchor="w", pady=(4, 0))
        ttk.Label(row_n, text="N =").pack(side="left")
        self.var_n = tk.IntVar(value=self.board_size)
        ttk.Spinbox(row_n, from_=5, to=25, width=5, textvariable=self.var_n,
                    command=self._on_manual_scale).pack(side="left", padx=(2, 10))
        ttk.Label(row_n, text="K =").pack(side="left")
        self.var_k = tk.IntVar(value=self.win_length)
        ttk.Spinbox(row_n, from_=3, to=25, width=5, textvariable=self.var_k,
                    command=self._on_manual_scale).pack(side="left", padx=2)

        row_t = ttk.Frame(box_cfg)
        row_t.pack(anchor="w", pady=(4, 0))
        ttk.Label(row_t, text="每步时限(秒) =").pack(side="left")
        self.var_time = tk.DoubleVar(value=self.time_limit)
        ttk.Spinbox(row_t, from_=0.2, to=30.0, increment=0.5, width=6,
                    textvariable=self.var_time).pack(side="left", padx=2)

        row_h = ttk.Frame(box_cfg)
        row_h.pack(anchor="w", pady=(4, 0))
        ttk.Label(row_h, text="人机模式人类执:").pack(side="left")
        self.var_human = tk.StringVar(value="黑棋(先手)")
        ttk.Combobox(row_h, textvariable=self.var_human, width=10, state="readonly",
                     values=["黑棋(先手)", "白棋(后手)"]).pack(side="left", padx=2)

        # ---- AI 选择（机机对战的白方 / 人机对战的电脑方） ----
        box_ai = ttk.LabelFrame(panel, text="电脑对手 (白方 AI)", padding=6)
        box_ai.pack(fill="x", pady=(0, 6))
        self._opponent_choices = self._discover_opponents()
        self.var_opponent = tk.StringVar(value=self._label_for_path(self.opponent_path))
        ttk.Combobox(box_ai, textvariable=self.var_opponent, width=20,
                     state="readonly",
                     values=[label for label, _ in self._opponent_choices]
                     ).pack(anchor="w")
        ttk.Button(box_ai, text="应用对手", command=self._on_opponent_change
                   ).pack(anchor="w", pady=(4, 0))
        self.var_opp_desc = tk.StringVar(value="")
        ttk.Label(box_ai, textvariable=self.var_opp_desc, wraplength=220,
                  justify="left", foreground="#444").pack(anchor="w", pady=(4, 0))

        # ---- 按钮 ----
        box_btn = ttk.LabelFrame(panel, text="操作", padding=6)
        box_btn.pack(fill="x", pady=(0, 6))
        ttk.Button(box_btn, text="新开一局", command=self._new_game).pack(fill="x")
        ttk.Button(box_btn, text="悔棋", command=self._undo).pack(fill="x", pady=2)
        ttk.Button(box_btn, text="退出", command=self.root.destroy).pack(fill="x")

        # ---- 状态 ----
        box_status = ttk.LabelFrame(panel, text="状态", padding=6)
        box_status.pack(fill="both", expand=True)
        self.var_status = tk.StringVar(value="准备开始")
        ttk.Label(box_status, textvariable=self.var_status, wraplength=220,
                  justify="left").pack(anchor="w")
        self.var_ai_info = tk.StringVar(value="")
        ttk.Label(box_status, textvariable=self.var_ai_info, wraplength=220,
                  justify="left", foreground="#444").pack(anchor="w", pady=(4, 0))

        self.log = tk.Text(panel, height=8, width=30, state="disabled",
                           font=("Consolas", 8))
        self.log.pack(fill="both", expand=True, pady=(6, 0))

    def _layout_board(self) -> None:
        """按 N 计算格子尺寸与画布大小（边距 = 一格宽，留出坐标标注）。"""
        n = self.board_size
        self.canvas.update_idletasks()
        available = 720
        self.cell = max(22, min(48, available // (n + 1)))
        self.margin = self.cell
        dim = self.cell * (n + 1)
        self.canvas.config(width=dim, height=dim)

    # -------------------------------------------------- 绘制
    def _xy(self, row: int, col: int) -> Tuple[int, int]:
        return (self.margin + col * self.cell, self.margin + row * self.cell)

    def _cell_at(self, x: int, y: int) -> Optional[Move]:
        n = self.board_size
        col = round((x - self.margin) / self.cell)
        row = round((y - self.margin) / self.cell)
        if 0 <= row < n and 0 <= col < n:
            cx, cy = self._xy(row, col)
            if abs(cx - x) <= self.cell * 0.5 and abs(cy - y) <= self.cell * 0.5:
                return (row, col)
        return None

    def redraw(self) -> None:
        cv = self.canvas
        cv.delete("all")
        n = self.board_size
        line_from = self.margin
        line_to = self.margin + (n - 1) * self.cell
        for i in range(n):
            x = self.margin + i * self.cell
            cv.create_line(x, line_from, x, line_to, fill=COLOR_LINE, width=1)
            cv.create_line(line_from, x, line_to, x, fill=COLOR_LINE, width=1)
        # 星位（仅传统 15/13/9 路棋盘）
        if n in (9, 13, 15, 19):
            pts = [n // 4, n // 2, n - 1 - n // 4] if n >= 13 else [n // 2]
            if n == 9:
                pts = [2, 4, 6]
            for r in pts:
                for c in pts:
                    x, y = self._xy(r, c)
                    cv.create_oval(x - 3, y - 3, x + 3, y + 3,
                                   fill=COLOR_STAR, outline=COLOR_STAR)
        # 坐标
        for i in range(n):
            x, y = self._xy(i, 0)
            cv.create_text(self.margin * 0.45, y, text=str(i),
                           font=("Segoe UI", max(7, self.cell // 4)))
            cv.create_text(x, self.margin * 0.45, text=str(i),
                           font=("Segoe UI", max(7, self.cell // 4)))

        radius = max(6, int(self.cell * 0.42))
        for r in range(n):
            for c in range(n):
                value = self.game.board.get(r, c)
                if value == EMPTY:
                    continue
                x, y = self._xy(r, c)
                fill = COLOR_BLACK if value == BLACK else COLOR_WHITE
                cv.create_oval(x - radius, y - radius, x + radius, y + radius,
                               fill=fill, outline="#333", width=1)
        # 悬停提示
        if self.hover and not self.game.is_over and not self.busy \
                and self.game.board.is_empty(*self.hover):
            x, y = self._xy(*self.hover)
            cv.create_oval(x - radius, y - radius, x + radius, y + radius,
                           outline=COLOR_HOVER, width=2, dash=(3, 2))
        # 最后一手标记
        if self.game.last_move:
            r, c = self.game.last_move
            x, y = self._xy(r, c)
            s = max(3, radius // 3)
            cv.create_rectangle(x - s, y - s, x + s, y + s,
                                fill=COLOR_LAST, outline=COLOR_LAST)

    def _log(self, text: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    # -------------------------------------------------- 对手选择
    def _discover_opponents(self) -> List[Tuple[str, str]]:
        """扫描目录，列出所有可作为对手的 AI 文件：[(显示名, 路径), ...]。

        显示名形如 ``gomoku_ai.py — Segment-AlphaBeta-v1``，把文件名与
        AI 自己声明的 NAME 都列出来，避免"看起来是同一个 AI"的误会。
        若某个文件的 NAME 说明它是随机/弱基线，会额外标注出来。
        """
        choices: List[Tuple[str, str]] = []
        candidates = [self.opponent_path, self.ai_path]
        candidates += sorted(
            os.path.join(HERE, name)
            for name in os.listdir(HERE) if name.endswith("_ai.py"))
        seen: set = set()
        for path in candidates:
            if not path:
                continue
            real = os.path.abspath(path)
            if real in seen or not os.path.isfile(real):
                continue
            seen.add(real)
            try:
                _, name = load_ai_class(real)
            except Exception:
                continue                      # 语法错误/缺少类的文件直接跳过
            base = os.path.basename(real)
            tag = ""
            if "random" in base.lower() or "random" in name.lower():
                tag = "（弱：随机落子，仅作基线）"
            choices.append((f"{base} — {name}{tag}", real))
        if not choices:                       # 极端兜底：至少能选自己
            choices.append((os.path.basename(self.ai_path), self.ai_path))
        return choices

    def _label_for_path(self, path: str) -> str:
        real = os.path.abspath(path)
        for label, candidate in self._opponent_choices:
            if os.path.abspath(candidate) == real:
                return label
        return self._opponent_choices[0][0]

    def _on_opponent_change(self) -> None:
        """应用用户选择的对手 AI 并重开一局。"""
        label = self.var_opponent.get()
        for text, path in self._opponent_choices:
            if text == label:
                self.opponent_path = path
                break
        self.var_opp_desc.set(f"当前对手文件：{os.path.basename(self.opponent_path)}")
        self._log(f"切换对手 -> {os.path.basename(self.opponent_path)}")
        self._new_game()

    # -------------------------------------------------- 事件
    def _on_motion(self, event) -> None:
        cell = self._cell_at(event.x, event.y)
        if cell != self.hover:
            self.hover = cell
            self.redraw()

    def _on_leave(self, _event) -> None:
        self.hover = None
        self.redraw()

    def _on_preset(self, _event=None) -> None:
        for text, n, k in PRESETS:
            if text == self.var_preset.get() and n > 0:
                self.var_n.set(n)
                self.var_k.set(k)
                self._new_game()
                return

    def _on_manual_scale(self) -> None:
        self.var_preset.set("自定义")
        self._new_game()

    def _on_mode_change(self) -> None:
        self.mode = self.var_mode.get()
        self._new_game()

    def _on_click(self, event) -> None:
        if self.busy or self.game.is_over:
            return
        if not self._human_turn():
            return
        cell = self._cell_at(event.x, event.y)
        if cell is None or not self.game.is_legal(cell):
            return
        self._apply_move(cell)
        self._maybe_ai_move()

    # -------------------------------------------------- 对局流程
    def _human_turn(self) -> bool:
        if self.mode == MODE_PVP:
            return True
        if self.mode == MODE_PVE:
            return self.game.current_player == self.human_color
        return False                       # 机机模式不接受点击

    def _new_game(self) -> None:
        try:
            n = int(self.var_n.get())
            k = int(self.var_k.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("配置错误", "N 与 K 必须是整数")
            return
        if n < 3:
            messagebox.showerror("配置错误", "棋盘边长 N 必须 >= 3")
            return
        if not (3 <= k <= n):
            messagebox.showerror("配置错误", "必须满足 3 <= K <= N")
            return
        try:
            limit = float(self.var_time.get())
        except (tk.TclError, ValueError):
            limit = 5.0
        self.time_limit = max(0.1, limit)
        self.board_size, self.win_length = n, k
        self.game = GomokuGame(n, k)
        self.mode = self.var_mode.get()
        self.human_color = BLACK if self.var_human.get().startswith("黑") else WHITE
        self.move_log.clear()
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")
        self.busy = False
        self.hover = None

        self._layout_board()
        self.redraw()
        self._prepare_players()
        self._set_status()
        self._log(f"新局：{n}x{n}，K={k}，时限 {self.time_limit:.1f}s，模式 {self.mode}")
        self._maybe_ai_move()

    def _prepare_players(self) -> None:
        """按当前模式构造/销毁 AI 参与者。

        * 人人对战：不构造任何 AI。
        * 人机对战：``ai_player`` 执人类的对家。
        * 机机对战：``ai_player`` 执黑，``ai2_player`` 执白（来自对手选择框）。

        信息栏会把**双方**的 AI 名称与来源文件都列清楚——
        机机对战最常见的问题就是"以为两边都是自己的 AI"，其实白方是别的文件。
        """
        self.ai_player = None
        self.ai2_player = None
        self.var_ai_info.set("")
        try:
            if self.mode in (MODE_PVE, MODE_EVE):
                cls, name = load_ai_class(self.ai_path)
                color = (opponent(self.human_color)
                         if self.mode == MODE_PVE else BLACK)
                self.ai_player = AIPlayer(cls, color, name,
                                          self.board_size, self.win_length)
            if self.mode == MODE_EVE:
                cls2, name2 = load_ai_class(self.opponent_path)
                self.ai2_player = AIPlayer(cls2, WHITE, name2,
                                           self.board_size, self.win_length)
        except Exception as exc:                      # AI 文件缺失/语法错误
            self.var_ai_info.set(f"AI 加载失败：{exc}")
            messagebox.showerror("AI 加载失败", str(exc))
            return

        info = []
        if self.ai_player is not None:
            who = "黑方" if self.ai_player.player_id == BLACK else "白方"
            info.append(f"{who} AI：{self.ai_player.name}"
                        f" [{os.path.basename(self.ai_path)}]")
        if self.ai2_player is not None:
            info.append(f"白方 AI：{self.ai2_player.name}"
                        f" [{os.path.basename(self.opponent_path)}]")
        if self.ai2_player is not None and \
                os.path.abspath(self.ai_path) == os.path.abspath(self.opponent_path):
            info.append("双方为同一份 AI（同强度自我对战，先手占优）")
        self.var_ai_info.set("\n".join(info))
        self.var_opp_desc.set(
            f"当前对手文件：{os.path.basename(self.opponent_path)}")

    def _set_status(self) -> None:
        if self.game.is_over:
            text = self.game.result_text()
            if self.game.winner == EMPTY:
                text = "和棋（棋盘已满）"
            else:
                who = "黑棋" if self.game.winner == BLACK else "白棋"
                text = f"{who}获胜！({self.game.win_reason})"
            self.var_status.set(f"{text}  共 {self.game.move_count} 手")
        else:
            who = "黑棋 X" if self.game.current_player == BLACK else "白棋 O"
            self.var_status.set(f"轮到 {who}　已下 {self.game.move_count} 手")

    def _apply_move(self, move: Move) -> None:
        player = self.game.current_player
        self.game.play(move)
        who = "X" if player == BLACK else "O"
        self.move_log.append(f"{len(self.move_log) + 1}. {who} {move}")
        self._log(f"{len(self.move_log):>3}. {who} {move}")
        self.redraw()
        self._set_status()

    def _maybe_ai_move(self) -> None:
        """若当前轮到 AI，则在后台线程里思考。"""
        if self.game.is_over or self.busy:
            return
        actor: Optional[AIPlayer] = None
        if self.mode == MODE_PVE and self.game.current_player != self.human_color:
            actor = self.ai_player
        elif self.mode == MODE_EVE:
            actor = self.ai_player if self.game.current_player == BLACK \
                else self.ai2_player
        if actor is None:
            return

        self.busy = True
        board = self.game.board.to_grid()
        last = self.game.last_move_of(opponent(actor.player_id))
        limit = self.time_limit

        def worker():
            started = time.perf_counter()
            try:
                move = actor.ai.get_move(board, last, limit)
                self._result_queue.put(("move", actor, move,
                                        time.perf_counter() - started))
            except Exception as exc:                  # noqa: BLE001
                self._result_queue.put(("error", actor, exc, 0.0))

        threading.Thread(target=worker, daemon=True).start()
        self.root.after(40, self._poll_ai)

    def _poll_ai(self) -> None:
        try:
            kind, actor, payload, elapsed = self._result_queue.get_nowait()
        except queue.Empty:
            self.root.after(40, self._poll_ai)
            return

        self.busy = False
        if kind == "error":
            self._log(f"AI 异常：{payload!r}")
            messagebox.showerror("AI 异常", f"{actor.name} 抛出异常：\n{payload!r}")
            return
        move = payload
        if not self.game.is_legal(move):
            self._log(f"AI 非法落子：{move!r}")
            messagebox.showerror("AI 非法落子", f"{actor.name} 返回了非法坐标 {move!r}")
            return
        actor.note_elapsed(elapsed)
        self._log(f"    {actor.name} 思考 {elapsed:.2f}s")
        self._apply_move(move)
        self._maybe_ai_move()          # 机机模式：直接续下一手

    def _undo(self) -> None:
        if self.busy:
            return
        if self.mode == MODE_PVE:
            steps = 2 if self.game.move_count >= 2 else 1
        elif self.mode == MODE_EVE:
            steps = 2 if self.game.move_count >= 2 else 1
        else:
            steps = 1
        if not self.game.undo(steps):
            return
        for _ in range(min(steps, len(self.move_log))):
            self.move_log.pop()
        self._log(f"悔棋 {steps} 手")
        self.redraw()
        self._set_status()


def run(ai_path: Optional[str] = None,
        opponent_path: Optional[str] = None) -> int:
    """启动 GUI；无图形环境时返回错误码而不是抛异常。"""
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"无法启动图形界面（{exc}）。请改用命令行入口：python main_cli.py",
              file=sys.stderr)
        return 1
    try:                        # Windows 下让界面更清晰
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    GomokuGUI(root, ai_path, opponent_path)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
