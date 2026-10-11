# -*- coding: utf-8 -*-
"""Regression test: in AI-vs-AI mode the White player must actually defend.

This covers the reported bug "机机对战时后手白棋完全不防守乱下".

Root cause (fixed): the GUI/CLI defaulted the White AI to ``random_ai.py``,
so in AI-vs-AI mode the second player really was a random-move baseline.
The default is now the same Alpha-Beta AI, and the opponent is selectable.

These tests assert:
  1. the default opponent in AI-vs-AI mode is NOT the random baseline;
  2. when Black builds a K-1 threat, White's reply blocks it (both engines);
  3. White keeps blocking over a whole threat sequence;
  4. White never plays a move that ignores an immediate Black win.
"""
from __future__ import annotations

import os
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import tkinter as tk

import gui as gui_mod
from game_cli import load_ai_class
from gomoku_game import BLACK, EMPTY, WHITE, Board, GomokuGame, opponent

MAIN = os.path.join(HERE, "gomoku_ai.py")
RANDOM = os.path.join(HERE, "random_ai.py")

gui_mod.messagebox.showerror = lambda *a, **k: None


def fresh_gui(**kwargs):
    root = tk.Tk()
    root.withdraw()
    app = gui_mod.GomokuGUI(root, **kwargs)
    return root, app


def pump(root, app, seconds, cond):
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def set_position(app, stones, to_move, last_move):
    """直接把 GUI 的棋局设成给定局面。"""
    n = app.board_size
    grid = [[EMPTY] * n for _ in range(n)]
    for r, c, p in stones:
        grid[r][c] = p
    app.game.board = Board.from_grid(grid)
    app.game.history = list(last_move)
    app.game.current_player = to_move
    app.game.winner = EMPTY
    app.game.win_reason = ""
    app.redraw()
    app._set_status()


# ---------------------------------------------------------------- 测试 1

def test_default_opponent_is_not_random():
    root, app = fresh_gui()
    try:
        assert os.path.basename(app.opponent_path) != "random_ai.py", \
            f"默认对手不应是随机 AI，实际是 {app.opponent_path}"
        print(f"  1) AI 默认对手 = {os.path.basename(app.opponent_path)}  OK")
        # 随机 AI 必须仍然可选（作为弱基线）
        labels = [lbl for lbl, _ in app._opponent_choices]
        assert any("random_ai.py" in l for l in labels), labels
        print("     随机基线仍可在下拉框中选择  OK")
    finally:
        root.destroy()


# ---------------------------------------------------------------- 测试 2

def test_white_blocks_in_gui():
    """黑棋四连（K=5 时差一子成五），白棋必须堵其中一端。"""
    root, app = fresh_gui()
    try:
        app.var_mode.set(gui_mod.MODE_EVE)
        app.var_n.set(15)
        app.var_k.set(5)
        app.var_time.set(0.5)
        app._on_mode_change()
        # 黑 (7,3)..(7,6) 四连；白已在 (0,0) 落子；轮到白走
        stones = [(7, c, BLACK) for c in range(3, 7)] + [(0, 0, WHITE)]
        set_position(app, stones, WHITE, [(0, 0), (7, 6)])
        app._maybe_ai_move()
        ok = pump(root, app, 15.0, lambda: app.game.move_count > 2)
        assert ok, "白棋没有落子"
        move = app.game.history[-1]
        assert move in [(7, 2), (7, 7)], f"白棋应堵 (7,2)/(7,7)，实际 {move}"
        print(f"  2) GUI 机机对战：白棋封堵黑棋四连 -> {move}  OK")
    finally:
        root.destroy()


# ---------------------------------------------------------------- 测试 3

#: (说明, 黑棋在 row 7 上的连子, 允许的白棋应手列)
THREAT_LADDER = [
    ("黑 2 连", [(7, 3), (7, 4)], (2, 5)),
    ("黑 3 连", [(7, 3), (7, 4), (7, 5)], (2, 6)),
    ("黑 4 连（K-1，必堵）", [(7, 3), (7, 4), (7, 5), (7, 6)], (2, 7)),
]


def test_white_blocks_threat_ladder():
    """威胁逐级升高时，白棋每一步都必须封堵在这一条线上（或反制）。"""
    cls, _ = load_ai_class(MAIN)
    n = 15
    for label, black_stones, allowed_cols in THREAT_LADDER:
        board = Board(n)
        for r, c in black_stones:
            board.set(r, c, BLACK)
        board.set(0, 0, WHITE)                 # 白棋有一颗孤子，不是首手
        ai = cls(WHITE, n, 5)
        reply = ai.get_move(board.to_grid(), black_stones[-1], 0.5)
        assert board.is_empty(*reply), f"白棋非法落子 {reply}"
        on_line = reply[0] == 7 and reply[1] in allowed_cols
        counter = _self_threatens(board, reply, WHITE, 5, n)
        assert on_line or counter, (
            f"{label}: 黑棋在 row7 = {black_stones}，白棋却下在 {reply}"
            f"（既未封堵 {allowed_cols} 也未形成反制）")
        print(f"      {label:<22} -> 白棋 {reply}  "
              f"（封堵={on_line} 反制={counter}）")
    print("  3) 威胁逐级升高时白棋始终应对  OK")


def test_opening_book_yields_to_threats():
    """开局库不能在对手已经连子时抢走决策权。"""
    cls, _ = load_ai_class(MAIN)
    n = 15
    # 黑棋两连 + 白棋孤子：开局库的固定着法是 (7,7)，必须被封堵逻辑覆盖
    board = Board(n)
    board.set(7, 3, BLACK)
    board.set(7, 4, BLACK)
    board.set(0, 0, WHITE)
    ai = cls(WHITE, n, 5)
    reply = ai.get_move(board.to_grid(), (7, 4), 0.5)
    assert reply != (7, 7), f"开局库未被威胁覆盖：白棋下在 {reply}"
    assert reply in [(7, 2), (7, 5)] or _self_threatens(board, reply, WHITE, 5, n), \
        f"白棋应封堵 (7,2)/(7,5)，实际 {reply}"
    print(f"  4) 开局库让位于威胁封堵 -> 白棋 {reply}  OK")

    # 平静局面下开局库仍应生效（黑棋只有 1 子）
    calm = Board(n)
    calm.set(7, 7, BLACK)
    ai2 = cls(WHITE, n, 5)
    book_move = ai2.get_move(calm.to_grid(), (7, 7), 0.5)
    assert ai2.stats["depth"] == 0, "平静开局应直接走开局库，不进入搜索"
    print(f"     平静开局仍走开局库 -> 白棋 {book_move}  OK")


def _self_threatens(board, move, player, k, n):
    """这一手是否形成了自己的 K-1 威胁（反制）。"""
    board.set(move[0], move[1], player)
    try:
        for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
            run = 1
            for s in (1, -1):
                rr, cc = move[0] + s * dr, move[1] + s * dc
                while 0 <= rr < n and 0 <= cc < n and board.get(rr, cc) == player:
                    run += 1
                    rr += s * dr
                    cc += s * dc
            if run >= k - 1:
                return True
        return False
    finally:
        board.set(move[0], move[1], EMPTY)


# ---------------------------------------------------------------- 测试 5

def test_white_never_ignores_immediate_loss():
    """白棋在任何局面下都不能无视"黑棋下一手就成 K 连"。"""
    cls, _ = load_ai_class(MAIN)
    for k, n in ((5, 15), (4, 9), (3, 7)):
        ai = cls(WHITE, n, k)
        board = Board(n)
        for c in range(k - 1):
            board.set(4, c, BLACK)             # 黑棋 K-1 连
        board.set(0, 0, WHITE)
        reply = ai.get_move(board.to_grid(), (4, k - 2), 1.0)
        assert reply == (4, k - 1), f"K={k}: 白棋应堵 (4,{k - 1})，实际 {reply}"
        print(f"  5) K={k}: 白棋不放过必败点 -> {reply}  OK")


if __name__ == "__main__":
    print("=== AI 机机对战：白棋防守回归测试 ===")
    test_default_opponent_is_not_random()
    test_white_blocks_in_gui()
    test_white_blocks_threat_ladder()
    test_opening_book_yields_to_threats()
    test_white_never_ignores_immediate_loss()
    print("\nALL WHITE-DEFENCE REGRESSION TESTS PASSED")
