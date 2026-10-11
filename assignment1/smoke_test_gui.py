# -*- coding: utf-8 -*-
"""Headless GUI smoke test: builds the real tkinter widgets and drives them.

We cannot look at the window from an automated script, so instead we construct
the actual ``GomokuGUI`` object, invoke the same handlers the mouse would
trigger, and assert on the resulting game state. Run with a normal desktop
session (the window is created but never enters ``mainloop``).
"""
from __future__ import annotations

import os
import sys
import time
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tkinter as tk

import gui as gui_mod
from gomoku_game import BLACK, WHITE

# 弹窗在无人值守的测试里会阻塞；换成记录调用，只在最后断言。
_dialogs = []


def _fake_showerror(title="", message="", **kw):
    _dialogs.append(("error", title, message))


def _fake_showinfo(title="", message="", **kw):
    _dialogs.append(("info", title, message))


gui_mod.messagebox.showerror = _fake_showerror
gui_mod.messagebox.showinfo = _fake_showinfo


def pump(root, seconds=0.5, until=None):
    """让 tkinter 处理事件（AI 在后台线程里，需要轮询队列）。"""
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        if until is not None and until():
            return True
        time.sleep(0.02)
    return until() if until else True


def main():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        print(f"SKIP: no display available ({exc})")
        return 0
    root.withdraw()                      # 不弹出窗口，只构建控件

    app = gui_mod.GomokuGUI(root, ai_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "gomoku_ai.py"),
        opponent_path=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "random_ai.py"))
    print("  GUI constructed")

    # ---- 默认配置 ----
    assert app.board_size == 15 and app.win_length == 5, (app.board_size, app.win_length)
    print(f"  default config N={app.board_size} K={app.win_length} mode={app.mode}")

    # ---- 切换棋盘尺寸并重绘 ----
    for preset, (n, k) in [("9x9 四连", (9, 4)), ("5x5 三连", (5, 3)),
                           ("标准 15x15 五连", (15, 5))]:
        app.var_preset.set(preset)
        app._on_preset()
        assert (app.board_size, app.win_length) == (n, k), (app.board_size, app.win_length)
        app.redraw()
    print("  presets switch + redraw OK")

    # ---- 人人对战：模拟两次点击 ----
    app.var_mode.set(gui_mod.MODE_PVP)
    app._on_mode_change()
    assert app.mode == gui_mod.MODE_PVP
    app.var_n.set(9)
    app.var_k.set(4)
    app.var_preset.set("自定义")
    app._new_game()
    assert app.board_size == 9, app.board_size

    for (r, c) in [(4, 4), (4, 5), (5, 5)]:
        x, y = app._xy(r, c)
        event = types.SimpleNamespace(x=x, y=y)
        app._on_click(event)
    assert app.game.board.get(4, 4) == BLACK
    assert app.game.board.get(4, 5) == WHITE
    assert app.game.board.get(5, 5) == BLACK
    assert app.game.move_count == 3, app.game.move_count
    print("  PvP click handling OK (3 stones placed)")

    # 点击已占用的格子应被忽略
    x, y = app._xy(4, 4)
    app._on_click(types.SimpleNamespace(x=x, y=y))
    assert app.game.move_count == 3, "occupied cell click must be ignored"
    print("  occupied-cell click ignored OK")

    # ---- 悔棋 ----
    app._undo()
    assert app.game.move_count == 2, app.game.move_count
    print("  undo OK")

    # ---- 悬停高亮 ----
    app._on_motion(types.SimpleNamespace(x=x, y=y))
    app._on_leave(None)
    print("  hover handlers OK")

    # ---- 人机对战：人类执黑，AI 执白，等 AI 走一步 ----
    app.var_mode.set(gui_mod.MODE_PVE)
    app.var_human.set("黑棋(先手)")
    app._on_mode_change()
    app.var_n.set(9)
    app.var_k.set(4)
    app.var_time.set(0.4)
    app._new_game()
    assert app.ai_player is not None
    x, y = app._xy(4, 4)
    app._on_click(types.SimpleNamespace(x=x, y=y))
    ok = pump(root, seconds=8.0, until=lambda: app.game.move_count >= 2)
    assert ok, f"AI did not answer in time (moves={app.game.move_count})"
    print(f"  PvE: AI answered, {app.game.move_count} stones on board")

    # 人类点击 AI 回合时应被忽略（等待期间 busy 或轮到 AI）
    while app.busy:
        pump(root, 0.2)
    if not app.game.is_over and app.game.current_player != app.human_color:
        before = app.game.move_count
        app._on_click(types.SimpleNamespace(x=app._xy(0, 0)[0], y=app._xy(0, 0)[1]))
        assert app.game.move_count == before or app.game.current_player == app.human_color
    print("  PvE turn ownership OK")

    # ---- 机机对战：跑几步 ----
    app.var_mode.set(gui_mod.MODE_EVE)
    app._on_mode_change()
    app.var_n.set(7)
    app.var_k.set(4)
    app.var_time.set(0.3)
    app._new_game()
    ok = pump(root, seconds=20.0, until=lambda: app.game.move_count >= 4)
    assert ok, f"EvE did not progress (moves={app.game.move_count})"
    print(f"  EvE: {app.game.move_count} stones placed automatically")

    # ---- 异常 AI（非法落子）应被界面拦下而不是崩溃 ----
    bad = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_bad_ai_tmp.py")
    with open(bad, "w", encoding="utf-8") as fh:
        fh.write("class GomokuAI:\n"
                 "    def __init__(self, player_id, board_size, win_length): pass\n"
                 "    def get_move(self, board, last_opponent_move, time_limit):\n"
                 "        return (999, 999)\n")
    app.ai_path = bad
    app.var_mode.set(gui_mod.MODE_EVE)
    app._on_mode_change()
    app.var_n.set(7)
    app.var_k.set(4)
    app.var_time.set(0.2)
    try:
        app._new_game()
        pump(root, seconds=8.0, until=lambda: bool(_dialogs))
        assert _dialogs, "illegal AI move should have raised a dialog"
        kind, title, message = _dialogs[-1]
        assert kind == "error" and "非法" in title, _dialogs[-1]
        print(f"  illegal-move AI reported, not crashed: {title}")
    finally:
        os.remove(bad)

    root.destroy()
    print("\nGUI SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
