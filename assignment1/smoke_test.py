# -*- coding: utf-8 -*-
"""Smoke test: game logic + AI protocol compliance, run from the submission dir."""
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gomoku_game import (Board, GomokuGame, check_win, validate_config,
                         EMPTY, BLACK, WHITE)
from gomoku_ai import GomokuAI


def test_game_logic():
    g = GomokuGame(9, 4)
    assert g.current_player == BLACK
    for c in range(4):
        g.play((0, c))
        if g.is_over:
            break
        g.play((5, c))
    assert g.winner == BLACK, g.winner
    assert g.move_count == 7, g.move_count
    print("  game logic: black 4-in-a-row OK ->", g.result_text())

    # draw detection on a tiny board: 3x3, K=3 -> 9 moves, first player wins on move 5
    g2 = GomokuGame(3, 3)
    assert g2.win_length == 3
    print("  config validation OK")

    # draw on 4x4 with K=4 is impossible quickly; just check undo works
    g3 = GomokuGame(15, 5)
    g3.play((7, 7))
    g3.play((7, 8))
    assert g3.board.get(7, 7) == BLACK and g3.board.get(7, 8) == WHITE
    g3.undo(2)
    assert g3.move_count == 0 and g3.board.occupied == 0
    assert g3.current_player == BLACK
    print("  undo OK")

    try:
        validate_config(15, 16)
        raise AssertionError("should have rejected K > N")
    except ValueError:
        pass
    try:
        validate_config(4, 5)
        raise AssertionError("should have rejected N < 5")
    except ValueError:
        pass
    print("  invalid config rejected OK")


def test_ai_basics():
    # 1) must take an immediate win
    ai = GomokuAI(1, 9, 4)
    board = [[0] * 9 for _ in range(9)]
    for c in range(3):
        board[4][c] = 1              # black has 3 in a row at (4,0..2)
    for c in range(3):
        board[8][c] = 2
    t0 = time.perf_counter()
    move = ai.get_move(board, (8, 2), 2.0)
    dt = time.perf_counter() - t0
    assert move in ((4, 3), (4, -1)), move
    print(f"  AI takes immediate win: {move} in {dt:.3f}s")

    # 2) must block an immediate loss
    ai2 = GomokuAI(2, 9, 4)
    board2 = [[0] * 9 for _ in range(9)]
    for c in range(3):              # opponent (black) threatens at (4,3) / (4,-1)
        board2[4][c] = 1
    board2[0][0] = 2
    move2 = ai2.get_move(board2, (4, 2), 2.0)
    assert move2 in ((4, 3),), move2
    print(f"  AI blocks immediate loss: {move2}")

    # 3) opening move is the center
    ai3 = GomokuAI(1, 15, 5)
    m3 = ai3.get_move([[0] * 15 for _ in range(15)], None, 5.0)
    assert m3 == (7, 7), m3
    print(f"  opening move = {m3}")

    # 4) timing on a realistic midgame position (15x15, K=5, 5s limit)
    ai4 = GomokuAI(1, 15, 5)
    board4 = [[0] * 15 for _ in range(15)]
    seq = [(7, 7, 1), (7, 8, 2), (8, 8, 1), (6, 6, 2), (8, 7, 1), (6, 8, 2),
           (9, 9, 1), (5, 7, 2)]
    for r, c, p in seq:
        board4[r][c] = p
    t0 = time.perf_counter()
    m4 = ai4.get_move(board4, (5, 7), 5.0)
    dt = time.perf_counter() - t0
    print(f"  midgame move = {m4} in {dt:.3f}s  nodes={ai4.stats['nodes']:.0f} "
          f"depth={ai4.stats['depth']:.0f} pv={ai4._pv[:4]}")
    assert dt < 5.0, f"exceeded time limit: {dt}"
    assert 0 <= m4[0] < 15 and 0 <= m4[1] < 15
    assert board4[m4[0]][m4[1]] == 0

    # 5) tiny board with K=N
    ai5 = GomokuAI(1, 5, 5)
    b5 = [[0] * 5 for _ in range(5)]
    m5 = ai5.get_move(b5, None, 1.0)
    print(f"  5x5 K=5 opening = {m5}")

    # 6) K=3 on 5x5
    ai6 = GomokuAI(2, 5, 3)
    b6 = [[0] * 5 for _ in range(5)]
    b6[2][2] = 1
    m6 = ai6.get_move(b6, (2, 2), 1.0)
    print(f"  5x5 K=3 reply to (2,2) = {m6}")

    # 7) does not mutate the input board
    before = [row[:] for row in board4]
    ai7 = GomokuAI(2, 15, 5)
    ai7.get_move(board4, (5, 7), 1.0)
    assert board4 == before, "AI must not modify the input board"
    print("  input board not mutated OK")


if __name__ == "__main__":
    print("== game logic ==")
    test_game_logic()
    print("== AI ==")
    test_ai_basics()
    print("\nALL SMOKE TESTS PASSED")
