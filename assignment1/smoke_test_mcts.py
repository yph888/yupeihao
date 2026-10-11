# -*- coding: utf-8 -*-
"""Quick checks for the MCTS variant: speed, legality, and time discipline."""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcts_ai import GomokuAI as MctsAI


def main():
    # legality + timing on a midgame position
    ai = MctsAI(1, 15, 5, seed=7)
    board = [[0] * 15 for _ in range(15)]
    for r, c, p in [(7, 7, 1), (7, 8, 2), (8, 8, 1), (6, 6, 2), (8, 7, 1)]:
        board[r][c] = p
    for limit in (0.5, 1.0):
        t0 = time.perf_counter()
        move = ai.get_move(board, (8, 7), limit)
        dt = time.perf_counter() - t0
        print(f"  limit={limit}s -> move={move} elapsed={dt:.3f}s "
              f"playouts={ai.stats['playouts']:.0f} "
              f"per_sec={ai.stats['playouts'] / max(dt, 1e-9):.0f}")
        assert dt < limit, f"over time budget: {dt} >= {limit}"
        assert board[move[0]][move[1]] == 0

    # takes an immediate win
    b2 = [[0] * 9 for _ in range(9)]
    for c in range(3):
        b2[4][c] = 1
    for c in range(3):
        b2[8][c] = 2
    ai2 = MctsAI(1, 9, 4, seed=3)
    m2 = ai2.get_move(b2, (8, 2), 0.5)
    print(f"  immediate win taken: {m2}")
    assert m2 in ((4, 3),), m2

    # blocks an immediate loss
    ai3 = MctsAI(2, 9, 4, seed=3)
    b3 = [[0] * 9 for _ in range(9)]
    for c in range(3):
        b3[4][c] = 1
    b3[0][0] = 2
    m3 = ai3.get_move(b3, (4, 2), 0.5)
    print(f"  immediate loss blocked: {m3}")
    assert m3 == (4, 3), m3

    # does not mutate input
    before = [row[:] for row in board]
    ai.get_move(board, (8, 7), 0.3)
    assert board == before, "MCTS must not mutate the input board"
    print("  input board not mutated OK")
    print("\nMCTS CHECKS PASSED")


if __name__ == "__main__":
    main()
