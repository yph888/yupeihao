# -*- coding: utf-8 -*-
"""test_gomoku.py -- 单元测试（标准库 unittest，无需第三方依赖）。

运行::

    python -m unittest test_gomoku -v
    python test_gomoku.py

覆盖范围：
  * 棋局逻辑：落子、K 连判定（含长连）、和棋、悔棋、(N,K) 配置校验；
  * 统一协议：GomokuAI 构造签名、get_move 返回合法坐标、不修改传入棋盘；
  * 战术正确性：能立即取胜、能封堵对手、空盘走中心；
  * 鲁棒性：畸形输入、极小/极大棋盘、极短时限都不抛异常且返回合法点；
  * 多配置自适应：不硬编码 N 与 K（同一份代码在多种 (N,K) 下正确工作）。
"""

from __future__ import annotations

import os
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gomoku_ai import GomokuAI
from gomoku_game import (BLACK, EMPTY, WHITE, Board, GomokuGame, check_win,
                         count_line, find_winning_move, opponent,
                         validate_config)

EMPTY_BOARD_15 = [[EMPTY] * 15 for _ in range(15)]


def blank(n=15):
    return [[EMPTY] * n for _ in range(n)]


def place(board, moves):
    """在 Board 或二维列表上快速摆子。"""
    for r, c, p in moves:
        if hasattr(board, "set") and not isinstance(board, list):
            board.set(r, c, p)
        else:
            board[r][c] = p
    return board


# ================================================================ 棋局逻辑

class TestBoard(unittest.TestCase):
    def test_place_and_query(self):
        b = Board(9)
        self.assertTrue(b.is_empty(3, 4))
        b.set(3, 4, BLACK)
        self.assertFalse(b.is_empty(3, 4))
        self.assertEqual(b.get(3, 4), BLACK)
        self.assertEqual(b.occupied, 1)

    def test_double_placement_rejected(self):
        b = Board(9)
        b.set(1, 1, BLACK)
        with self.assertRaises(ValueError):
            b.set(1, 1, WHITE)

    def test_round_trip_grid(self):
        grid = blank(7)
        grid[2][3] = BLACK
        grid[6][0] = WHITE
        b = Board.from_grid(grid)
        self.assertEqual(b.to_grid(), grid)

    def test_small_board_rejected(self):
        with self.assertRaises(ValueError):
            Board(2)

    def test_candidates_are_near_stones(self):
        b = Board(15)
        b.set(7, 7, BLACK)
        cand = set(b.candidate_moves(radius=1))
        self.assertIn((6, 6), cand)
        self.assertIn((8, 8), cand)
        self.assertNotIn((0, 0), cand)
        self.assertEqual(len(cand), 8)          # 3x3 邻域去掉自身
        self.assertNotIn((7, 7), cand)

    def test_candidates_empty_board_is_center(self):
        b = Board(15)
        self.assertEqual(b.candidate_moves(), [(7, 7)])

    def test_empty_cells_enumerates_all(self):
        b = Board(5)
        b.set(0, 0, BLACK)
        empties = list(b.empty_cells())
        self.assertEqual(len(empties), 24)
        self.assertNotIn((0, 0), empties)


class TestWinDetection(unittest.TestCase):
    def test_horizontal(self):
        g = GomokuGame(15, 5)
        for c in range(4):
            g.play((7, c))          # black
            g.play((9, c))          # white
        g.play((7, 4))              # black completes 5
        self.assertEqual(g.winner, BLACK)
        self.assertTrue(g.is_over)

    def test_vertical(self):
        g = GomokuGame(15, 5)
        for r in range(4):
            g.play((r, 3))
            g.play((r, 10))
        g.play((4, 3))
        self.assertEqual(g.winner, BLACK)

    def test_diagonal_down_right(self):
        g = GomokuGame(15, 5)
        for i in range(4):
            g.play((i, i))
            g.play((i, 10))
        g.play((4, 4))
        self.assertEqual(g.winner, BLACK)

    def test_diagonal_down_left(self):
        g = GomokuGame(15, 5)
        for i in range(4):
            g.play((i, 10 - i))
            g.play((i, 0))
        g.play((4, 6))
        self.assertEqual(g.winner, BLACK)

    def test_overline_counts_as_win(self):
        """长连同样判胜：白必须先堵住一端的"活四"，黑再从另一端接成 6 连。

        摆法：黑 (7,1..4) 四连，白堵 (7,5)；黑下 (7,0) 时已形成 5 连，
        为了验证 6 连也能判胜，这里把黑摆成 (6,0) 与 (7,0..4) —— 直接构造
        一个"落子后形成 6 连"的静态局面，用 check_win 判定。
        """
        b = Board(15)
        place(b, [(7, c, BLACK) for c in range(5)])          # 已有 5 连
        # 再在延长线上补一子，序列长度为 6，仍应判胜
        b.set(7, 5, BLACK)
        self.assertEqual(count_line(b, 7, 0, 0, 1, BLACK), 6)
        self.assertTrue(check_win(b, 7, 5, BLACK, 5))
        self.assertTrue(check_win(b, 7, 2, BLACK, 5))        # 中间任一点都算

    def test_four_in_a_row_is_not_a_win_at_k5(self):
        b = Board(15)
        place(b, [(7, c, BLACK) for c in range(4)])
        self.assertFalse(check_win(b, 7, 0, BLACK, 5))

    def test_k_adapts(self):
        """K=4 时四连即胜；K=6 时五连不算胜。"""
        g4 = GomokuGame(9, 4)
        for c in range(3):
            g4.play((4, c))
            g4.play((8, c))
        g4.play((4, 3))
        self.assertEqual(g4.winner, BLACK)

        g6 = GomokuGame(15, 6)
        for c in range(5):
            g6.play((4, c))
            g6.play((8, c))
        self.assertNotEqual(g6.winner, BLACK)
        self.assertFalse(g6.is_over)

    def test_count_line_both_directions(self):
        """count_line 沿两个方向延伸，统计整条连续段（不只是单侧）。"""
        b = Board(15)
        place(b, [(7, 5, BLACK), (7, 6, BLACK), (7, 8, BLACK)])
        # (7,7) 为空，因此向左是独立的一段 (7,5)(7,6)，向右是 (7,8)
        self.assertEqual(count_line(b, 7, 6, 0, 1, BLACK), 2)
        self.assertEqual(count_line(b, 7, 8, 0, 1, BLACK), 1)
        # 补上 (7,7) 后，从中间任一子出发都能看到整条 4 连
        b.set(7, 7, BLACK)
        self.assertEqual(count_line(b, 7, 5, 0, 1, BLACK), 4)
        self.assertEqual(count_line(b, 7, 7, 0, 1, BLACK), 4)
        self.assertEqual(count_line(b, 7, 8, 0, 1, BLACK), 4)
        # 竖直方向不受影响
        self.assertEqual(count_line(b, 7, 5, 1, 0, BLACK), 1)

    def test_find_winning_move(self):
        b = Board(9)
        place(b, [(4, 2, BLACK), (4, 3, BLACK), (4, 4, BLACK)])
        self.assertIn(find_winning_move(b, BLACK, 4), [(4, 1), (4, 5)])
        self.assertIsNone(find_winning_move(b, WHITE, 4))


class TestGameFlow(unittest.TestCase):
    def test_draw_on_full_board(self):
        """3x3 棋盘 K=3：填满且无人三连时应判和棋。

        这里直接用静态局面验证和棋判定的两个入口：
        ``Board.is_full`` 与 ``GomokuGame.result_text``。
        """
        from gomoku_game import Board as B
        b = B.from_grid([[1, 2, 1], [1, 2, 2], [2, 1, 1]])
        self.assertTrue(b.is_full())
        self.assertFalse(check_win(b, 0, 0, BLACK, 3))
        self.assertFalse(check_win(b, 2, 2, BLACK, 3))
        self.assertFalse(check_win(b, 0, 1, WHITE, 3))
        self.assertFalse(check_win(b, 1, 2, WHITE, 3))

    def test_draw_detected_when_board_fills(self):
        """走满一个 3x3 棋盘且无人连成 3 子时，游戏应判为和棋。"""
        g = GomokuGame(3, 3)
        # 手数顺序：黑 (0,0) 白 (0,1) 黑 (0,2) 白 (1,1) 黑 (1,0)
        #           白 (2,0) 黑 (2,1) 白 (1,2) 黑 (2,2)
        for move in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 0),
                     (2, 0), (2, 1), (1, 2), (2, 2)]:
            if g.is_over:
                break
            g.play(move)
        self.assertTrue(g.is_over)
        if g.winner == EMPTY:
            self.assertTrue(g.board.is_full())
            self.assertIn("Draw", g.result_text())

    def test_undo_restores_state(self):
        g = GomokuGame(15, 5)
        g.play((7, 7))
        g.play((7, 8))
        g.play((8, 8))
        self.assertEqual(g.move_count, 3)
        self.assertTrue(g.undo(3))
        self.assertEqual(g.move_count, 0)
        self.assertEqual(g.current_player, BLACK)
        self.assertEqual(g.board.occupied, 0)
        self.assertFalse(g.undo())

    def test_undo_after_win(self):
        g = GomokuGame(9, 3)
        for c in range(2):
            g.play((0, c))
            g.play((5, c))
        g.play((0, 2))
        self.assertEqual(g.winner, BLACK)
        g.undo()
        self.assertEqual(g.winner, EMPTY)
        self.assertFalse(g.is_over)
        self.assertEqual(g.current_player, BLACK)

    def test_illegal_moves_rejected(self):
        g = GomokuGame(9, 4)
        g.play((4, 4))
        for bad in [(4, 4), (-1, 0), (0, 9), (9, 9), (1.5, 2), (True, 2),
                    (1,), "x", None]:
            with self.assertRaises(ValueError, msg=f"{bad!r} should be rejected"):
                g.play(bad)

    def test_last_move_of(self):
        g = GomokuGame(9, 4)
        self.assertIsNone(g.last_move_of(BLACK))
        g.play((1, 1))              # black
        g.play((2, 2))              # white
        self.assertEqual(g.last_move_of(BLACK), (1, 1))
        self.assertEqual(g.last_move_of(WHITE), (2, 2))

    def test_reset(self):
        g = GomokuGame(9, 4)
        g.play((0, 0))
        g.reset()
        self.assertEqual(g.move_count, 0)
        self.assertEqual(g.board.occupied, 0)

    def test_opponent_helper(self):
        self.assertEqual(opponent(BLACK), WHITE)
        self.assertEqual(opponent(WHITE), BLACK)


class TestConfigValidation(unittest.TestCase):
    def test_valid(self):
        for n, k in [(5, 3), (5, 5), (15, 5), (9, 4), (19, 5), (7, 7), (15, 3)]:
            validate_config(n, k)

    def test_invalid(self):
        for n, k in [(4, 5), (15, 2), (15, 16), (0, 0), (5, 6), (-1, 3)]:
            with self.assertRaises(ValueError):
                validate_config(n, k)


# ================================================================ 协议 AI

class TestProtocolCompliance(unittest.TestCase):
    def test_constructor_signature(self):
        for pid in (BLACK, WHITE):
            ai = GomokuAI(pid, 15, 5)
            self.assertEqual(ai.player_id, pid)
            self.assertEqual(ai.board_size, 15)
            self.assertEqual(ai.win_length, 5)

    def test_invalid_constructor_args(self):
        with self.assertRaises(ValueError):
            GomokuAI(3, 15, 5)          # 非法棋色
        with self.assertRaises(ValueError):
            GomokuAI(1, 15, 2)          # K 太小
        with self.assertRaises(ValueError):
            GomokuAI(1, 15, 16)         # K > N

    def test_returns_legal_tuple(self):
        ai = GomokuAI(1, 15, 5)
        board = blank(15)
        place(board, [(7, 7, BLACK), (7, 8, WHITE), (8, 8, BLACK)])
        move = ai.get_move([row[:] for row in board], (8, 8), 0.5)
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)
        r, c = move
        self.assertIsInstance(r, int)
        self.assertIsInstance(c, int)
        self.assertNotIsInstance(r, bool)
        self.assertTrue(0 <= r < 15 and 0 <= c < 15)
        self.assertEqual(board[r][c], EMPTY)

    def test_does_not_mutate_input(self):
        ai = GomokuAI(2, 9, 4)
        board = blank(9)
        place(board, [(4, 4, BLACK), (4, 5, WHITE)])
        snapshot = [row[:] for row in board]
        ai.get_move(board, (4, 5), 0.5)
        self.assertEqual(board, snapshot)

    def test_first_move_is_center(self):
        for n in (5, 9, 15):
            ai = GomokuAI(1, n, min(5, n))
            move = ai.get_move(blank(n), None, 1.0)
            self.assertEqual(move, (n // 2, n // 2))

    def test_handles_full_board_gracefully(self):
        """棋盘已满时不应抛异常（裁判会判和棋）。"""
        ai = GomokuAI(1, 3, 3)
        board = [[1, 2, 1], [2, 1, 2], [2, 1, 2]]
        move = ai.get_move(board, None, 0.2)     # 不抛异常即通过
        self.assertEqual(len(move), 2)

    def test_robust_to_malformed_board(self):
        """异常输入不应抛异常，且返回合法坐标。"""
        ai = GomokuAI(1, 9, 4)
        for bad in ([[0] * 9], [], [[0] * 3 for _ in range(3)],
                    [[None] * 9 for _ in range(9)],
                    [[0] * 9 for _ in range(8)] + [[0] * 8 + [0]]):
            move = ai.get_move(bad, None, 0.3)
            r, c = move
            self.assertTrue(0 <= r < 9 and 0 <= c < 9, (bad, move))

    def test_robust_to_weird_time_limit(self):
        ai = GomokuAI(1, 9, 4)
        board = blank(9)
        place(board, [(4, 4, BLACK)])
        for limit in (0, -1, float("nan"), float("inf"), None, "abc", 0.001):
            move = ai.get_move([row[:] for row in board], (4, 4), limit)
            self.assertTrue(0 <= move[0] < 9 and 0 <= move[1] < 9)

    def test_respects_tight_time_limit(self):
        """极短时限也必须按时返回（超时即判负）。"""
        ai = GomokuAI(1, 15, 5)
        board = blank(15)
        place(board, [(7, 7, BLACK), (7, 8, WHITE), (8, 8, BLACK),
                      (6, 6, WHITE), (8, 7, BLACK)])
        for limit in (0.05, 0.1, 0.25):
            start = time.perf_counter()
            ai.get_move([row[:] for row in board], (8, 7), limit)
            elapsed = time.perf_counter() - start
            self.assertLess(elapsed, limit, f"limit={limit} took {elapsed:.3f}s")


class TestTactics(unittest.TestCase):
    def test_takes_immediate_win(self):
        for k, n in ((5, 15), (4, 9), (3, 7)):
            with self.subTest(k=k, n=n):
                ai = GomokuAI(1, n, k)
                board = blank(n)
                for c in range(k - 1):
                    board[4][c] = BLACK
                    board[n - 1][c] = WHITE      # 对手也有 k-1 连，但轮到我
                move = ai.get_move([row[:] for row in board], (n - 1, k - 2), 1.0)
                self.assertEqual(move, (4, k - 1),
                                 f"k={k}: 应自己成 {k} 连而不是去防守")

    def test_blocks_immediate_loss(self):
        for k, n in ((5, 15), (4, 9), (3, 7)):
            ai = GomokuAI(2, n, k)
            board = blank(n)
            for c in range(k - 1):
                board[4][c] = BLACK
            board[0][0] = WHITE
            move = ai.get_move([row[:] for row in board], (4, k - 2), 1.0)
            self.assertEqual(move, (4, k - 1),
                             f"k={k}: must block the open {k - 1}")

    def test_blocks_both_open_ends(self):
        ai = GomokuAI(2, 15, 5)
        board = blank(15)
        for c in range(1, 5):
            board[7][c] = BLACK       # 黑在 (7,1..4)，两端 (7,0)/(7,5) 都能成 5
        board[0][0] = WHITE
        move = ai.get_move([row[:] for row in board], (7, 4), 1.0)
        self.assertIn(move, [(7, 0), (7, 5)])

    def test_wins_instead_of_blocking(self):
        """自己一手能赢时，不应去防守。"""
        ai = GomokuAI(1, 9, 4)
        board = blank(9)
        for c in range(3):
            board[4][c] = BLACK       # 黑可 (4,3) 取胜
        for c in range(3):
            board[8][c] = WHITE       # 白也有三连，但轮到黑走
        move = ai.get_move([row[:] for row in board], (8, 2), 1.0)
        self.assertEqual(move, (4, 3))


class TestOpeningBook(unittest.TestCase):
    """开局库必须以"局面平静"为前提，不能抢走封堵对手的决策权。

    回归背景：机机对战中出现过"后手白棋完全不防守"的观感，
    其中一部分原因是开局库只看手数（≤4 手）就返回固定的中心偏好着法，
    于是对手在前几手沿一条线连下时，白棋会无视威胁去走偏好点。
    """

    def test_book_used_when_position_is_calm(self):
        for n, k in ((15, 5), (9, 4)):
            with self.subTest(board=n, win=k):
                ai = GomokuAI(WHITE, n, k)
                board = blank(n)
                board[n // 2][n // 2] = BLACK          # 对手只有 1 子
                move = ai.get_move([row[:] for row in board], (n // 2, n // 2), 1.0)
                self.assertEqual(ai.stats["depth"], 0,
                                 "平静开局应直接走开局库，不进入搜索")
                self.assertTrue(0 <= move[0] < n and 0 <= move[1] < n)
                self.assertEqual(board[move[0]][move[1]], EMPTY)

    def test_book_yields_when_opponent_has_two_in_a_row(self):
        """对手已成两连时必须让位给封堵/搜索，不能走固定开局点。"""
        n, k = 15, 5
        ai = GomokuAI(WHITE, n, k)
        board = blank(n)
        board[7][3] = BLACK
        board[7][4] = BLACK
        board[0][0] = WHITE                            # 白棋不是首手
        book_point = (7, 7)                            # 白方开局库的首选点
        move = ai.get_move([row[:] for row in board], (7, 4), 0.5)
        self.assertNotEqual(move, book_point,
                            "开局库不应压过对两连的应对")
        self.assertEqual(board[move[0]][move[1]], EMPTY)

    def test_book_yields_when_own_side_has_two_in_a_row(self):
        """自己已成两连时也应进入搜索（继续发展棋型），而非机械走开局点。"""
        n, k = 15, 5
        ai = GomokuAI(BLACK, n, k)
        board = blank(n)
        board[7][3] = BLACK
        board[7][4] = BLACK
        board[0][0] = WHITE
        move = ai.get_move([row[:] for row in board], (0, 0), 0.5)
        self.assertEqual(board[move[0]][move[1]], EMPTY)
        self.assertGreaterEqual(ai.stats["depth"], 1,
                                "两连局面应进入搜索而不是直接返回开局库着法")

    def test_book_guard_threshold_is_two(self):
        """_book_is_safe 的阈值确认为 2（窗口内最大子数 >= 2 即视为战斗）。

        注意：``_model.reset_from`` 接收的是**一维**列表。
        """
        n, k = 15, 5
        ai = GomokuAI(BLACK, n, k)
        flat = [EMPTY] * (n * n)
        ai._model.reset_from(flat, BLACK)
        self.assertTrue(ai._book_is_safe())            # 空盘平静
        flat[7 * n + 7] = WHITE                        # 对手 1 子仍平静
        ai._model.reset_from(flat, BLACK)
        self.assertTrue(ai._book_is_safe())
        flat[7 * n + 8] = WHITE                        # 对手 2 连 -> 不再平静
        ai._model.reset_from(flat, BLACK)
        self.assertFalse(ai._book_is_safe())
        flat[7 * n + 7] = EMPTY                        # 换成本方 2 连 -> 同样不平静
        flat[7 * n + 6] = BLACK
        flat[7 * n + 5] = BLACK
        ai._model.reset_from(flat, BLACK)
        self.assertFalse(ai._book_is_safe())

    def test_book_only_applies_to_first_few_moves(self):
        """开局库只在前 4 手生效，之后必须完全交给搜索。"""
        n, k = 15, 5
        ai = GomokuAI(BLACK, n, k)
        flat = [EMPTY] * (n * n)
        # 空盘时 _opening_move 返回中心（一般性启发），
        # 而 get_move 在空盘时会更早直接返回中心、根本不调用它。
        self.assertEqual(ai._opening_move(flat, n * n), (n // 2, n // 2))
        flat[7 * n + 7] = BLACK
        self.assertIsNotNone(ai._opening_move(flat, n * n - 1))
        for i in range(6):                                  # 5 手以上：不再使用
            flat[i * n] = BLACK if i % 2 == 0 else WHITE
        self.assertIsNone(ai._opening_move(flat, n * n - 6))


class TestWhiteDefence(unittest.TestCase):
    """后手（白棋）必须会防守——对应"机机对战白棋乱下"的回归测试。"""

    def test_blocks_threat_ladder(self):
        # K=5：黑棋两连 / 三连 / 四连，白棋都必须封堵在同一条线上
        for run_len, allowed in ((2, (2, 5)), (3, (2, 6)), (4, (2, 7))):
            with self.subTest(black_run=run_len):
                n, k = 15, 5
                ai = GomokuAI(WHITE, n, k)
                board = blank(n)
                for c in range(3, 3 + run_len):
                    board[7][c] = BLACK
                board[0][0] = WHITE
                move = ai.get_move([row[:] for row in board], (7, 3 + run_len - 1), 0.5)
                self.assertEqual(board[move[0]][move[1]], EMPTY)
                self.assertTrue(
                    move[0] == 7 and move[1] in allowed,
                    f"黑棋 {run_len} 连，白棋应封堵 {allowed}，实际 {move}")

    def test_never_ignores_immediate_loss(self):
        for k, n in ((5, 15), (4, 9), (3, 7), (6, 15)):
            with self.subTest(win=k):
                ai = GomokuAI(WHITE, n, k)
                board = blank(n)
                for c in range(k - 1):
                    board[4][c] = BLACK
                board[n - 1][n - 1] = WHITE
                move = ai.get_move([row[:] for row in board], (4, k - 2), 1.0)
                self.assertEqual(move, (4, k - 1),
                                 f"K={k}: 白棋必须堵住 (4,{k - 1})")

    def test_black_and_white_play_symmetric_solid_opening(self):
        """双方都不应在对方成两连时走无关位置。"""
        for player in (BLACK, WHITE):
            with self.subTest(player=player):
                n, k = 15, 5
                ai = GomokuAI(player, n, k)
                board = blank(n)
                foe = 3 - player
                board[7][3] = foe
                board[7][4] = foe
                move = ai.get_move([row[:] for row in board], (7, 4), 0.5)
                self.assertEqual(board[move[0]][move[1]], EMPTY)
                # 要么封堵，要么形成自己的强威胁（两者都是合理应对）
                self.assertTrue(move[0] == 7 or ai.stats["depth"] >= 1)


class TestAdaptivity(unittest.TestCase):
    """不硬编码 N 与 K：同一份实现在多种配置下都必须正常工作。"""

    def test_many_configs(self):
        configs = [(5, 3), (5, 5), (7, 4), (9, 4), (9, 9), (11, 5),
                   (13, 5), (15, 3), (15, 5), (15, 6), (19, 5)]
        for n, k in configs:
            with self.subTest(board=n, win=k):
                ai = GomokuAI(1, n, k)
                board = blank(n)
                center = n // 2
                place(board, [(center, center, BLACK),
                              (center, center + 1, WHITE),
                              (center + 1, center + 1, BLACK)])
                move = ai.get_move([row[:] for row in board],
                                   (center + 1, center + 1), 0.3)
                r, c = move
                self.assertTrue(0 <= r < n and 0 <= c < n)
                self.assertEqual(board[r][c], EMPTY)

    def test_install_win_length_used_in_scoring(self):
        """不同 K 下内部评估权重确实随 K 变化（不是写死五连）。"""
        w3 = GomokuAI(1, 9, 3)._weights
        w5 = GomokuAI(1, 9, 5)._weights
        w7 = GomokuAI(1, 9, 7)._weights
        self.assertEqual(len(w3), 5)
        self.assertEqual(len(w5), 7)
        self.assertEqual(len(w7), 9)
        self.assertNotEqual(w3, w5)
        # 成 K 连的权重必须是最大值
        self.assertEqual(w5[5], 10_000_000)

    def test_full_game_selfplay_completes(self):
        """两个实例互相对弈，必须能在有限手数内正常结束。"""
        n, k = 9, 4
        ai_black = GomokuAI(BLACK, n, k)
        ai_white = GomokuAI(WHITE, n, k)
        game = GomokuGame(n, k)
        limit = 0.2
        while not game.is_over:
            ai = ai_black if game.current_player == BLACK else ai_white
            move = ai.get_move(game.board.to_grid(),
                               game.last_move_of(opponent(game.current_player)),
                               limit)
            self.assertTrue(game.is_legal(move), f"illegal move {move}")
            game.play(move)
            self.assertLessEqual(game.move_count, n * n)
        self.assertTrue(game.is_over)
        self.assertIn(game.winner, (EMPTY, BLACK, WHITE))


# ================================================================ MCTS 变体

class TestMctsVariant(unittest.TestCase):
    """进阶方向的 MCTS 实现也必须满足同一套协议。"""

    @classmethod
    def setUpClass(cls):
        try:
            from mcts_ai import GomokuAI as MctsAI
        except Exception as exc:                      # pragma: no cover
            raise unittest.SkipTest(f"mcts_ai 不可用: {exc}")
        cls.MctsAI = MctsAI

    def test_legal_and_timely(self):
        ai = self.MctsAI(1, 9, 4, seed=1)
        board = blank(9)
        place(board, [(4, 4, BLACK), (4, 5, WHITE)])
        start = time.perf_counter()
        move = ai.get_move([row[:] for row in board], (4, 5), 0.5)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 0.5)
        self.assertEqual(board[move[0]][move[1]], EMPTY)

    def test_takes_win_and_blocks(self):
        ai = self.MctsAI(1, 9, 4, seed=2)
        board = blank(9)
        for c in range(3):
            board[4][c] = BLACK
        for c in range(3):
            board[8][c] = WHITE
        self.assertEqual(ai.get_move([row[:] for row in board], (8, 2), 0.5),
                         (4, 3))

        ai2 = self.MctsAI(2, 9, 4, seed=2)
        board2 = blank(9)
        for c in range(3):
            board2[4][c] = BLACK
        board2[0][0] = WHITE
        self.assertEqual(ai2.get_move([row[:] for row in board2], (4, 2), 0.5),
                         (4, 3))

    def test_first_move_center(self):
        ai = self.MctsAI(1, 15, 5, seed=3)
        self.assertEqual(ai.get_move(blank(15), None, 0.3), (7, 7))


if __name__ == "__main__":
    unittest.main(verbosity=2)
