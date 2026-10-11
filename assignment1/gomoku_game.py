# -*- coding: utf-8 -*-
"""gomoku_game.py -- 广义五子棋 (m,n,k)-Game 棋局逻辑。

本模块只负责"棋局"本身：棋盘状态、落子、K 连判定、和棋判定、
候选点生成与棋盘绘制。搜索与评估属于 gomoku_ai.py，界面属于
main_cli.py / gui.py。三者互不依赖，便于单独测试。

规则（与作业讲义"一、游戏规则"一致）：
  * N x N 方格棋盘，N 由裁判/配置指定（默认 15）。
  * 黑棋（编号 1）先行，双方轮流在任意空位落子，落子后不可移动。
  * 横、竖、两条斜线任一方向上连成 K 子及以上者获胜（长连同样判胜，无禁手）。
  * 棋盘下满仍无人连成 K 子，判为和棋。

统一对战协议 (Gomoku AI Protocol v1) 的常量与工具函数也放在这里，
使游戏引擎、AI 与裁判工具对"棋盘表示"只有一份定义。
"""

from __future__ import annotations

import random
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

# ------------------------------------------------------------------ 常量

EMPTY = 0
BLACK = 1
WHITE = 2

STONE_CHAR = {EMPTY: ".", BLACK: "X", WHITE: "O"}

#: 四个扫描方向：横、竖、主对角线、副对角线。
#: 只取每个方向的一半，配合双向延伸即可覆盖整条线，避免重复计数。
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))

DEFAULT_BOARD_SIZE = 15
DEFAULT_WIN_LENGTH = 5
MIN_BOARD_SIZE = 3

Move = Tuple[int, int]


def opponent(player: int) -> int:
    """返回对手编号（1 <-> 2）。"""
    return 3 - player


# ------------------------------------------------------------------ 棋盘


class Board:
    """固定尺寸的棋盘，内部用一维列表存储，坐标 (row, col) 从 0 开始。"""

    __slots__ = ("size", "cells", "occupied")

    def __init__(self, size: int, cells: Optional[Sequence[int]] = None) -> None:
        if size < MIN_BOARD_SIZE:
            raise ValueError(f"board size must be >= {MIN_BOARD_SIZE}, got {size}")
        self.size = size
        self.cells: List[int] = list(cells) if cells is not None else [EMPTY] * (size * size)
        if len(self.cells) != size * size:
            raise ValueError("cells length does not match board size")
        self.occupied = sum(1 for v in self.cells if v != EMPTY)

    # -------------------------------------------------- 基本访问
    def __len__(self) -> int:
        return self.size

    def get(self, row: int, col: int) -> int:
        return self.cells[row * self.size + col]

    def set(self, row: int, col: int, value: int) -> None:
        idx = row * self.size + col
        if self.cells[idx] != EMPTY and value != EMPTY:
            raise ValueError(f"cell ({row}, {col}) is already occupied")
        if self.cells[idx] == EMPTY and value != EMPTY:
            self.occupied += 1
        elif self.cells[idx] != EMPTY and value == EMPTY:
            self.occupied -= 1
        self.cells[idx] = value

    def is_empty(self, row: int, col: int) -> bool:
        return self.cells[row * self.size + col] == EMPTY

    def in_bounds(self, row: int, col: int) -> bool:
        n = self.size
        return 0 <= row < n and 0 <= col < n

    def is_full(self) -> bool:
        return self.occupied >= self.size * self.size

    def empty_cells(self) -> Iterator[Move]:
        n, cells = self.size, self.cells
        for idx, value in enumerate(cells):
            if value == EMPTY:
                yield divmod(idx, n)

    def neighbors(self, row: int, col: int, radius: int) -> Iterator[Move]:
        """(row, col) 周围切比雪夫距离 <= radius 的格点（含越界过滤）。"""
        n = self.size
        for dr in range(-radius, radius + 1):
            rr = row + dr
            if rr < 0 or rr >= n:
                continue
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                cc = col + dc
                if 0 <= cc < n:
                    yield rr, cc

    def candidate_moves(self, radius: int = 2) -> List[Move]:
        """所有"已有棋子附近 radius 格"内的空位；空棋盘时返回中心点。

        这是作业要求的"合理的候选着法生成"：只考虑已有棋子附近的空位，
        而不是全部 N^2 个空位，可把分支因子从 ~225 降到 ~30。
        """
        if self.occupied == 0:
            c = self.size // 2
            return [(c, c)]
        marked = bytearray(self.size * self.size)
        n = self.size
        for idx, value in enumerate(self.cells):
            if value == EMPTY:
                continue
            r, c = divmod(idx, n)
            for nr, nc in self.neighbors(r, c, radius):
                marked[nr * n + nc] = 1
        return [(idx // n, idx % n) for idx, flag in enumerate(marked)
                if flag and self.cells[idx] == EMPTY]

    def copy(self) -> "Board":
        clone = Board.__new__(Board)
        clone.size = self.size
        clone.cells = self.cells[:]
        clone.occupied = self.occupied
        return clone

    def to_grid(self) -> List[List[int]]:
        """导出为协议规定的"列表的列表"。"""
        n = self.size
        return [self.cells[i * n:(i + 1) * n] for i in range(n)]

    @classmethod
    def from_grid(cls, grid: Sequence[Sequence[int]]) -> "Board":
        n = len(grid)
        flat: List[int] = []
        for row in grid:
            if len(row) != n:
                raise ValueError("board must be square")
            flat.extend(row)
        return cls(n, flat)

    def place(self, row: int, col: int, player: int) -> None:
        self.set(row, col, player)

    # -------------------------------------------------- 绘制
    def render(self, last_move: Optional[Move] = None,
               highlight: Optional[Move] = None) -> str:
        """ASCII 棋盘，行号/列号均从 0 开始。"""
        n = self.size
        header = "    " + "".join(f"{c:>3}" for c in range(n))
        lines = [header]
        for r in range(n):
            cells = []
            for c in range(n):
                ch = STONE_CHAR[self.get(r, c)]
                if last_move == (r, c):
                    ch = ch.lower()
                if highlight == (r, c):
                    ch = "*"
                cells.append(f"{ch:>3}")
            lines.append(f"{r:>3} " + "".join(cells))
        return "\n".join(lines)


# ------------------------------------------------------------------ 胜负判定


def count_line(board: Board, row: int, col: int, dr: int, dc: int, player: int) -> int:
    """沿 (dr, dc) 双向延伸，统计经过 (row, col) 的连续 player 棋子总数。"""
    n = board.size
    total = 1
    for sign in (1, -1):
        rr, cc = row + sign * dr, col + sign * dc
        while 0 <= rr < n and 0 <= cc < n and board.get(rr, cc) == player:
            total += 1
            rr += sign * dr
            cc += sign * dc
    return total


def check_win(board: Board, row: int, col: int, player: int, k: int) -> bool:
    """落子 (row, col) 后，判断 player 是否连成 K 子及以上（含长连）。"""
    for dr, dc in DIRECTIONS:
        if count_line(board, row, col, dr, dc, player) >= k:
            return True
    return False


def find_winning_move(board: Board, player: int, k: int,
                      candidates: Optional[Iterable[Move]] = None) -> Optional[Move]:
    """若 player 有一手立即成 K 连的点，返回该点，否则 None。"""
    moves = candidates if candidates is not None else board.empty_cells()
    for r, c in moves:
        if not board.is_empty(r, c):
            continue
        board.set(r, c, player)
        won = check_win(board, r, c, player, k)
        board.set(r, c, EMPTY)
        if won:
            return (r, c)
    return None


# ------------------------------------------------------------------ 棋局


class GomokuGame:
    """一局棋的完整状态机，供 CLI / GUI / 测试共用。

    用法::

        game = GomokuGame(board_size=15, win_length=5)
        game.play((7, 7))              # 当前玩家落子
        game.winner                    # 0 / BLACK / WHITE
        game.undo()                    # 悔棋
    """

    def __init__(self, board_size: int = DEFAULT_BOARD_SIZE,
                 win_length: int = DEFAULT_WIN_LENGTH,
                 first_player: int = BLACK,
                 board: Optional[Board] = None) -> None:
        validate_config(board_size, win_length)
        self.board_size = board_size
        self.win_length = win_length
        self.board = board if board is not None else Board(board_size)
        self.first_player = first_player
        self.current_player = first_player
        self.winner = EMPTY
        self.win_reason = ""
        self.history: List[Move] = []
        self._winner_checked = False

    # -------------------------------------------------- 查询
    @property
    def is_over(self) -> bool:
        return self.winner != EMPTY or self.board.is_full()

    @property
    def move_count(self) -> int:
        return len(self.history)

    @property
    def last_move(self) -> Optional[Move]:
        return self.history[-1] if self.history else None

    def last_move_of(self, player: int) -> Optional[Move]:
        """player 的最后一手。黑先手且尚无人落子时返回 None。"""
        for idx in range(len(self.history) - 1, -1, -1):
            if self.player_of_move(idx) == player:
                return self.history[idx]
        return None

    def player_of_move(self, index: int) -> int:
        """第 index 手（0 起）是哪一方的。"""
        return self.first_player if index % 2 == 0 else opponent(self.first_player)

    # -------------------------------------------------- 落子 / 撤销
    def is_legal(self, move) -> bool:
        if not isinstance(move, (tuple, list)) or len(move) != 2:
            return False
        r, c = move
        if isinstance(r, bool) or isinstance(c, bool):
            return False
        if not isinstance(r, int) or not isinstance(c, int):
            return False
        return self.board.in_bounds(r, c) and self.board.is_empty(r, c)

    def play(self, move: Move, player: Optional[int] = None) -> None:
        """落子。player 省略时使用 current_player。"""
        if self.is_over:
            raise ValueError("game is already over")
        if not self.is_legal(move):
            raise ValueError(f"illegal move: {move!r}")
        actor = self.current_player if player is None else player
        r, c = move
        self.board.set(r, c, actor)
        self.history.append((r, c))
        self._winner_checked = False

        if check_win(self.board, r, c, actor, self.win_length):
            self.winner = actor
            self.win_reason = f"{self.win_length}-in-a-row"
        elif self.board.is_full():
            self.winner = EMPTY
            self.win_reason = "board full (draw)"
        else:
            self.current_player = opponent(actor)

    def undo(self, steps: int = 1) -> bool:
        """悔棋 steps 手。成功返回 True。"""
        undone = 0
        while steps > 0 and self.history:
            r, c = self.history.pop()
            player = self.player_of_move(len(self.history))
            self.board.set(r, c, EMPTY)
            self.current_player = player
            self.winner = EMPTY
            self.win_reason = ""
            steps -= 1
            undone += 1
        return undone > 0

    def reset(self, first_player: Optional[int] = None) -> None:
        if first_player is not None:
            self.first_player = first_player
        self.board = Board(self.board_size)
        self.current_player = self.first_player
        self.winner = EMPTY
        self.win_reason = ""
        self.history.clear()

    # -------------------------------------------------- 结果描述
    def result_text(self, names: Optional[dict] = None) -> str:
        def label(player: int) -> str:
            if player == EMPTY:
                return "Draw"
            base = "Black" if player == BLACK else "White"
            if names and player in names:
                return f"{names[player]} ({base})"
            return base

        if self.winner != EMPTY:
            return f"{label(self.winner)} wins by {self.win_reason} in {self.move_count} moves"
        if self.board.is_full():
            return f"Draw (board full) in {self.move_count} moves"
        return (f"Game in progress: {self.move_count} moves, "
                f"{label(self.current_player)} to move")

    def render(self) -> str:
        return self.board.render(last_move=self.last_move)


# ------------------------------------------------------------------ 工具


def validate_config(board_size: int, win_length: int) -> None:
    """校验 (N, K) 配置，非法时抛 ValueError。"""
    if not isinstance(board_size, int) or isinstance(board_size, bool):
        raise ValueError("board_size must be an int")
    if not isinstance(win_length, int) or isinstance(win_length, bool):
        raise ValueError("win_length must be an int")
    if board_size < MIN_BOARD_SIZE:
        raise ValueError(f"board_size must be >= {MIN_BOARD_SIZE}, got {board_size}")
    if not (3 <= win_length <= board_size):
        raise ValueError(f"must satisfy 3 <= win_length <= board_size, "
                         f"got win_length={win_length}, board_size={board_size}")


def config_from_win_length(win_length: int, board_size: int) -> Tuple[int, int]:
    """辅助：把 (N, K) 规整为合法组合（K 超界时裁剪到 N）。"""
    win_length = max(3, min(win_length, board_size))
    return board_size, win_length


def random_empty_move(board: Board, rng: Optional[random.Random] = None) -> Move:
    """随机挑一个空位（随机 AI / 开局扰动用）。"""
    rng = rng or random
    empties = list(board.empty_cells())
    if not empties:
        raise ValueError("no empty cell left")
    return rng.choice(empties)


__all__ = [
    "EMPTY", "BLACK", "WHITE", "STONE_CHAR", "DIRECTIONS",
    "DEFAULT_BOARD_SIZE", "DEFAULT_WIN_LENGTH", "MIN_BOARD_SIZE",
    "Board", "GomokuGame", "Move",
    "opponent", "check_win", "count_line", "find_winning_move",
    "validate_config", "config_from_win_length", "random_empty_move",
]
