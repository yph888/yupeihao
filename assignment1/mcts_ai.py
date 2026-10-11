# -*- coding: utf-8 -*-
"""mcts_ai.py -- BME1322 Assignment 1 进阶方向：蒙特卡罗树搜索 (MCTS/UCT)。

本文件实现作业"四、AI 算法要求 → 进阶方向"中的 **MCTS 及 UCT**，
遵循同一套统一对战协议 v1，因此可以直接用 ``arena.py`` 与
``gomoku_ai.py``（Alpha-Beta 版本）做对比实验。

实现要点
--------
* **选择 (Selection)**：从根节点沿 UCT 最大的子节点下行::

      UCT(child) = Q(child)/N(child) + c * sqrt( ln N(parent) / N(child) )

  未访问过的子节点优先展开（相当于 UCT 的 +inf 首访）。
* **扩展 (Expansion)**：每次只展开一个未访问子节点，节省内存。
* **模拟 (Simulation)**：默认用"启发式快速走子"(heuristic playout)：
  以较高概率选择能立即取胜/封堵对手的点，否则在已有棋子邻域内随机落子。
  纯随机走子在五子棋上信噪比极低，启发式走子明显更有效（见报告实验）。
* **回传 (Backpropagation)**：把胜负结果沿路径回传，节点分别累计
  收益与访问次数。

依赖：仅 Python 标准库。
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

NAME = "MCTS-UCT-v1"

EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))

Move = Tuple[int, int]


def _opponent(player: int) -> int:
    return 3 - player


# ---------------------------------------------------------------- 快速走子


class _FastBoard:
    """走子模拟用的轻量棋盘：整数位棋盘 + 常数级胜负判定。"""

    __slots__ = ("size", "k", "cells", "empty_count", "winner")

    def __init__(self, size: int, k: int, cells: Sequence[int]) -> None:
        self.size = size
        self.k = k
        self.cells = list(cells)
        self.empty_count = self.cells.count(EMPTY)
        self.winner = EMPTY

    def place(self, cell: int, player: int) -> None:
        self.cells[cell] = player
        self.empty_count -= 1

    def undo(self, cell: int) -> None:
        self.cells[cell] = EMPTY
        self.empty_count += 1

    def is_win(self, cell: int, player: int) -> bool:
        """落子在 cell 后是否连成 K 子。"""
        n, k = self.size, self.k
        r, c = divmod(cell, n)
        cells = self.cells
        for dr, dc in DIRECTIONS:
            count = 1
            for sign in (1, -1):
                rr, cc = r + sign * dr, c + sign * dc
                while 0 <= rr < n and 0 <= cc < n and cells[rr * n + cc] == player:
                    count += 1
                    rr += sign * dr
                    cc += sign * dc
            if count >= k:
                return True
        return False

    def local_candidates(self, radius: int = 2) -> List[int]:
        """已有棋子邻域内的空位（走子阶段使用，保持分支合理）。"""
        n = self.size
        cells = self.cells
        marked = bytearray(n * n)
        found = False
        for idx, value in enumerate(cells):
            if value == EMPTY:
                continue
            found = True
            r, c = divmod(idx, n)
            for rr in range(max(0, r - radius), min(n - 1, r + radius) + 1):
                base = rr * n
                for cc in range(max(0, c - radius), min(n - 1, c + radius) + 1):
                    marked[base + cc] = 1
        if not found:
            center = (n // 2) * n + n // 2
            return [center] if cells[center] == EMPTY else []
        return [i for i, flag in enumerate(marked) if flag and cells[i] == EMPTY]


def _quick_win_spot(board: _FastBoard, candidates: Sequence[int],
                    player: int) -> Optional[int]:
    """candidates 中能让 player 立即成 K 连的点。"""
    for cell in candidates:
        board.place(cell, player)
        won = board.is_win(cell, player)
        board.undo(cell)
        if won:
            return cell
    return None


def heuristic_playout(board: _FastBoard, player: int, rng: random.Random,
                      max_moves: int) -> int:
    """启发式快速走子，返回胜者（0 表示和棋）。"""
    current = player
    moves = 0
    while moves < max_moves:
        if board.empty_count == 0:
            return EMPTY
        candidates = board.local_candidates(2)
        if not candidates:
            return EMPTY
        # 1) 自己能赢就直接赢
        win = _quick_win_spot(board, candidates, current)
        if win is not None:
            return current
        # 2) 对手下一手能赢就封堵
        block = _quick_win_spot(board, candidates, _opponent(current))
        if block is not None:
            choice = block
        else:
            choice = rng.choice(candidates)
        board.place(choice, current)
        if board.is_win(choice, current):
            return current
        current = _opponent(current)
        moves += 1
    return EMPTY


# ---------------------------------------------------------------- MCTS 节点


class _Node:
    """UCT 搜索树节点。"""

    __slots__ = ("cell", "parent", "player", "untried", "children",
                 "visits", "wins", "expanded")

    def __init__(self, cell: int, parent: Optional["_Node"], player: int,
                 untried: List[int]) -> None:
        self.cell = cell                # 到达本节点所落的一手（根为 -1）
        self.parent = parent
        self.player = player            # 本节点状态下"轮到谁走"
        self.untried = untried          # 尚未展开的着法
        self.children: List["_Node"] = []
        self.visits = 0
        self.wins = 0.0                 # 以 self.parent 的行棋方视角累计
        self.expanded = False

    def is_fully_expanded(self) -> bool:
        return not self.untried

    def is_terminal(self) -> bool:
        return self.expanded and not self.children

    def uct_select(self, c: float, rng: random.Random) -> "_Node":
        """按 UCT 选子节点；并列时随机打破，避免固定偏好。"""
        log_n = math.log(self.visits)
        best, best_value = None, -1.0
        for child in self.children:
            if child.visits == 0:
                return child
            exploit = child.wins / child.visits
            explore = c * math.sqrt(log_n / child.visits)
            value = exploit + explore + rng.random() * 1e-9
            if value > best_value:
                best, best_value = child, value
        return best if best is not None else self.children[0]

    def best_child(self) -> "_Node":
        """最终决策：访问次数最多的子节点（比最大均值更稳健）。"""
        return max(self.children, key=lambda node: (node.visits, node.wins))


# ---------------------------------------------------------------- 主类


class GomokuAI:
    """统一对战协议 v1 的 MCTS/UCT 实现（进阶方向，用于对比实验）。"""

    #: UCT 探索常数
    EXPLORATION = 1.05
    #: 候选着法半径
    RADIUS = 2
    #: 走子阶段的最大手数（防止长局拖慢模拟）
    MAX_PLAYOUT_MOVES = 120

    def __init__(self, player_id: int, board_size: int, win_length: int,
                 exploration: Optional[float] = None,
                 seed: Optional[int] = None,
                 playout: str = "heuristic") -> None:
        if player_id not in (BLACK, WHITE):
            raise ValueError("player_id must be 1 (black) or 2 (white)")
        if not (3 <= win_length <= board_size):
            raise ValueError("require 3 <= win_length <= board_size")
        self.player_id = int(player_id)
        self.board_size = int(board_size)
        self.win_length = int(win_length)
        if exploration is not None:
            self.EXPLORATION = float(exploration)
        self.playout = playout
        self._rng = random.Random(seed if seed is not None else 0xA11CE ^ player_id)
        self.stats: Dict[str, float] = {"calls": 0.0, "playouts": 0.0, "elapsed": 0.0}

    # -------------------------------------------------- 协议入口
    def get_move(self, board, last_opponent_move, time_limit):
        started = time.perf_counter()
        self.stats["calls"] += 1
        try:
            limit = float(time_limit)
        except (TypeError, ValueError):
            limit = 5.0
        if not math.isfinite(limit) or limit <= 0:
            limit = 5.0

        n = self.board_size
        try:
            cells = [value if value in (EMPTY, BLACK, WHITE) else EMPTY
                     for row in board for value in row]
        except TypeError:
            cells = [EMPTY] * (n * n)
        if len(cells) != n * n:
            cells = (cells + [EMPTY] * (n * n))[:n * n]
        empties = [i for i, v in enumerate(cells) if v == EMPTY]
        if not empties:
            return (0, 0)
        if len(empties) == n * n:
            center = n // 2
            return (center, center)

        try:
            move = self._search(cells, empties, limit)
        except Exception:
            move = self._greedy_fallback(cells, empties)
        self.stats["elapsed"] += time.perf_counter() - started
        return move

    # -------------------------------------------------- 搜索
    def _search(self, cells: List[int], empties: List[int], limit: float) -> Move:
        n = self.board_size
        board = _FastBoard(n, self.win_length, cells)
        deadline = time.perf_counter() + limit * 0.72

        candidates = board.local_candidates(self.RADIUS)
        # 安全网：能赢先赢、对手能赢则堵（避免 MCTS 因采样噪声漏杀）
        win = _quick_win_spot(board, candidates, self.player_id)
        if win is not None:
            return divmod(win, n)
        block = _quick_win_spot(board, candidates, _opponent(self.player_id))
        if block is not None:
            return divmod(block, n)
        if not candidates:
            return divmod(empties[0], n)

        root = _Node(-1, None, self.player_id, candidates[:])
        playouts = 0
        while True:
            if (playouts & 15) == 0 and time.perf_counter() >= deadline:
                break
            self._iterate(root, board, cells)
            playouts += 1
        self.stats["playouts"] += playouts

        if not root.children:
            best = max(candidates,
                       key=lambda c: self._static_priority(board, c))
            return divmod(best, n)
        return divmod(root.best_child().cell, n)

    def _iterate(self, root: _Node, board: _FastBoard,
                 cells: Sequence[int]) -> None:
        """一次 MCTS 迭代：选择 -> 扩展 -> 模拟 -> 回传。"""
        rng = self._rng
        node = root
        path: List[_Node] = []

        # ---- 选择 ----
        while node.is_fully_expanded() and node.children:
            node = node.uct_select(self.EXPLORATION, rng)
            board.place(node.cell, node.player)
            path.append(node)

        # ---- 扩展 ----
        if node.untried:
            idx = rng.randrange(len(node.untried))
            cell = node.untried.pop(idx)
            board.place(cell, node.player)
            child = _Node(cell, node, _opponent(node.player),
                          self._child_candidates(board, cell))
            node.children.append(child)
            path.append(child)
            node = child

        # ---- 模拟 ----
        if board.is_win(node.cell, _opponent(node.player)):
            winner = _opponent(node.player)      # 刚落下的一手已经赢了
        else:
            winner = heuristic_playout(board, node.player, rng,
                                       self.MAX_PLAYOUT_MOVES)
        node.expanded = True

        # ---- 回传 ----
        # current.parent.player 是落下 current.cell 的一方，因此该手的结果
        # 记在 current.wins 上（以"下这一手的一方"为视角）。
        current: Optional[_Node] = node
        while current is not None:
            current.visits += 1
            if current.parent is not None:
                actor = current.parent.player
                current.wins += 0.5 if winner == EMPTY else (
                    1.0 if winner == actor else 0.0)
            current = current.parent

        # ---- 撤销模拟期间落下的子，恢复棋盘 ----
        for played in reversed(path):
            board.undo(played.cell)

    @staticmethod
    def _child_candidates(board: _FastBoard, last_cell: int) -> List[int]:
        """子节点未展开着法：以最新落子为中心的邻域空位，控制分支。"""
        size = board.size
        r, c = divmod(last_cell, size)
        radius = 2
        cells = board.cells
        out = []
        for rr in range(max(0, r - radius), min(size - 1, r + radius) + 1):
            base = rr * size
            for cc in range(max(0, c - radius), min(size - 1, c + radius) + 1):
                idx = base + cc
                if cells[idx] == EMPTY:
                    out.append(idx)
        return out

    def _static_priority(self, board: _FastBoard, cell: int) -> int:
        """静态启发值：该点四个方向上相邻同色子的数量（用于极端兜底）。"""
        n = board.size
        r, c = divmod(cell, n)
        cells = board.cells
        score = 0
        for dr, dc in DIRECTIONS:
            for sign in (1, -1):
                rr, cc = r + sign * dr, c + sign * dc
                step = 0
                while 0 <= rr < n and 0 <= cc < n and cells[rr * n + cc] != EMPTY \
                        and step < self.win_length:
                    score += 2 if cells[rr * n + cc] == self.player_id else 1
                    rr += sign * dr
                    cc += sign * dc
                    step += 1
        return score

    def _greedy_fallback(self, cells: Sequence[int], empties: Sequence[int]) -> Move:
        """兜底：先赢、再堵、再靠近中心。"""
        n, k = self.board_size, self.win_length
        board = _FastBoard(n, k, cells)
        for target in (self.player_id, _opponent(self.player_id)):
            for cell in empties:
                board.place(cell, target)
                won = board.is_win(cell, target)
                board.undo(cell)
                if won:
                    return divmod(cell, n)
        center = (n - 1) / 2.0
        best = min(empties, key=lambda c: abs(c // n - center) + abs(c % n - center))
        return divmod(best, n)


__all__ = ["GomokuAI", "NAME"]
