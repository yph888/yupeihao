# -*- coding: utf-8 -*-
"""gomoku_ai.py -- BME1322 Assignment 1: 广义五子棋 AI (统一对战协议 v1)。

本文件实现作业讲义"二、统一对战协议"要求的接口::

    class GomokuAI:
        def __init__(self, player_id: int, board_size: int, win_length: int): ...
        def get_move(self, board, last_opponent_move, time_limit) -> (row, col)

算法主线是课堂讲授的对抗搜索框架：**MiniMax + Alpha-Beta 剪枝**，
配合深度截断、棋型评估函数、候选着法生成与迭代加深。MCTS/UCT 的
对比实现放在同目录的 ``mcts_ai.py``，本文件是单一、稳定的评赛版本。

设计要点
--------
1. **不硬编码 N 与 K**：棋盘尺寸、连珠数、评估权重都由构造参数或
   ``win_length`` 推导，支持 5 <= N <= 15、3 <= K <= N 的任意组合。
2. **线段 (segment) 索引 + 增量评估**：预先把每条长度恰为 K 的"窗口"
   编号。落一子只影响经过该格的至多 8 个窗口，因此胜负判定
   (``mine == K``) 与局面评估都能 O(1) 增量更新，搜索内部从不扫描整盘。
   这是本实现相对"每层重扫棋盘"版本的主要加速来源。
3. **棋型评估**：窗口分值按 (己方子数, 对方子数) 查表；只要窗口里出现
   对手棋子即视为死窗口，从而自然表达"活 K-1 / 冲 K-1 / 被堵死"等课堂
   讨论的棋型。权重以 K 为基准定义，对 K=3/4/5/6 自适应。
4. **时间管理**：迭代加深 + 软/硬双截止时间；只采纳完整跑完且未超软截止
   的那一层结果，并给下一层预留预算，避免顶格用满时限。
5. **兜底安全网**：任何异常都回退到"先赢、再堵、再启发式"的确定性选择，
   绝不返回非法坐标或把异常抛给裁判。

依赖：仅 Python 标准库。
"""

from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

NAME = "Segment-AlphaBeta-v1"

# ---------------------------------------------------------------- 协议常量

EMPTY, BLACK, WHITE = 0, 1, 2
DIRECTIONS: Tuple[Tuple[int, int], ...] = ((0, 1), (1, 0), (1, 1), (1, -1))

Move = Tuple[int, int]

#: 胜利分值，必须远大于任何启发式分值。
WIN_SCORE = 10_000_000
#: 搜索中表示"必胜/必败"的界。
MATE_INF = WIN_SCORE // 2

#: 评估函数中对手棋型的权重：>1 表示防守略优先于进攻。
BLOCK_WEIGHT = 1.15


def _opponent(player: int) -> int:
    return 3 - player


# ---------------------------------------------------------------- 增量模型


class _BoardModel:
    """棋盘内部模型：线段索引 + 双方视角的增量评估。

    数据
    ----
    ``segments[s]``       第 s 个长度恰为 K 的窗口（格点索引元组）
    ``cell_segments[c]``  经过格点 c 的所有窗口编号
    ``mine[s]/theirs[s]`` 双方在第 s 个窗口内的子数
    ``score_mine[s]``     第 s 个窗口对"我方视角"的贡献（两侧对称维护）
    ``score_opp[s]``      第 s 个窗口对"对手视角"的贡献

    维护两个视角的数组使 ``negamax`` 在任意行棋方都能 O(1) 取到
    "以该方为视角"的分值，无需临时求和。
    """

    __slots__ = ("size", "k", "segments", "cell_segments", "cells",
                 "mine", "theirs", "score_mine", "score_opp",
                 "sum_mine", "sum_opp", "table",
                 "_zobrist", "zobrist")

    def __init__(self, size: int, k: int) -> None:
        self.size = size
        self.k = k
        self.cells: List[int] = [EMPTY] * (size * size)
        self.table = self._build_table(k)
        self._zobrist = self._build_zobrist(size)
        self.zobrist = 0

        segments: List[Tuple[int, ...]] = []
        cell_segments: List[List[int]] = [[] for _ in range(size * size)]
        for r in range(size):
            for c in range(size):
                for dr, dc in DIRECTIONS:
                    er, ec = r + (k - 1) * dr, c + (k - 1) * dc
                    if 0 <= er < size and 0 <= ec < size:
                        cells = tuple((r + i * dr) * size + (c + i * dc)
                                      for i in range(k))
                        sid = len(segments)
                        segments.append(cells)
                        for cell in cells:
                            cell_segments[cell].append(sid)
        self.segments = segments
        self.cell_segments = cell_segments
        n_seg = len(segments)
        self.mine = [0] * n_seg
        self.theirs = [0] * n_seg
        self.score_mine = [0] * n_seg
        self.score_opp = [0] * n_seg
        self.sum_mine = 0
        self.sum_opp = 0

    # -------------------------------------------------- 表
    @staticmethod
    def _build_table(k: int) -> Dict[Tuple[int, int], int]:
        """(a, b) -> 窗口分值；a 为视角方子数，b 为对手子数，a + b <= K。"""
        table: Dict[Tuple[int, int], int] = {}
        for a in range(k + 1):
            for b in range(k + 1 - a):
                if a == 0 or b > 0:
                    score = 0                    # 空窗口 / 被对手阻断
                elif a >= k:
                    score = WIN_SCORE            # 已成 K 连
                else:
                    gap = k - a                   # 还差几子成 K
                    score = int(10 ** (k - gap) / (gap * gap))
                    if gap == 1:
                        score = max(score, 2_000)  # 活 K-1 保底 2000
                table[(a, b)] = score
        return table

    @staticmethod
    def _build_zobrist(size: int) -> List[Tuple[int, int]]:
        rng = random.Random(0xC0FFEE)
        return [(rng.getrandbits(61), rng.getrandbits(61))
                for _ in range(size * size)]

    # -------------------------------------------------- 装载
    def reset_from(self, cells: Sequence[int], player: int) -> None:
        """按传入棋盘重建全部增量结构。player 为我方编号。"""
        n_seg = len(self.mine)
        for i in range(n_seg):
            self.mine[i] = 0
            self.theirs[i] = 0
            self.score_mine[i] = 0
            self.score_opp[i] = 0
        self.sum_mine = 0
        self.sum_opp = 0
        self.zobrist = 0
        for i in range(len(self.cells)):
            self.cells[i] = EMPTY
        for cell, value in enumerate(cells):
            if value == EMPTY:
                continue
            self.cells[cell] = value
            self.zobrist ^= self._zobrist[cell][value - 1]
            self._apply_delta(cell, mine_delta=1 if value == player else 0,
                              theirs_delta=0 if value == player else 1)

    def _apply_delta(self, cell: int, mine_delta: int, theirs_delta: int) -> None:
        """在格点 cell 上增加/减少双方的子数，并同步两个视角的分值。"""
        table = self.table
        mine, theirs = self.mine, self.theirs
        sm, so = self.score_mine, self.score_opp
        for sid in self.cell_segments[cell]:
            a, b = mine[sid], theirs[sid]
            old_m = table[(a, b)]
            old_o = table[(b, a)]
            na, nb = a + mine_delta, b + theirs_delta
            new_m = table[(na, nb)]
            new_o = table[(nb, na)]
            sm[sid] += new_m - old_m
            so[sid] += new_o - old_o
            self.sum_mine += new_m - old_m
            self.sum_opp += new_o - old_o
            mine[sid] = na
            theirs[sid] = nb

    # -------------------------------------------------- 落子 / 撤销
    def place(self, cell: int, is_mine: bool) -> bool:
        """落子；返回该手是否形成 K 连。"""
        self.cells[cell] = BLACK if is_mine else WHITE
        self.zobrist ^= self._zobrist[cell][0 if is_mine else 1]
        self._apply_delta(cell, 1 if is_mine else 0, 0 if is_mine else 1)
        arr = self.mine if is_mine else self.theirs
        k = self.k
        for sid in self.cell_segments[cell]:
            if arr[sid] >= k:
                return True
        return False

    def remove(self, cell: int, is_mine: bool) -> None:
        self.cells[cell] = EMPTY
        self.zobrist ^= self._zobrist[cell][0 if is_mine else 1]
        self._apply_delta(cell, -1 if is_mine else 0, 0 if is_mine else -1)

    # -------------------------------------------------- 查询
    def evaluate(self) -> int:
        """以我方为视角的局面分。"""
        return int(self.sum_mine - BLOCK_WEIGHT * self.sum_opp)

    def evaluate_for_opponent(self) -> int:
        """以对手为视角的局面分。"""
        return int(self.sum_opp - BLOCK_WEIGHT * self.sum_mine)

    def move_priority(self, cell: int, weights: Sequence[int]) -> int:
        """着法排序启发值：同时计入推进己方棋型与封堵对方棋型。"""
        mine, theirs = self.mine, self.theirs
        score = 0
        for sid in self.cell_segments[cell]:
            a = mine[sid]
            b = theirs[sid]
            if b == 0:
                score += weights[a]
            if a == 0:
                score += weights[b]
        return score

    def immediate_wins(self, candidates: Sequence[int], is_mine: bool) -> List[int]:
        """候选点中能立刻成 K 连的点（假定该点为空）。"""
        need = self.k - 1
        arr = self.mine if is_mine else self.theirs
        out = []
        for cell in candidates:
            for sid in self.cell_segments[cell]:
                if arr[sid] >= need:
                    out.append(cell)
                    break
        return out

    def has_win(self, is_mine: bool) -> bool:
        arr = self.mine if is_mine else self.theirs
        return max(arr) >= self.k if arr else False


# ---------------------------------------------------------------- 搜索主体


class GomokuAI:
    """统一对战协议 v1 的评赛 AI：迭代加深 + Alpha-Beta 搜索。"""

    #: 迭代加深的深度上限（通常在此之前就因时限停止）
    MAX_DEPTH = 16
    #: 硬截止比例：搜索必须在此刻之前收手，剩余时间留作余量，
    #: 用于吸收"检查间隔内的最后一小段搜索"与 Python GC 抖动。
    HARD_TIME_RATIO = 0.72
    #: 预测下一层耗时时的安全系数（乘在预测值上）
    PROJECT_SAFETY = 1.35
    #: 每隔多少节点检查一次时限（越小越精确，但检查本身有开销）
    TIME_CHECK_INTERVAL = 128
    #: 候选着法上限（按启发值排序后截断），控制有效分支因子
    MAX_CANDIDATES = 14
    #: 候选着法半径：只看已有棋子附近 RADIUS 格内的空位
    RADIUS = 2
    #: 开局偏好（相对中心的行列偏移），按先后手略作区分
    OPENING_BOOK: Dict[int, Tuple[Tuple[int, int], ...]] = {
        1: ((0, 0), (0, 1), (1, 0), (1, 1), (1, -1), (-1, 1), (0, -1),
            (-1, 0), (-1, -1), (0, 2), (2, 0), (2, 2), (-2, -2)),
        2: ((0, 0), (0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1),
            (-1, 1), (-1, -1), (0, 2), (2, 0), (2, 2), (-2, -2)),
    }

    def __init__(self, player_id: int, board_size: int, win_length: int) -> None:
        if player_id not in (BLACK, WHITE):
            raise ValueError("player_id must be 1 (black) or 2 (white)")
        if not (3 <= win_length <= board_size):
            raise ValueError("require 3 <= win_length <= board_size")
        self.player_id = int(player_id)
        self.board_size = int(board_size)
        self.win_length = int(win_length)

        self._model = _BoardModel(self.board_size, self.win_length)
        self._weights = self._build_move_weights(self.win_length)
        self._rng = random.Random(0x5EED ^ (self.player_id * 7919))

        self._tt: Dict[int, Tuple[int, int, int, int]] = {}
        self._deadline_hard = 0.0
        self._nodes = 0
        self._aborted = False
        #: 三角 PV 表：_pv_table[d] 是第 d 层搜索出的主变首着
        self._pv_table: List[List[int]] = []
        self._pv: List[Move] = []

        #: 供实验统计（对局结束后可读取）
        self.stats: Dict[str, float] = {"calls": 0.0, "nodes": 0.0,
                                        "depth": 0.0, "elapsed": 0.0}

    # -------------------------------------------------- 初始化辅助
    @staticmethod
    def _build_move_weights(k: int) -> Tuple[int, ...]:
        """着法排序权重表：index = 窗口内某方已有子数。"""
        weights = [0] * (k + 2)
        for i in range(k + 2):
            if i >= k:
                weights[i] = WIN_SCORE
            elif i == 0:
                weights[i] = 1
            else:
                gap = k - i
                weights[i] = int(10 ** (k - gap) / (gap * gap)) + 2
        return tuple(weights)

    # -------------------------------------------------- 协议入口
    def get_move(self, board, last_opponent_move, time_limit):
        """返回本步落子 (row, col)。必须在 time_limit 秒内给出合法坐标。"""
        started = time.perf_counter()
        self.stats["calls"] += 1
        try:
            limit = float(time_limit)
        except (TypeError, ValueError):
            limit = 5.0
        if not math.isfinite(limit) or limit <= 0:
            limit = 5.0

        n = self.board_size
        cells = self._flatten(board, n)
        empties = [i for i, v in enumerate(cells) if v == EMPTY]
        if not empties:                       # 棋盘已满：返回哨兵坐标由裁判判和
            return (0, 0)
        if len(empties) == n * n:             # 先手第一手：占据中心
            c = n // 2
            return (c, c)

        try:
            move = self._plan(cells, empties, limit)
        except Exception:                     # 绝不让异常泄漏给裁判
            move = self._fallback_move(cells, empties)
        self.stats["elapsed"] += time.perf_counter() - started
        return move

    # -------------------------------------------------- 决策流程
    def _plan(self, cells: List[int], empties: List[int], limit: float) -> Move:
        n = self.board_size
        start = time.perf_counter()
        self._model.reset_from(cells, self.player_id)
        self._deadline_hard = limit * self.HARD_TIME_RATIO
        self._tt = {}
        self._nodes = 0
        self._aborted = False
        self._pv_table = []
        self._pv = []

        candidates = self._candidates()
        if not candidates:
            candidates = empties[:]

        # --- 安全网 1：自己能立刻成 K 连 ---
        wins = self._model.immediate_wins(candidates, True)
        if wins:
            return divmod(wins[0], n)

        # --- 安全网 2：对手下一手能成 K 连，必须封堵 ---
        # 只挡离成 K 最近的那个点：对手若真有"活 K-1"（两条成 K 路线），
        # 局面本已无解，用带威胁权重的启发值在并列封堵点里择优即可。
        threats = self._model.immediate_wins(candidates, False)
        if threats:
            best = max(threats,
                       key=lambda c: self._model.move_priority(c, self._weights))
            return divmod(best, n)
        # --- 开局若干手使用一般性启发，避免在稀疏棋盘上浪费搜索预算 ---
        # 仅当局面"平静"时才走开局库：若任一方已经形成接近成 K 的连线
        # （gap <= 2），说明对手已经先手进攻，此时必须交给搜索/封堵逻辑判断，
        # 否则固定开局着法会无视威胁（例如对手已有两连，却去走中心偏好点）。
        book = self._opening_move(cells, len(empties))
        if book is not None and self._book_is_safe():
            return book

        # --- 迭代加深 Alpha-Beta ---
        # 时限策略（第三代实现，见报告 §4.4/§5.1）：
        #   1) 每层单独计时，用"上一层 / 上上层"的耗时比预测下一层；
        #   2) 只有当预测耗时加上安全系数仍落在硬截止内时，才开新的一层；
        #   3) 一旦开了新层就一定让它跑完（除非撞上硬截止），
        #      半途被砍掉的层不会污染结果，但会白白浪费时间。
        # 这样既不会"顶格"用满时限，也不会像早期版本那样因为过度保守
        # 而放弃本可跑完的更深一层。
        best_cell = max(candidates,
                        key=lambda c: self._model.move_priority(c, self._weights))
        depth_done = 0
        previous_cost = 0.0
        last_cost = 0.0
        for depth in range(1, self.MAX_DEPTH + 1):
            elapsed = time.perf_counter() - start
            if elapsed >= self._deadline_hard:
                break
            if depth > 3 and last_cost > 0.0 and previous_cost > 0.0:
                growth = max(1.5, last_cost / previous_cost)
                projected = last_cost * growth * self.PROJECT_SAFETY
                if elapsed + projected > self._deadline_hard:
                    break                       # 下一层大概率跑不完，主动收手
            layer_start = time.perf_counter()
            ranked = self._search_root(candidates, depth, best_cell, start)
            layer_cost = time.perf_counter() - layer_start
            if self._aborted or not ranked:
                break                           # 该层被硬截止打断 → 弃用
            previous_cost, last_cost = last_cost, layer_cost
            best_cell = ranked[0][1]
            depth_done = depth
            if abs(ranked[0][0]) >= MATE_INF:   # 已见必胜/必败
                break
        self.stats["nodes"] += self._nodes
        self.stats["depth"] = depth_done
        self._pv = self._extract_pv(best_cell)
        return divmod(best_cell, n)

    def _book_is_safe(self) -> bool:
        """局面是否"平静"到可以安全使用开局库。

        判据：双方在任一长度 K 窗口内的最大子数都不超过 **1**。
        只要有任意一方已经连成 2 子（``max >= 2``），局面就已进入接触战，
        开局库必须让位给搜索与封堵逻辑。

        这条检查是必要的：开局库原本只看"手数 ≤ 4"，于是当对手在前几手
        沿一条线连下时（例如黑 (7,3)(7,4)），固定开局着法会完全无视威胁，
        看起来就像"白棋不防守"。阈值取 2 而不是 ``K-2`` 的原因：
        在 K=5 时"两连"看起来还很早，但此刻对手下一手就能形成三连，
        而白棋只有一颗孤子、毫无攻势，任何非封堵的固定着法都是实质亏损。
        """
        if self._model.mine and max(self._model.mine) >= 2:
            return False
        if self._model.theirs and max(self._model.theirs) >= 2:
            return False
        return True

    def _opening_move(self, cells: Sequence[int], empty_count: int) -> Optional[Move]:
        """开局若干手用相对中心的固定偏好（一般性启发，与对手身份无关）。"""
        n = self.board_size
        if n * n - empty_count > 4:           # 只在前 4 手使用
            return None
        center = (n - 1) // 2
        for dr, dc in self.OPENING_BOOK.get(self.player_id, ()):
            r, c = center + dr, center + dc
            if 0 <= r < n and 0 <= c < n and cells[r * n + c] == EMPTY:
                return (r, c)
        return None

    def _search_root(self, candidates: Sequence[int], depth: int,
                     first_choice: int, start: float) -> List[Tuple[int, int]]:
        """根节点：逐一搜索候选着法，返回按分值降序的 (score, cell) 列表。"""
        model = self._model
        order = self._order_moves(candidates, first_choice)
        alpha = -MATE_INF
        results: List[Tuple[int, int]] = []
        self._pv_table = [[] for _ in range(depth + 1)]
        for cell in order:
            # 根节点也检查时限：即使某一层整体未完，也尽快交还给上层决策
            if time.perf_counter() - start >= self._deadline_hard:
                self._aborted = True
                break
            child_pv: List[int] = []
            if model.place(cell, True):
                score = WIN_SCORE
            else:
                score = -self._negamax(depth - 1, -MATE_INF, -alpha,
                                       _opponent(self.player_id), start, child_pv)
            model.remove(cell, True)
            if self._aborted:
                break
            results.append((score, cell))
            if score > alpha:
                alpha = score
                # 刷新主变：本手 + 子树回传的最佳续着
                self._pv_table[0] = [cell]
                if child_pv:
                    self._pv_table[1] = child_pv[:depth]
        results.sort(key=lambda item: -item[0])
        return results

    def _negamax(self, depth: int, alpha: int, beta: int,
                 player: int, start: float,
                 pv_out: Optional[List[int]] = None) -> int:
        """Negamax + Alpha-Beta 剪枝；返回值以 player 为视角。

        ``pv_out`` 非空时，会被填入本子树的最优续着序列（用于显示主变）。
        """
        self._nodes += 1
        if (self._nodes % self.TIME_CHECK_INTERVAL) == 0:
            if time.perf_counter() - start >= self._deadline_hard:
                self._aborted = True
                return 0

        model = self._model
        is_mine = (player == self.player_id)
        if depth <= 0:
            return model.evaluate() if is_mine else model.evaluate_for_opponent()
        if model.has_win(is_mine):
            return WIN_SCORE + depth
        if model.has_win(not is_mine):
            return -WIN_SCORE - depth

        key = model.zobrist * 3 + (0 if is_mine else 1)
        entry = self._tt.get(key)
        tt_move = -1
        if entry is not None:
            e_depth, e_score, e_flag, e_move = entry
            tt_move = e_move
            if e_depth >= depth:
                if e_flag == 0:
                    return e_score
                if e_flag == 1:
                    if e_score > alpha:
                        alpha = e_score
                elif e_score < beta:
                    beta = e_score
                if alpha >= beta:
                    return e_score

        candidates = self._candidates()
        if not candidates:
            return model.evaluate() if is_mine else model.evaluate_for_opponent()

        pv_level = None
        if pv_out is not None:
            pv_level = len(pv_out)
            if pv_level < len(self._pv_table):
                self._pv_table[pv_level] = []

        order = self._order_moves(candidates, tt_move if tt_move >= 0 else None)
        best = -MATE_INF
        best_cell = order[0]
        alpha0 = alpha
        for cell in order:
            child_pv: List[int] = []
            if model.place(cell, is_mine):
                score = WIN_SCORE + depth
            else:
                score = -self._negamax(depth - 1, -beta, -alpha,
                                       _opponent(player), start, child_pv)
            model.remove(cell, is_mine)
            if self._aborted:
                return 0
            if score > best:
                best = score
                best_cell = cell
                if pv_out is not None:
                    pv_out[:] = [cell] + (child_pv[:depth - 1] if child_pv else [])
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break

        if best >= beta:
            flag = 1
        elif best <= alpha0:
            flag = 2
        else:
            flag = 0
        self._tt[key] = (depth, best, flag, best_cell)
        return best

    # -------------------------------------------------- 辅助
    def _flatten(self, board, size: int) -> List[int]:
        """把裁判传入的 2D 列表压平，并对异常输入容错。"""
        cells: List[int] = []
        try:
            for row in board:
                for value in row:
                    cells.append(value if value in (EMPTY, BLACK, WHITE) else EMPTY)
        except TypeError:
            return [EMPTY] * (size * size)
        expected = size * size
        if len(cells) != expected:
            cells = (cells + [EMPTY] * expected)[:expected]
        return cells

    def _candidates(self) -> List[int]:
        """已有棋子附近 RADIUS 格内的空位（邻域候选着法生成）。"""
        n = self.board_size
        cells = self._model.cells
        marked = bytearray(n * n)
        found = False
        for idx, value in enumerate(cells):
            if value == EMPTY:
                continue
            found = True
            r, c = divmod(idx, n)
            r0, r1 = max(0, r - self.RADIUS), min(n - 1, r + self.RADIUS)
            c0, c1 = max(0, c - self.RADIUS), min(n - 1, c + self.RADIUS)
            for rr in range(r0, r1 + 1):
                base = rr * n
                for cc in range(c0, c1 + 1):
                    marked[base + cc] = 1
        if not found:
            return []
        return [i for i, flag in enumerate(marked) if flag and cells[i] == EMPTY]

    def _order_moves(self, candidates: Sequence[int],
                     first: Optional[int] = None) -> List[int]:
        """按启发值降序排列，并把 TT / 上一层最优着法提前（剪枝效率关键）。"""
        weights = self._weights
        priority = self._model.move_priority
        scored = [(priority(cell, weights), cell) for cell in candidates]
        scored.sort(key=lambda item: -item[0])
        order = [cell for _, cell in scored[:self.MAX_CANDIDATES]]
        if first is not None and first in candidates:
            if first in order:
                order.remove(first)
            order.insert(0, first)
        return order

    def _fallback_move(self, cells: Sequence[int], empties: Sequence[int]) -> Move:
        """极端情况下的确定性兜底：先赢、再堵、再中心优先。"""
        n = self.board_size
        k = self.win_length
        for target in (self.player_id, _opponent(self.player_id)):
            for cell in empties:
                if self._would_win(cells, cell, target, k, n):
                    return divmod(cell, n)
        center = (n - 1) / 2.0
        best = min(empties, key=lambda c: abs(c // n - center) + abs(c % n - center))
        return divmod(best, n)

    @staticmethod
    def _would_win(cells: Sequence[int], cell: int, player: int,
                   k: int, n: int) -> bool:
        """把 player 放在 cell 上是否立即成 K 连。"""
        r, c = divmod(cell, n)
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

    def _extract_pv(self, first_cell: int) -> List[Move]:
        """取最后一层迭代搜索到的主变（principal variation）。

        ``_search_root`` 每层会记录"根着法 + 子树最优续着"，这里把它转成
        坐标序列。注意：迭代加深搜索的主变是启发式信息，只用于报告与
        调试展示，不参与决策（决策只看该层的分值）。
        """
        n = self.board_size
        pv: List[Move] = []
        seen: set = set()
        chain = [first_cell]
        if len(self._pv_table) > 1:
            chain.extend(self._pv_table[1])
        for cell in chain:
            if not isinstance(cell, int) or not (0 <= cell < n * n) or cell in seen:
                continue
            seen.add(cell)
            pv.append(divmod(cell, n))
        return pv


__all__ = ["GomokuAI", "NAME", "WIN_SCORE"]
