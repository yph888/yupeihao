# -*- coding: utf-8 -*-
"""game_cli.py -- 命令行对局逻辑与玩家/AI 装载。

被 ``main_cli.py`` 与 ``main.py`` 共同使用，集中处理：

* 把命令行里的"玩家描述符"解析成参与者对象
  （``human`` / ``random`` / ``mcts`` / 任意 ``path/to/gomoku_ai.py``）；
* 三种模式（人人 / 人机 / 机机）的对局主循环；
* 终端棋盘的绘制与人类走子的输入解析。

依赖：仅 Python 标准库（AI 文件按路径动态加载）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import time
from typing import List, Optional, Tuple

from gomoku_game import (BLACK, EMPTY, STONE_CHAR, WHITE, Board, GomokuGame,
                         Move, opponent)

#: 内置参与者别名 -> (相对本目录的模块文件名, 备注)
#: 同一个文件支持多种写法（`random` / `random_ai` / `random_ai.py`），
#: 避免"以为指定了对手、其实落回了默认值"这类困惑。
BUILTIN_PLAYERS = {
    "random": ("random_ai.py", "随机合法落子（课程参考实现，弱基线）"),
    "random_ai": ("random_ai.py", "随机合法落子（课程参考实现，弱基线）"),
    "random_ai.py": ("random_ai.py", "随机合法落子（课程参考实现，弱基线）"),
    "mcts": ("mcts_ai.py", "MCTS/UCT 蒙特卡罗树搜索（进阶方向）"),
    "mcts_ai": ("mcts_ai.py", "MCTS/UCT 蒙特卡罗树搜索（进阶方向）"),
    "mcts_ai.py": ("mcts_ai.py", "MCTS/UCT 蒙特卡罗树搜索（进阶方向）"),
    "self": ("gomoku_ai.py", "本作业的 Alpha-Beta AI（默认）"),
    "main": ("gomoku_ai.py", "本作业的 Alpha-Beta AI（默认）"),
    "gomoku_ai": ("gomoku_ai.py", "本作业的 Alpha-Beta AI（默认）"),
    "gomoku_ai.py": ("gomoku_ai.py", "本作业的 Alpha-Beta AI（默认）"),
}


def list_builtin_players(here: str) -> List[Tuple[str, str, str]]:
    """列出可用的内置对手：[(别名, 文件, 说明), ...]（去重）。"""
    seen: set = set()
    out: List[Tuple[str, str, str]] = []
    for alias, (filename, note) in sorted(BUILTIN_PLAYERS.items()):
        if filename in seen:
            continue
        seen.add(filename)
        out.append((alias, filename, note))
    return out


# ---------------------------------------------------------------- 装载


def load_ai_class(path: str):
    """按文件路径加载一个符合统一协议的 GomokuAI 类与 NAME。"""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"AI 文件不存在: {path}")
    module_name = f"_gomoku_player_{abs(hash(os.path.abspath(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 AI 文件: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "GomokuAI"):
        raise AttributeError(f"{path} 中未找到 GomokuAI 类")
    name = getattr(module, "NAME", None) or os.path.splitext(os.path.basename(path))[0]
    return module.GomokuAI, str(name)


def resolve_player(spec: str, here: str) -> Tuple[str, str]:
    """把玩家描述符解析为 (文件路径, 显示名)。

    按以下顺序尝试，任一成功即返回：

    1. 内置别名（``random`` / ``mcts`` / ``self``，大小写不敏感，
       也接受 ``random_ai`` / ``random_ai.py`` 这类写法）；
    2. 相对于 ``here`` 的路径，或绝对路径；
    3. 相对当前工作目录的路径。

    全都失败时抛 ``FileNotFoundError``，并在消息里列出所有内置别名——
    这样"名字写错"会立刻报错，而不是静默退化成别的对手。
    """
    key = spec.strip().lower()
    if key in BUILTIN_PLAYERS:
        filename, _ = BUILTIN_PLAYERS[key]
        path = os.path.join(here, filename)
        load_ai_class(path)                 # 早失败：路径错误立刻报出
        return path, key

    for path in (os.path.join(here, spec), os.path.abspath(spec)):
        if os.path.isfile(path):
            _, name = load_ai_class(path)
            return path, name

    aliases = ", ".join(sorted({a for a, _, _ in list_builtin_players(here)}))
    raise FileNotFoundError(
        f"找不到 AI 文件: {spec!r}（完整路径: {os.path.join(here, spec)}）。"
        f"可用内置别名: {aliases}；也可以直接给出 gomoku_ai.py 的路径。")


# ---------------------------------------------------------------- 参与者


class Player:
    """参与者基类：人类与 AI 共用同一套调用接口。"""

    kind = "base"

    def __init__(self, player_id: int, name: str) -> None:
        self.player_id = player_id
        self.name = name
        self.stats = {"moves": 0.0, "elapsed": 0.0, "max_elapsed": 0.0}

    def choose_move(self, game: GomokuGame, time_limit: float) -> Move:
        raise NotImplementedError

    def note_elapsed(self, elapsed: float) -> None:
        self.stats["moves"] += 1
        self.stats["elapsed"] += elapsed
        self.stats["max_elapsed"] = max(self.stats["max_elapsed"], elapsed)


class HumanPlayer(Player):
    """终端人类玩家：输入 "行 列"（也接受 "行,列"、"r c"、h8 形式）。"""

    kind = "human"

    def choose_move(self, game: GomokuGame, time_limit: float) -> Move:
        n = game.board_size
        while True:
            try:
                raw = input(f"[{self.name}] 轮到你了，请输入落子 (行 列，0-{n - 1})"
                            f"，u=悔棋 q=退出 > ").strip()
            except (EOFError, KeyboardInterrupt):
                raise SystemExit("\n已退出对局。")
            if not raw:
                continue
            low = raw.lower()
            if low in ("q", "quit", "exit"):
                raise SystemExit("已退出对局。")
            if low in ("u", "undo"):
                return (-1, -1)              # 由主循环解释为悔棋请求
            if low in ("h", "help", "?"):
                print("  输入示例： 7 7 / 7,7 / h8（列字母 a..，行 1..）")
                continue
            move = parse_move(raw, n)
            if move is None:
                print("  ✗ 无法解析，请重新输入（例如 7 7）。")
                continue
            if not game.is_legal(move):
                print(f"  ✗ ({move[0]}, {move[1]}) 不是合法空位，请重试。")
                continue
            return move


class AIPlayer(Player):
    """AI 参与者：把游戏状态翻译成统一协议的调用参数。"""

    kind = "ai"

    def __init__(self, ai_class, player_id: int, name: str,
                 board_size: int, win_length: int) -> None:
        super().__init__(player_id, name)
        self.ai = ai_class(player_id, board_size, win_length)
        self.time_limit = 5.0

    def choose_move(self, game: GomokuGame, time_limit: float) -> Move:
        board = game.board.to_grid()
        last = game.last_move_of(opponent(self.player_id))
        started = time.perf_counter()
        move = self.ai.get_move(board, last, time_limit)
        self.note_elapsed(time.perf_counter() - started)
        return validate_ai_move(move, game, self.name)

    def sync_time_limit(self, time_limit: float) -> None:
        self.time_limit = time_limit


def validate_ai_move(move, game: GomokuGame, name: str) -> Move:
    """校验 AI 返回值；非法则抛错，便于在自建引擎中及早发现问题。"""
    if not isinstance(move, (tuple, list)) or len(move) != 2:
        raise ValueError(f"{name} 返回了非法格式: {move!r}")
    r, c = move
    if isinstance(r, bool) or isinstance(c, bool) \
            or not isinstance(r, int) or not isinstance(c, int):
        raise ValueError(f"{name} 返回的坐标不是 int: {move!r}")
    if not game.is_legal((r, c)):
        raise ValueError(f"{name} 返回了非法落子: {(r, c)}")
    return (r, c)


def parse_move(text: str, n: int) -> Optional[Move]:
    """解析人类输入。支持 "7 7" / "7,7" / "h8" / "8h"。"""
    cleaned = text.replace(",", " ").replace("，", " ").strip()
    parts = cleaned.split()
    if len(parts) == 2:
        try:
            r, c = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        if 0 <= r < n and 0 <= c < n:
            return (r, c)
        return None
    if len(cleaned) >= 2 and cleaned[0].isalpha():
        col = ord(cleaned[0].lower()) - ord("a")
        rest = cleaned[1:]
        if rest.isdigit() and 0 <= col < n:
            row = int(rest) - 1
            if 0 <= row < n:
                return (row, col)
    if len(cleaned) >= 2 and cleaned[-1].isalpha():
        row = int(cleaned[:-1]) - 1 if cleaned[:-1].isdigit() else -1
        col = ord(cleaned[-1].lower()) - ord("a")
        if 0 <= row < n and 0 <= col < n:
            return (row, col)
    return None


# ---------------------------------------------------------------- 对局主循环


def render_board(game: GomokuGame) -> str:
    """带列号、最新一手小写标记的 ASCII 棋盘。"""
    return game.board.render(last_move=game.last_move)


def play_game(black_player: Player, white_player: Player,
              board_size: int, win_length: int,
              time_limit: float = 5.0, verbose: bool = True,
              max_moves: Optional[int] = None) -> GomokuGame:
    """按统一规则跑完一整局，返回结束时的 GomokuGame。"""
    game = GomokuGame(board_size, win_length)
    players = {BLACK: black_player, WHITE: white_player}

    if verbose:
        print(f"\n对局开始：棋盘 {board_size}x{board_size}，连珠 {win_length}，"
              f"每步时限 {time_limit:.2f}s")
        print(f"  黑棋(X, 先手) = {black_player.name}")
        print(f"  白棋(O, 后手) = {white_player.name}")
        print(render_board(game))

    while not game.is_over:
        if max_moves is not None and game.move_count >= max_moves:
            break
        player = players[game.current_player]
        move = player.choose_move(game, time_limit)

        if move == (-1, -1):                  # 人类请求悔棋
            if not game.undo(2 if game.move_count >= 2 else 1):
                print("  没有可悔的棋。")
            if verbose:
                print(render_board(game))
            continue

        game.play(move)
        if verbose:
            actor = STONE_CHAR[3 - game.current_player] \
                if not game.is_over else STONE_CHAR[game.winner or EMPTY]
            print(f"\n第 {game.move_count:>3} 手  {player.name} 落子 {move}")
            print(render_board(game))

    if verbose:
        print("\n" + "=" * 46)
        print(game.result_text({BLACK: black_player.name, WHITE: white_player.name}))
        print("=" * 46)
    return game


def print_stats(players: List[Player]) -> None:
    """打印每方的平均/最长思考时间，便于观察时限遵守情况。"""
    for player in players:
        moves = player.stats["moves"]
        if moves <= 0:
            continue
        avg = player.stats["elapsed"] / moves
        print(f"  {player.name:<22} 出手 {int(moves):>3} 次，"
              f"平均 {avg:.3f}s，最长 {player.stats['max_elapsed']:.3f}s")


__all__ = [
    "Player", "HumanPlayer", "AIPlayer",
    "BUILTIN_PLAYERS", "load_ai_class", "resolve_player",
    "list_builtin_players",
    "play_game", "render_board", "print_stats", "parse_move",
    "validate_ai_move",
]
