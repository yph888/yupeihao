# -*- coding: utf-8 -*-
"""main_cli.py -- 广义五子棋命令行入口（人人 / 人机 / 机机 三种模式）。

示例::

    # 人人对战（默认 15x15 五连）
    python main_cli.py --mode pvp

    # 人机对战：人类执黑，AI 执白；9x9 四连，每步 1 秒
    python main_cli.py --mode pve --board 9 --win 4 --time 1.0 --human-color black

    # 机机对战：自己的 Alpha-Beta AI 对阵 MCTS
    python main_cli.py --mode eve --black gomoku_ai.py --white mcts --quiet

    # 机机对战：与同学的 AI 对战（任意路径）
    python main_cli.py --mode eve --black gomoku_ai.py --white ../classmate/gomoku_ai.py
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from game_cli import (AIPlayer, HumanPlayer, list_builtin_players, play_game,
                      print_stats, resolve_player, load_ai_class)
from gomoku_game import (BLACK, DEFAULT_BOARD_SIZE, DEFAULT_WIN_LENGTH,
                         WHITE, validate_config)

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_AI = os.path.join(HERE, "gomoku_ai.py")

COLOR_NAMES = {"black": BLACK, "b": BLACK, "1": BLACK,
               "white": WHITE, "w": WHITE, "2": WHITE}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main_cli.py",
        description="广义五子棋 (m,n,k)-Game — 命令行界面，支持人人/人机/机机三种模式")
    parser.add_argument("--mode", choices=("pvp", "pve", "eve"), default="pve",
                        help="pvp=人人，pve=人机，eve=机机（默认 pve）")
    parser.add_argument("--board", type=int, default=DEFAULT_BOARD_SIZE,
                        help=f"棋盘边长 N（默认 {DEFAULT_BOARD_SIZE}，允许 5..15 及以上）")
    parser.add_argument("--win", type=int, default=DEFAULT_WIN_LENGTH,
                        help=f"连珠数 K（默认 {DEFAULT_WIN_LENGTH}，需满足 3 <= K <= N）")
    parser.add_argument("--time", type=float, default=5.0,
                        help="每步时限（秒，默认 5.0）")
    parser.add_argument("--human-color", default="black",
                        help="人机模式下人类执子：black / white（默认 black）")
    parser.add_argument("--black", default=None,
                        help="机机模式下黑方 AI：内置别名(self/random/mcts) 或 AI 文件路径")
    parser.add_argument("--white", default=None,
                        help="机机模式下白方 AI：内置别名(self/random/mcts) 或 AI 文件路径"
                             "（默认 self，即与自己的 AI 自我对战）")
    parser.add_argument("--ai", default=None,
                        help="人机模式下 AI 使用的文件（默认本目录 gomoku_ai.py）")
    parser.add_argument("--quiet", action="store_true",
                        help="机机模式不逐步打印棋盘，只输出结果")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子（传给支持 seed 参数的 AI，用于复现实验）")
    parser.add_argument("--list-players", action="store_true",
                        help="列出可用的内置对手别名后退出")
    return parser


def make_ai(spec: str, player_id: int, board_size: int, win_length: int,
            seed=None) -> AIPlayer:
    """按描述符构造一个 AI 参与者。

    若 AI 类的构造函数接受 ``seed`` 关键字参数（例如本作业的 MCTS 实现），
    则把它透传进去，便于复现实验；否则按标准协议三参数构造。
    """
    path, name = resolve_player(spec, HERE)
    ai_class, _ = load_ai_class(path)
    player = AIPlayer(ai_class, player_id, name, board_size, win_length)
    if seed is not None:
        try:
            player.ai = ai_class(player_id, board_size, win_length, seed=seed)
        except TypeError:
            pass                     # 该 AI 不支持 seed，保持协议默认构造
    return player


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.list_players:
        print("可用的内置对手别名（--black / --white / --ai 都可使用）：")
        for alias, filename, note in list_builtin_players(HERE):
            exists = "存在" if os.path.isfile(os.path.join(HERE, filename)) else "缺失"
            print(f"  {alias:<14} -> {filename:<14} [{exists}]  {note}")
        print("\n也可以直接给出任意 AI 文件的路径，例如 "
              "--white ../classmate/gomoku_ai.py")
        return 0

    try:
        validate_config(args.board, args.win)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    if args.time <= 0:
        print("参数错误：--time 必须 > 0", file=sys.stderr)
        return 2

    ai_spec = args.ai or DEFAULT_AI

    try:
        if args.mode == "pvp":
            black = HumanPlayer(BLACK, "玩家1(黑)")
            white = HumanPlayer(WHITE, "玩家2(白)")
            verbose = True
        elif args.mode == "pve":
            human_color = COLOR_NAMES.get(str(args.human_color).lower())
            if human_color is None:
                print("参数错误：--human-color 只能是 black 或 white", file=sys.stderr)
                return 2
            human = HumanPlayer(human_color, "人类玩家")
            ai = make_ai(ai_spec, 3 - human_color, args.board, args.win, args.seed)
            black, white = (human, ai) if human_color == BLACK else (ai, human)
            verbose = True
        else:                                   # eve（机机对战）
            # 默认双方都用本作业的 AI（同强度自我对战）。
            # 若让白方默认落到 random_ai.py，观感上会像"白棋完全不防守"。
            black_spec = args.black or DEFAULT_AI
            white_spec = args.white or DEFAULT_AI
            black = make_ai(black_spec, BLACK, args.board, args.win, args.seed)
            white = make_ai(white_spec, WHITE, args.board, args.win, args.seed)
            verbose = not args.quiet
    except (FileNotFoundError, AttributeError, ImportError) as exc:
        print(f"加载 AI 失败：{exc}", file=sys.stderr)
        return 2

    try:
        game = play_game(black, white, args.board, args.win, args.time,
                         verbose=verbose)
    except ValueError as exc:
        print(f"\n对局中断：{exc}", file=sys.stderr)
        return 1

    print()
    print_stats([black, white])
    if black.kind == "ai" and hasattr(black, "ai"):
        stats = getattr(black.ai, "stats", {})
        if stats.get("nodes"):
            print(f"  {black.name} 搜索节点 {int(stats['nodes'])}，"
                  f"最深 {int(stats.get('depth', 0))} 层")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
