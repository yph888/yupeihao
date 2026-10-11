# -*- coding: utf-8 -*-
"""main.py -- 广义五子棋游戏主程序入口（BME1322 Assignment 1）。

默认启动图形界面（``gui.py``，tkinter，标准库自带）；若显式给出 ``--cli``
或当前环境没有图形显示，则退回到命令行入口（``main_cli.py``）。

示例::

    python main.py                                  # 打开 GUI，默认 15x15 五连
    python main.py --board 9 --win 4                # GUI，9x9 四连
    python main.py --cli --mode pvp                 # 命令行人人对战
    python main.py --cli --mode pve --board 11 --win 5 --time 2.0
    python main.py --cli --mode eve --black gomoku_ai.py --white mcts --quiet
"""

from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from gomoku_game import DEFAULT_BOARD_SIZE, DEFAULT_WIN_LENGTH, validate_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="广义五子棋 (m,n,k)-Game：支持人人 / 人机 / 机机三种模式")
    parser.add_argument("--cli", action="store_true",
                        help="强制使用命令行界面（默认优先图形界面）")
    parser.add_argument("--mode", choices=("pvp", "pve", "eve"), default=None,
                        help="命令行模式：pvp=人人，pve=人机，eve=机机")
    parser.add_argument("--board", type=int, default=DEFAULT_BOARD_SIZE,
                        help=f"棋盘边长 N（默认 {DEFAULT_BOARD_SIZE}）")
    parser.add_argument("--win", type=int, default=DEFAULT_WIN_LENGTH,
                        help=f"连珠数 K（默认 {DEFAULT_WIN_LENGTH}）")
    parser.add_argument("--time", type=float, default=5.0,
                        help="每步时限（秒，默认 5.0）")
    parser.add_argument("--human-color", default="black",
                        help="人机模式下人类执子：black / white")
    parser.add_argument("--black", default=None, help="机机模式黑方 AI")
    parser.add_argument("--white", default=None, help="机机模式白方 AI")
    parser.add_argument("--ai", default=None, help="人机模式 AI 文件路径")
    parser.add_argument("--quiet", action="store_true", help="机机模式只看结果")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--opponent", default=None,
                        help="GUI 机机模式下的第二个 AI 文件（默认 random_ai.py）")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        validate_config(args.board, args.win)
    except ValueError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2

    if not args.cli:
        try:
            import tkinter  # noqa: F401
            gui_available = True
        except Exception:
            gui_available = False
        if gui_available:
            import gui
            return gui.run(ai_path=args.ai, opponent_path=args.opponent)
        print("未检测到可用的图形环境，自动切换到命令行界面。", file=sys.stderr)

    # ---- 命令行模式 ----
    import main_cli
    cli_argv: list = ["--board", str(args.board), "--win", str(args.win),
                      "--time", str(args.time)]
    if args.mode:
        cli_argv += ["--mode", args.mode]
    if args.human_color:
        cli_argv += ["--human-color", args.human_color]
    if args.black:
        cli_argv += ["--black", args.black]
    if args.white:
        cli_argv += ["--white", args.white]
    if args.ai:
        cli_argv += ["--ai", args.ai]
    if args.quiet:
        cli_argv.append("--quiet")
    if args.seed is not None:
        cli_argv += ["--seed", str(args.seed)]
    return main_cli.main(cli_argv)


if __name__ == "__main__":
    raise SystemExit(main())
