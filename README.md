# BME1322 Assignment 1 — 广义五子棋（Gomoku）对抗搜索

本仓库按课程要求组织：**评赛文件位于 `assignment1/gomoku_ai.py`**
（路径与类名不可更改，助教将直接从该路径加载）。

## 提交内容（`assignment1/`）

| 文件 | 说明 |
|---|---|
| `gomoku_ai.py` | **统一对战协议 v1 的 AI**（MiniMax + Alpha-Beta + 增量棋型评估 + 迭代加深），唯一被评赛加载的文件 |
| `mcts_ai.py` | 进阶方向：MCTS/UCT 实现（同一协议，用于对比实验） |
| `gomoku_game.py` | 棋局逻辑：棋盘、落子、K 连判定、和棋、悔棋、(N,K) 配置校验 |
| `gui.py` | tkinter 图形界面（人人 / 人机 / 机机三种模式） |
| `game_cli.py` | 命令行对局逻辑与玩家/AI 动态装载 |
| `main_cli.py` | 命令行入口（`--mode pvp/pve/eve`） |
| `main.py` | 游戏主程序入口（默认 GUI，`--cli` 回落命令行） |
| `random_ai.py` | 课程提供的随机 AI 参考实现（便于自测） |
| `test_gomoku.py` | 单元测试（45 项，标准库 `unittest`） |
| `smoke_test.py` / `smoke_test_gui.py` / `smoke_test_mcts.py` | 冒烟测试（协议合规、GUI 无头、MCTS） |
| `report.md` | 实验报告 |
| `requirements.txt` | 依赖说明（**仅标准库，无第三方依赖**） |

## 快速开始

```bash
cd assignment1

# 单元测试
python -m unittest test_gomoku -v

# 图形界面（默认 15x15 五连，可在界面切换 N/K/时限）
python main.py

# 命令行：人人 / 人机 / 机机
python main.py --cli --mode pvp
python main.py --cli --mode pve --board 9 --win 4 --time 1.0 --human-color black
python main.py --cli --mode eve --black gomoku_ai.py --white mcts --quiet

# 查看可用的内置对手别名
python main_cli.py --list-players
```

## 提交前的自查

```bash
# 逐条核对讲义要求（42 项检查：文件结构 / 协议接口 / 裁判规则 /
# 不硬编码 N,K / 三种模式 / 报告章节 / 学术诚信声明）
python experiments/audit_submission.py
```

## 用课程裁判自测

```bash
# 主 AI vs 随机 AI（15x15 五连，交换先后手）
python third_party/arena.py assignment1/gomoku_ai.py assignment1/random_ai.py \
       --board 15 --win 5 --time 5.0 --games 2

# 与其他同学的 AI 对战（9x9 四连）
python third_party/arena.py assignment1/gomoku_ai.py classmate/gomoku_ai.py \
       --board 9 --win 4 --time 1.0 --games 1 --quiet
```

## 实验复现

`experiments/` 下是实验与评估工具（不属于提交内容，仅供复现报告数据）：

```bash
python experiments/run_all.py            # 全部对局实验（结果写 results/*.json）
python experiments/protocol_sweep.py     # 28 组 (N,K) 协议合规巡检
python experiments/timing_stress.py      # 极紧时限（最低 0.05s）超时检测
python experiments/bench_search.py       # 搜索效率与着法排序消融
python experiments/summarize.py          # 汇总生成 results/RESULTS.md
```

## 环境

Python 3.8+（开发与实验使用 CPython 3.10.0），**仅标准库**。
图形界面需要 `tkinter`（CPython 官方发行版自带）；若不可用，
所有功能都可通过 `main_cli.py` 在命令行完成。

## 第三方与学术诚信

* `third_party/arena.py`、`assignment1/random_ai.py` 为**课程讲义随附**的
  参考代码，仅用于自测与实力基线，未作修改（见 `third_party/README.md`）。
* AI 的全部决策均来自运行时的搜索与评估，未调用任何外部五子棋引擎、
  在线服务或网络资源；未内嵌针对特定对手的策略；未内嵌预计算的固定应答
  （仅保留"首手下中心"这类讲义明确允许的一般性启发规则）。
* 报告中对算法来源的引用见 `assignment1/report.md` 附录 C。
