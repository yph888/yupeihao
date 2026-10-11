# third_party — 课程随附的参考代码（非本人实现）

本目录下的文件由**课程讲义随附发布**，仅供协议自测与实力基线使用。
为遵守学术诚信要求（"引用任何外部资料须在报告中注明出处"），
在此明确标注来源，并在报告中说明用途。

| 文件 | 来源 | 用途 | 是否修改 |
|---|---|---|---|
| `arena.py` | 课程随附（与讲义"五、课程提供的裁判工具"一同发布） | 加载两份符合统一协议的 AI 进行比赛，用于协议合规性自测 | **未修改** |
| `../assignment1/random_ai.py` | 课程随附（讲义 §2.4 参考实现） | 随机合法落子的实力基线对手 | **未修改**（仅复制到提交目录以便自测） |

## 原始文件校验

两份文件均从课程提供的压缩包 `assignment-1-code.zip` 中直接解压得到：

```
assignment-1-code.zip
├── arena.py        9340 bytes
└── random_ai.py     545 bytes
```

## 使用方式

```bash
# 主 AI 对随机 AI（默认 15x15 五连、每步 5 秒、交换先后手两局）
python third_party/arena.py assignment1/gomoku_ai.py assignment1/random_ai.py \
       --board 15 --win 5 --time 5.0 --games 2
```

`arena.py` 的判定规则（超时 / 非法落子 / 异常直接判负）与讲义 §2.3 一致，
因此它的结论可以直接作为"协议合规"的证据；报告 §4.1 引用了它的运行结果。

## 声明

本人实现的部分（`gomoku_ai.py`、`gomoku_game.py`、`gui.py`、`game_cli.py`、
`main.py`、`main_cli.py`、`mcts_ai.py` 及全部测试与实验脚本）为独立完成。
`arena.py` 与 `random_ai.py` **没有**参与 AI 的任何决策过程，也未被复制或
改写进 AI 的实现中。
