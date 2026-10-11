# 用网页直接提交到 GitHub（不用命令行）

> 目标仓库：<https://github.com/yph888/yupeihao>
> 需要上传的文件已打包：`yupeihao-github-upload.zip`（84.9 KB，含目录结构）
> 也可以直接用已解压好的文件夹：`upload-to-github\`

---

## 方法一：网页拖拽上传（推荐，最省事）

GitHub 网页支持**直接拖拽文件夹**上传，会自动保留目录结构。

### 步骤

1. **解压**
   在文件资源管理器里打开 `D:\dsh\assignment1\`，
   右键 `yupeihao-github-upload.zip` → **全部解压到当前文件夹**
   （或者直接使用已经解压好的 `D:\dsh\assignment1\upload-to-github\` 文件夹）

2. **打开仓库的上传页面**

   浏览器打开：<https://github.com/yph888/yupeihao>

   - 如果仓库是**空的**：页面上会有一个 **uploading an existing file** 链接，点它。
   - 如果仓库**已有文件**：点右上角 **Add file** → **Upload files**。

   直达链接（仓库为空时可用）：<https://github.com/yph888/yupeihao/upload/main>

3. **拖拽上传**

   打开解压出来的 `upload-to-github` 文件夹，**全选里面的内容**
   （`assignment1` 文件夹、`third_party` 文件夹、`README.md`、`PUSH_TO_GITHUB.md`），
   直接**拖进浏览器的上传区域**。

   > ⚠️ 关键点：要拖 **`assignment1` 这个文件夹本身**，不要在浏览器里把它点开再拖里面的文件。
   > 必须先拖文件夹，GitHub 才会保留 `assignment1/` 这一层路径——
   > 助教是从 `assignment1/gomoku_ai.py` 加载评赛 AI 的，路径不能变。

4. **确认文件列表**

   拖完后页面上会列出所有文件。请核对**至少有这些**：

   ```
   assignment1/gomoku_ai.py      ← 最重要，评赛文件
   assignment1/main.py
   assignment1/main_cli.py
   assignment1/gomoku_game.py
   assignment1/gui.py
   assignment1/mcts_ai.py
   assignment1/random_ai.py
   assignment1/report.md         ← 实验报告
   assignment1/requirements.txt
   assignment1/test_gomoku.py
   third_party/arena.py
   third_party/README.md
   README.md
   ```

   如果发现文件都平铺在根目录（没有 `assignment1/` 这一层），
   说明拖错了层级——把浏览器里的文件全部删掉重来，改成拖文件夹。

5. **提交**

   页面底部 **Commit changes** 区域：
   - 提交信息填：`BME1322 Assignment 1: generalized Gomoku with alpha-beta AI`
   - 选 **Commit directly to the `main` branch**
   - 点绿色 **Commit changes**

6. **确认结果**

   刷新仓库首页，应能看到 `assignment1` 文件夹。
   点进去确认 `gomoku_ai.py`、`report.md` 都在。

---

## 方法二：拖拽失败时，用网页编辑器逐个新建（兜底方案）

如果拖拽不生效（某些浏览器或网络环境会失败），可以用网页的代码编辑器逐个建文件。
文件不多，但请**优先保证下面这几个**，它们是评分必需的：

| 顺序 | 路径 | 说明 |
|---|---|---|
| 1 | `assignment1/gomoku_ai.py` | **评赛文件，最重要** |
| 2 | `assignment1/report.md` | 实验报告 |
| 3 | `assignment1/main.py` | 游戏主程序入口 |
| 4 | `assignment1/main_cli.py` | 命令行入口 |
| 5 | `assignment1/gomoku_game.py` | 棋局逻辑 |
| 6 | `assignment1/gui.py` | 图形界面 |
| 7 | `assignment1/mcts_ai.py` | MCTS（进阶方向） |
| 8 | `assignment1/random_ai.py` | 随机 AI 参考 |
| 9 | `assignment1/requirements.txt` | 依赖说明 |
| 10 | `assignment1/test_gomoku.py` | 单元测试 |
| 11 | `third_party/arena.py` | 课程裁判（随附代码） |
| 12 | `third_party/README.md` | 来源声明 |
| 13 | `README.md` | 仓库说明 |

### 操作方式

1. 仓库首页 → **Add file** → **Create new file**
2. 在文件名输入框里**直接输入带斜杠的路径**，例如：
   `assignment1/gomoku_ai.py`
   —— GitHub 会自动把 `assignment1/` 识别为目录，**不需要手动建文件夹**。
3. 把你电脑上 `D:\dsh\assignment1\assignment1\gomoku_ai.py` 的内容
   用记事本打开、全选、复制，粘贴到网页编辑器里。
4. 点 **Commit changes**。
5. 重复以上步骤建立其余文件。

> 提示：用记事本打开文件时请确认编码是 **UTF-8**（"另存为"里可以看到），
> 否则中文注释会变乱码。

---

## 上传后自查

```powershell
# 如果你电脑上装了 git，可以拉下来核对（可选）
cd D:\dsh
git clone https://github.com/yph888/yupeihao.git check-repo
cd check-repo
dir assignment1
```

或者直接在网页上确认这一点就够：
**仓库里存在 `assignment1/gomoku_ai.py` 这个文件**，且内容非空。

---

## 顺便说一下命令行为什么报错

你遇到的命令行报错，最可能是这两类：

### 1. 认证失败（最常见）

GitHub **已经不支持用账号密码推送**。如果你看到类似：

```
remote: Support for password authentication was removed
fatal: Authentication failed for 'https://github.com/yph888/yupeihao.git/'
```

必须改用 **Personal Access Token**：

1. 浏览器打开 <https://github.com/settings/tokens>
2. **Generate new token** → **Generate new token (classic)**
3. Note 随便填（如 `assignment1`），Expiration 选 30 天
4. **勾选 `repo`**（这一项必须勾，否则不能推送）
5. 拉到底点 **Generate token**，**复制**生成的 token（只显示一次）
6. 回到 PowerShell 执行 `git push -u origin main`，
   提示 `Username` 填 `yph888`，提示 `Password` **粘贴 token**（不是账号密码）

### 2. 推送被拒绝

如果看到 `rejected` / `fetch first` / `non-fast-forward`，
说明远端已经有提交（比如建库时勾了 README）：

```powershell
cd D:\dsh\assignment1
git fetch origin
git rebase origin/main
git push -u origin main
```

> **如果网页上传已经完成，就不需要再折腾命令行了**——
> 网页提交同样计入仓库的最后一次 push 时间（讲义 §8）。
