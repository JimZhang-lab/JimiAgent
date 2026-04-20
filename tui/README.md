# @jimiagent/tui

JimiAgent 的 React + Ink 终端 UI。通过 stdio 子进程与 Python `JimiAgent` 通信。

## 先决条件

- Node.js **>= 20**
- Yarn Classic **1.22.x**（不是 Berry / 非 PnP）
- Python 端已安装 JimiAgent（见项目根 `requirements.txt`）

## 首次运行

```bash
# 1) 一次性把 JimiAgent 安装成 Python package（注册 jimi 命令）
pip install -e .

# 2) 构建 TUI 前端
yarn --cwd tui install
yarn --cwd tui build
```

随后直接：

```bash
jimi chat                    # 新 React/Ink TUI（默认）
jimi chat -s <session-id>    # 恢复指定会话
jimi chat --classic          # 紧急回退旧 rich.live TUI（下一版本移除）
```

> 若未执行 `pip install -e .`，也可用 `python cli.py chat` 代替。  
> 若缺 Node 或 `tui/dist/cli.js` 不存在，命令会提示如何构建。

## 功能

**对话主线**

- 单条消息流式增量渲染，`assistant` 走内置 Markdown（**bold**/*italic*/`code`/fenced code/# 标题/- 列表/> 引用/链接）
- **ActivityBar**：prompt 上方常驻一条，实时显示 agent 正在做什么（`⚙ 工具名` / `◆ 生成回复中`）+ spinner + 毫秒级耗时
- `confirm_required` 事件弹 ConfirmBanner，**y** 允许 / **n** 拒绝
- 流式中 **Ctrl+C** 取消生成；空闲时 **Ctrl+C** 退出
- 上/下方向键回溯输入历史（会话内）

**客户端斜杠命令**

- `/help` — 帮助
- `/quit` | `/exit` — 退出
- `/clear` — 清空当前屏上消息（不删会话）
- `/new [标题]` — 新建会话
- `/sessions` — 打开会话列表
- `/memory` | `/mem` — 打开记忆管理面板（浏览 / 搜索 / 删除跨会话长期记忆）
- `/palette` — 打开命令面板
- `/theme dark|light|auto`
- `/copy` — 复制最后一条 assistant 内容到系统剪贴板
- `/vim [on|off]` — 切换 Vim 模式

**全局快捷键**

| 组合 | 动作 |
| --- | --- |
| **Ctrl+P** | 命令面板（fuzzy 搜 client + server 命令） |
| **Ctrl+S** | 会话列表 |
| **Ctrl+M** | 记忆管理面板 |
| **Esc** | 关闭弹层 / 关闭 slash 补全 |
| **Ctrl+C** | 取消生成 / 关闭弹层 / 退出 |
| **PgUp / PgDn** | 消息列表上/下翻 5 行 |
| **Ctrl+U / Ctrl+D** | 上/下翻 10 条 |
| **Ctrl+T / Ctrl+G** | 跳到最早 / 回到最新 |
| **/** | 在输入框开头敲 `/` 自动弹 slash 补全；↑/↓ 选，Tab 补全，Esc 关 |

**命令面板**：`fuse.js` 模糊匹配；↑/↓ 选择，Enter 执行；服务端命令来自 worker 的 `list_commands`。

**会话列表**：显示所有会话（当前高亮●），↑/↓ 选中，**Enter** 切换；**n** 新建；**d → Enter** 确认删除。

**记忆管理**（`/memory` 或 Ctrl+M）：

- ↑/↓ 选中；**Enter** 展开详情（再 Enter 收起）
- **Space** 切换当前项多选（`[✓]` 标记）；**a** 全选过滤结果；**A** 清空选择
- **d → Enter**：未多选时删单条；多选时批量删（顶部提示 "已选 N"，删除前顶部会强调一次 Enter 确认）
- **r** 刷新；**/** 进入搜索模式（向量 / FTS 混合，Esc 取消）
- **k** 循环过滤 kind：`all → semantic → episodic → procedural → triple → all`
- 底部展开区显示完整文本 + 元信息（id / kind / source_session / hits / score）；triple 记忆额外显示 `subject —predicate→ object`

**Diff 渲染**：`tool` 消息内容若看起来是 unified diff，自动用 `DiffView` 着色（add 绿 / del 红 / meta 蓝 / 前缀 +/- 保证无色终端可读）。

**Vim 模式**（`/vim` 开启）

- Normal：**i/a/A** 进入 Insert · **o** 清空并 Insert · **x** / **dd** 清空 buffer · **u** 撤销上一次提交 · **Esc** 取消操作
- Insert：正常打字；**Esc** 回到 Normal

## 开发

```bash
yarn --cwd tui dev          # 监听改动，自动重建并跑一遍
yarn --cwd tui typecheck    # TS 严格检查
yarn --cwd tui test         # vitest 单测（24 条）
yarn --cwd tui smoke        # spawn Python worker 跑协议 smoke
yarn --cwd tui lint         # ESLint
```

直接 debug：

```bash
node tui/dist/cli.js
```

会自动 spawn `python -m server.core.tui_worker` 作为 agent 后端。

## 目录

```
src/
├── cli.tsx                 # 入口：shebang + render(<App/>)
├── App.tsx                 # 注入 Providers
├── screens/                # MainLayout / CommandPalette / HistoryPanel / StatusBar
├── components/             # PromptInput / Messages / Diff / Markdown
├── engine/                 # Renderer 抽象 + InkRenderer
├── global/                 # Keybindings / Focus / Selection / Vim
├── transport/              # AgentTransport + StdioTransport
├── state/                  # Zustand store
├── themes/                 # dark / light / auto
├── protocol/               # NDJSON 事件类型
├── utils/                  # markdown / diff / ansi 辅助
└── __tests__/              # 单测
```

## 协议（stdio NDJSON）

- Node → Python：`{ kind: "chat" | "resume" | "cancel" | "list_sessions" | ... }`
- Python → Node：`{ type: "text" | "tool" | "session" | "confirm_required" | "ready" | "done" | ... }`

详见 `src/protocol/events.ts`。

## 环境变量

- `JIMI_TUI_SESSION` — 启动时恢复的 session id（由 `cli.py` 注入）
- `JIMI_TUI_SAFE=1` — 强制 16 色 palette（兼容老终端）
- `JIMI_TUI_WORKER_CMD` — 覆盖默认 worker 命令（默认 `python -m server.core.tui_worker`）
- `JIMI_TUI_THEME` — 覆盖主题（`auto` | `dark` | `light`）
- `DEBUG=jimi:*` — 打开 debug 日志（输出到 `~/.jimiagent/tui.log`，不污染 TTY）
