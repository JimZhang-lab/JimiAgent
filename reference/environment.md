# 项目运行环境

## 先决条件

- **Python** 3.10+（推荐 3.12，conda 环境名 `jimiAgent312`）
- **Node.js** >= 20（TUI 前端构建）
- **Yarn Classic** 1.22.x（不是 Berry）
- **编辑器 / 终端**：macOS Terminal / iTerm2 / Alacritty 等真彩色 TTY

## Python 环境

```bash
conda activate jimiAgent312
```

## 安装依赖

### Python

```bash
# 标准运行时依赖
pip install -r requirements.txt

# 一次性把项目注册成可编辑 package（获得 jimi 命令）
pip install -e .
```

安装完成后 `jimi` 会出现在当前 env 的 bin 目录（`which jimi`）；可取代 `python cli.py` 使用。

### TUI 前端

```bash
yarn --cwd tui install
yarn --cwd tui build
```

产物在 `tui/dist/cli.js`（一个 ESM bundle，由 `jimi chat` 透明 spawn）。

## 启动

### Gateway（后端 HTTP / WS 服务）

```bash
# 方式一：直接启动
python main.py

# 方式二：通过 CLI
jimi start                    # 或 python cli.py start
jimi start --verbose          # DEBUG 日志 + reload
jimi start --host 0.0.0.0 -p 18789
```

默认监听 `18789`；改端口见 `config/agent_config.yaml` 的 `agent.gateway.port`。

### 终端交互式对话

```bash
jimi chat                     # 新 React/Ink TUI（默认）
jimi chat -s <session-id>     # 恢复指定会话
jimi chat --classic           # 临时回退旧 rich.live TUI（下一版本移除）
```

**首次启动检查清单**：

1. `jimi chat` 不报错 → 进入 TUI
2. 顶部显示 `JimiAgent · React TUI · <model>`
3. 底部 PromptInput 边框高亮；输入 `你好` Enter 应收到流式回复
4. 按 **Ctrl+P** 弹命令面板 · **Ctrl+S** 会话列表 · **Ctrl+M** 记忆管理 · **Esc** 关闭弹层
5. 输入 `/` 立刻弹补全；Tab 选中一条命令
6. `~/.jimiagent/tui-worker.log` 应有 `tui_worker ready in X.XXs` 行

若 `jimi chat` 报 `未找到 tui/dist/cli.js`：

```bash
yarn --cwd tui build          # 重新构建
```

若报 `未找到 Node.js`：装 Node 20+ 后重开终端。

### 健康检查

```bash
jimi doctor
# Python 版本、关键依赖、API Key 可达性、Workspace/Skills、日志目录
```

## 常用子命令速查

```bash
# 基本
jimi start                    # 启动 Gateway
jimi chat                     # 进入 TUI
jimi chat -s <id>             # 恢复会话
jimi status                   # 系统状态
jimi doctor                   # 健康检查

# 配置
jimi config path              # 当前 yaml 路径
jimi config list              # 可修改字段 + 当前值
jimi config get <key>
jimi config set <key> <val>
jimi config edit              # $EDITOR 打开

# 会话
jimi sessions list
jimi sessions delete <id>

# 插件
jimi plugins list
jimi plugins install <spec>
jimi plugins doctor

# 自进化
jimi evolver run [--review]
jimi evolver signals
```

未执行 `pip install -e .` 时，上面任何 `jimi xxx` 都可用 `python cli.py xxx` 平替。

## TUI 开发 / 调试

```bash
yarn --cwd tui dev            # 监听改动自动重建 + 启动
yarn --cwd tui typecheck      # 严格 TS 检查
yarn --cwd tui test           # vitest 单测
yarn --cwd tui smoke          # 端到端协议烟囱（spawn worker 跑 ping/sessions/memories）
yarn --cwd tui lint           # ESLint
```

Worker 日志（不污染 TTY）：

```bash
tail -f ~/.jimiagent/tui-worker.log
```

关键环境变量：

- `JIMI_TUI_SESSION` — 启动时恢复的 session id（`jimi chat -s` 自动注入）
- `JIMI_TUI_WORKER_CMD` — 覆盖 worker 启动命令（默认 `{sys.executable} -m server.core.tui_worker`）
- `JIMI_TUI_THEME` — `auto` | `dark` | `light`
- `JIMI_TUI_WORKER_LOG_LEVEL` — `DEBUG` / `INFO` / `WARNING`

## 访问

- **Web UI**：<http://localhost:18789>
- **REST API**：<http://localhost:18789/api/status>
- **WebSocket**：<ws://localhost:18789/ws/chat>

## 目录速览

```
JimiAgent/
├── cli.py                  # Click CLI 入口（pip -e 后暴露 jimi 命令）
├── main.py                 # Gateway 直接启动
├── pyproject.toml          # [project.scripts] jimi = "cli:cli"
├── requirements.txt
├── config/agent_config.yaml
├── server/                 # 后端 Agent 核心
│   ├── core/agent.py
│   ├── core/tui_worker.py  # TUI 的 Python stdio worker
│   └── ...
├── tui/                    # React + Ink 前端
│   ├── src/                # TS 源码
│   ├── dist/cli.js         # 构建产物（Node 执行入口）
│   └── README.md           # TUI 详细文档
├── data/                   # 会话 / 记忆 SQLite DB
├── logs/                   # Gateway 日志
└── workspace/              # 用户工作区、SKILL.md、USER.md
```
