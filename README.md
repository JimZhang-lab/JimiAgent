# JimiAgent

> 基于 **LangChain / LangGraph** 的个人 AI 助手。Skills 语义召回、SQLite 持久化记忆、Web UI/CLI/REST 多前端、OpenClaw 插件生态兼容、多模态图文输入。

| Python | LangGraph | FastAPI | 许可证 |
| :-: | :-: | :-: | :-: |
| 3.12 | 1.1+ | 0.115+ | MIT |

---

## 📑 目录

- [为什么做这个](#-为什么做这个)
- [亮点速览](#-亮点速览)
- [架构一图流](#-架构一图流)
- [目录结构](#-目录结构)
- [5 分钟跑起来](#-5-分钟跑起来)
- [配置详解](#-配置详解)
- [使用方式](#-使用方式)
- [斜杠命令](#-斜杠命令)
- [多模态（VLM）](#-多模态vlm)
- [插件生态（OpenClaw 兼容）](#-插件生态openclaw-兼容)
- [长期记忆](#-长期记忆)
- [编写新 Skill](#-编写新-skill)
- [API 端点](#-api-端点)
- [部署](#-部署)
- [CLI 速查](#-cli-速查)
- [故障排查](#-故障排查)
- [License](#-license)

---

## 💡 为什么做这个

市面上的 AI 助手多是 SaaS，数据、记忆、插件都不是自己的。JimiAgent 的目标：

- **一切本地化**：SQLite 存记忆、yaml 存配置、workspace 存人格 —— 没有云依赖，断网也能跑（模型除外）。
- **前端随便挑**：同一个 Agent，Web UI、终端 TUI、REST、WebSocket、SSE、Webhook 六种接入方式**后端零差异**。
- **Skills 按需加载**：每轮对话根据问题语义挑 Top-K 工具挂载，上下文不爆炸。
- **兼容 OpenClaw 插件生态**：直接读社区里的 `openclaw.plugin.json` / `.claude-plugin` / `.cursor-plugin` / `.codex-plugin`，不绑死任何一家。
- **可观察、可改、可扩**：全配置 yaml 化，所有核心路径（记忆、上下文引擎）都留了插件替换位。

不是要造一个"最强 Agent"，而是**"你自己的 Agent，跑在你自己的机器上"**。

---

## ✨ 亮点速览

| 能力 | 说明 |
| --- | --- |
| LangGraph ReAct Agent | `create_react_agent` 组装推理/行动循环，工具动态挂载 |
| Skills Hybrid 召回 | BM25 + Vector + RRF 融合 · TTL 缓存 · 精确名短路 · `always` 常驻 |
| SQLite 持久化 | `AsyncSqliteSaver` + 独立 `MemoryStore`，每会话 `thread_id` 隔离 |
| 跨会话长期记忆 | 3 类记忆（semantic/episodic/procedural）+ FTS5/向量双路检索 |
| 时间知识图 | Triple store `(s,p,o)` + `valid_from/valid_until`，支持"当时发生了什么"查询 |
| 多模态 | 独立 VLM / VL-Embedding 配置，图片可粘贴/拖拽/上传 |
| Computer Use | AI 操控电脑：鼠标/键盘/截图 + Playwright 浏览器 + VLM 任务循环，4 层安全兜底 |
| 插件生态 | OpenClaw 兼容宿主，静态解析 manifest，`slots` 可替换内置实现 |
| 模型 Failover | `Runnable.with_fallbacks` 主→备切换 |
| 自动 compact | 超阈值时 LLM 摘要前文 + 保留最近 N 条 |
| 斜杠命令 | 所有前端统一的 `/status` `/new` `/compact` `/config` `/model` `/reload` `/edit`… |
| Webhook 渠道 | `POST /api/webhook/{name}`，sender_id 稳定哈希映射会话，可选 HMAC 密钥 |
| Cron 调度器 | 标准 5 字段表达式，热重载；支持 LLM job 和 evolver job |
| TUI 终端 | `prompt_toolkit` 斜杠补全 + 历史 + 多行 + `rich.live` 流式 Markdown |
| 日志系统 | 等级/格式/输出（控制台/文件/双写）全 yaml 化，文件自动轮转 |

---

## 🏗️ 架构一图流

```
                        ┌──────────────────────────────────────────────┐
                        │                  接入层（前端）              │
                        ├──────────┬──────────┬──────────┬─────────────┤
                        │ Web UI   │  CLI TUI │   REST   │ WS/SSE/Webhook│
                        └─────┬────┴─────┬────┴─────┬────┴──────┬──────┘
                              │          │          │           │
                              └──────────┴──────────┴───────────┘
                                         │
                                  FastAPI Gateway
                                         │
                        ┌────────────────┼────────────────┐
                        │                │                │
                CommandDispatcher  ActivationPolicy   ChannelRouter
                        │                │                │
                        └────────────────┼────────────────┘
                                         │
                            ┌────────────┴──────────────┐
                            │       JimiAgent 核心      │
                            │  ┌─────────────────────┐  │
                            │  │  LangGraph Agent    │  │  ← cache by tool-sig
                            │  │  (create_react...)  │  │
                            │  └─────────┬───────────┘  │
                            │            │              │
                            │  Workspace │ SkillRetriever (LlamaIndex)
                            │  (4 层 MD) │  └→ StructuredTool
                            │            │
                            │  MemoryStore (跨会话长期记忆)
                            │  Memory    (AsyncSqliteSaver / thread-level)
                            │  Scheduler (cron)
                            │  Evolver   (GEP 自进化)
                            │  Plugins   (OpenClaw 兼容层)
                            └───────────────────────────┘
                                         │
                        ┌────────────────┼────────────────┐
                        │                │                │
               data/memory.db   data/memory_store.db   data/skill_index/
               （thread 消息）   （语义/事件/规则/triple） （向量索引）
```

**数据流（一次对话）**：

1. 前端消息抵达 Gateway → `ChannelRouter` 选 agent
2. `ActivationPolicy` 检查 `mention` / `always` 策略
3. `CommandDispatcher` 先看是否斜杠命令（是则直接返回，不走 LLM）
4. `MemoryStore.build_recall_block()` 注入 top-K 记忆到 system prompt
5. `SkillRetriever.retrieve(query)` 取 Top-K skills → 动态构建工具集
6. 工具签名命中缓存则复用已编译 `LangGraph`，否则新建
7. ReAct 循环跑完后：`SessionManager.increment_message_count`、后台 `_hot_extract` 异步抽记忆

---

## 📁 目录结构

```
JimiAgent/
├── config/
│   ├── agent_config.example.yaml       配置模板（入库）
│   └── agent_config.yaml               真实配置（.gitignore 屏蔽）
├── workspace/                          Agent 人格与技能库
│   ├── SOUL.md                         人格定义
│   ├── AGENTS.md                       行为规则
│   ├── TOOLS.md                        环境信息
│   ├── USER.md                         个人上下文（.gitignore 屏蔽）
│   └── skills/<name>/
│       ├── SKILL.md                    YAML frontmatter + 文档
│       └── *.py                        可选实现
├── server/
│   ├── api/
│   │   ├── gateway.py                  FastAPI app + lifespan + static 挂载
│   │   ├── routes.py                   REST / WS / SSE / Webhook / Upload
│   │   └── models.py                   Pydantic schema
│   ├── config/
│   │   ├── settings.py                 yaml + JIMI_* 环境变量加载
│   │   ├── config_io.py                ruamel.yaml roundtrip 读写
│   │   └── logging_setup.py            日志初始化
│   ├── core/
│   │   ├── agent.py                    JimiAgent（LangGraph ReAct）
│   │   ├── agent_registry.py           多 Agent 路由表
│   │   ├── commands.py                 斜杠命令派发器
│   │   ├── memory.py                   SQLite Checkpointer 封装
│   │   ├── memory_store.py             跨会话长期记忆（FTS5 + 向量 + triple）
│   │   ├── session.py                  会话元信息管理
│   │   ├── workspace.py                4 层 MD 拼 system prompt
│   │   ├── skill_loader.py             SKILL.md → StructuredTool
│   │   ├── skill_retriever.py          LlamaIndex 语义召回
│   │   ├── skill_exec.py               exec runtime（bash/node/python…）
│   │   ├── multimodal.py               VLM 描述 + VL-embedding 写记忆
│   │   ├── plugin_manifest.py          OpenClaw manifest 静态解析
│   │   ├── plugin_registry.py          插件发现 / 启用 / 诊断
│   │   ├── plugin_slots.py             memory / contextEngine 替换位
│   │   ├── plugin_installer.py         本地 / Git / npm 安装
│   │   ├── file_tools.py               read/write/list_dir 白名单文件工具
│   │   ├── scheduler.py                cron 调度器
│   │   ├── evolver.py                  GEP 自进化引擎
│   │   ├── tui.py                      终端 TUI
│   │   └── ports/                      MemoryPort / ContextEnginePort Protocol
│   ├── plugin/channels/                接入渠道（webchat / webhook / terminal）
│   └── static/                         Web UI（index.html / app.js / style.css）
├── data/                               运行时产物（.gitignore）
│   ├── memory.db                       LangGraph checkpointer
│   ├── memory_store.db                 长期记忆 + FTS5 + triple
│   ├── sessions.json                   会话元信息
│   ├── skill_index/                    LlamaIndex 持久化索引
│   ├── uploads/<yyyymmdd>/             上传图片
│   ├── tool_log.jsonl                  工具调用审计
│   └── gep/events.jsonl                Evolver 事件审计
├── logs/                               日志输出目录
├── main.py                             uvicorn 入口
├── cli.py                              命令行工具入口
└── requirements.txt
```

---

## 🚀 5 分钟跑起来

### 1. 克隆 + 环境

```bash
git clone https://github.com/JimZhang-lab/JimiAgent.git
cd JimiAgent

# 推荐 conda（项目内脚本默认指向 jimiAgent312）
conda create -n jimiAgent312 python=3.12 -y
conda activate jimiAgent312
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config/agent_config.example.yaml config/agent_config.yaml
```

最小可用配置（改 `api_key` 和 `base_url` 即可）：

```yaml
agent:
  model:
    provider: openai
    model_id: qwen3-next-80b        # 或 gpt-4o / claude-3-5-sonnet-20241022
    api_key: sk-your-real-key
    base_url: https://your-openai-compatible-endpoint/v1

  embedding:
    model: text-embedding-3-small   # 非标准名（qwen3-embedding-8b）也自动兼容
    # api_key / base_url 留空则复用上面 model 的
```

### 3. 启动

```bash
python cli.py start              # 启动 FastAPI Gateway（或直接 python main.py）
# 另一个终端：
python cli.py chat               # 进入 TUI 聊天
```

### 4. 打开

| 端点 | 地址 |
| --- | --- |
| Web UI | http://localhost:18789 |
| Swagger | http://localhost:18789/docs |
| ReDoc | http://localhost:18789/redoc |
| WebSocket | ws://localhost:18789/ws/chat |
| REST | http://localhost:18789/api/chat |

### 5. 健康检查

```bash
python cli.py doctor
```

输出会列出：依赖版本、API Key 可达性、workspace/skills 完整度、日志目录写入权限等。

---

## ⚙️ 配置详解

完整模板见 `config/agent_config.example.yaml`。关键字段如下。

### 模型

```yaml
agent:
  model:
    provider: openai                 # openai / anthropic
    model_id: qwen3-next-80b
    temperature: 0.7
    max_tokens: 8192
    streaming: true
    api_key: sk-...
    base_url: https://api.openai.com/v1

    # 可选：主模型失败时按顺序切到备选
    fallbacks:
      - provider: openai
        model_id: gpt-4o-mini
        api_key: sk-...
      - provider: anthropic
        model_id: claude-3-5-sonnet-20241022
        api_key: sk-ant-...
```

### Embedding（Skills 召回）

```yaml
agent:
  embedding:
    model: text-embedding-3-small    # 标准名直接用
    # model: qwen3-embedding-8b      # DashScope 自定义名：自动走兼容模式
    api_key: ""                      # 留空复用 agent.model.api_key
    base_url: ""                     # 留空复用 agent.model.base_url
```

### Skills 检索（Hybrid）

```yaml
agent:
  skills:
    top_k: 3
    similarity_threshold: 0.5        # 仅 vector_only 模式用；hybrid 下的 RRF score 不适用
    index_persist_dir: ./data/skill_index

    retrieval_mode: hybrid           # hybrid / vector_only / bm25_only
    query_cache_ttl_seconds: 30      # 0 = 禁用缓存；命中 query 跳过 embedding API
    query_cache_size: 128
    exact_match_short_circuit: true
    bm25_weight_name: 3              # BM25 侧 name tf 放大倍数
    bm25_weight_description: 2
```

### 多模态（可选）

```yaml
agent:
  vlm:
    model_id: qwen-vl-max            # 未配置就沿用 agent.model
  vl_embedding:
    model: multimodal-embedding-v1   # 未配置就沿用 agent.embedding
```

### 记忆

```yaml
agent:
  memory:
    backend: sqlite
    sqlite_path: ./data/memory.db                   # 会话消息
    enabled: true
    store_path: ./data/memory_store.db              # 跨会话长期记忆
    extract_mode: per_turn                          # off / on_compact / per_turn
    extract_timeout_seconds: 10
    recall_max_inject: 5
    procedural_inject_max: 10
    dedup_threshold: 0.92
    embedding_model: auto                           # auto / off
    namespace: auto                                 # auto 用 agent_name 强隔离
```

### 会话

```yaml
agent:
  session:
    max_history_messages: 50
    compact_threshold: 40
    default_think_level: medium                     # low / medium / high
```

### Gateway

```yaml
gateway:
  host: 0.0.0.0
  port: 18789
  cors_origins: ["*"]                               # 生产务必白名单
```

### 渠道

```yaml
channels:
  webchat: { enabled: true }
  terminal: { enabled: true }
  webhook:
    enabled: true
    secret: your-shared-secret                      # 留空不校验
    session_prefix: wh
    dmPolicy: pairing                               # open / pairing / closed
    allowFrom: ["admin_*"]                          # 支持 fnmatch 通配
```

### 日志

```yaml
logging:
  level: INFO                                       # DEBUG / INFO / WARNING / ERROR
  format: rich                                      # rich / plain（仅控制台）
  output: console                                   # console / file / both
  file_path: ./logs/jimi.log
  rotate_mb: 10
  backup_count: 5
```

### 调度器

```yaml
scheduler:
  enabled: true
  tick_seconds: 30
  jobs:
    - id: daily-brief
      schedule: "0 9 * * *"
      prompt: "用三句话总结今天需要关注的事"
      session_title: 每日提醒
      enabled: true
    - id: evolver-hourly
      kind: evolver                                 # 不走 LLM，直接跑自进化
      schedule: "0 * * * *"
      strategy: balanced
```

### 插件

```yaml
plugins:
  enabled: true
  allow: []                                         # 空=全开
  deny: []
  load:
    paths:
      - ~/Projects/oss/voice-call-extension
  slots:
    memory: memory-core                             # none / <plugin-id>
    contextEngine: none
  entries:
    voice-call:
      enabled: true
      config:
        provider: twilio
  install_registry: ~/.openclaw/plugins
  npm_registry: ~/.openclaw/npm_store
```

### 环境变量覆盖（不落盘）

| 变量 | 作用 |
| --- | --- |
| `JIMI_CONFIG` | 自定义 yaml 路径 |
| `JIMI_API_KEY` / `JIMI_BASE_URL` / `JIMI_MODEL` | 主模型 |
| `JIMI_EMBEDDING_API_KEY` / `JIMI_EMBEDDING_BASE_URL` / `JIMI_EMBEDDING_MODEL` | Embedding |
| `JIMI_VLM_API_KEY` / `JIMI_VLM_BASE_URL` / `JIMI_VLM_MODEL` | VLM |
| `JIMI_VL_EMBEDDING_API_KEY` / `JIMI_VL_EMBEDDING_BASE_URL` / `JIMI_VL_EMBEDDING_MODEL` | VL-Embedding |
| `JIMI_GATEWAY_HOST` / `JIMI_GATEWAY_PORT` | Gateway |
| `JIMI_LOG_LEVEL` / `JIMI_LOG_OUTPUT` / `JIMI_LOG_FILE` | 日志 |

示例：

```bash
JIMI_LOG_LEVEL=DEBUG JIMI_GATEWAY_PORT=18790 python main.py
```

---

## 🖥️ 使用方式

### Web UI

浏览器打开 `http://localhost:18789`：

- 左侧会话列表 · 中间聊天面板 · 输入区域
- 附件按钮 / 粘贴 / 拖拽图片 → 自动上传 → 走 VLM
- 支持 Markdown 渲染（代码块、表格、列表）

### 终端 TUI

```bash
python cli.py chat                   # 复用默认会话
python cli.py chat -s <session-id>   # 恢复指定会话
```

特性：

- 打 `/` 弹出命令补全菜单（带描述）
- `Alt+Enter` 换行 · `Enter` 发送
- `Ctrl+C` 中断当前流式回复但不退出 · `Ctrl+D` 真正退出
- 输入历史持久化到 `~/.jimiagent/chat_history`
- 流式 Markdown 渲染 · 工具调用以独立面板展示

### REST

```bash
curl -X POST http://localhost:18789/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'
```

### WebSocket

```javascript
const ws = new WebSocket("ws://localhost:18789/ws/chat");
ws.send(JSON.stringify({ message: "你好", session_id: "可选" }));
ws.onmessage = (e) => {
  const evt = JSON.parse(e.data);
  // { type: "session", session_id }
  // { type: "stream", content }
  // { type: "tool", name }
  // { type: "end" }
  // { type: "error", content }
};
```

### SSE

```bash
curl -N -X POST http://localhost:18789/api/events \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}'
```

响应形如：

```
event: session
data: {"session_id": "502c2a71"}

event: stream
data: {"content": "你好"}

event: end
data: {}
```

### Webhook

外部系统（IM bot、表单、自建服务）都能以 webhook 方式接入：

```bash
curl -X POST http://localhost:18789/api/webhook/telegram \
  -H "Content-Type: application/json" \
  -H "X-Jimi-Webhook-Secret: your-shared-secret" \
  -d '{"sender_id": "user42", "message": "你好"}'
```

响应（同步等待 Agent 完整回复）：

```json
{
  "sender_id": "user42",
  "session_id": "wh-1c7082c8",
  "response": "你好..."
}
```

`sender_id` 经 MD5 前 8 位 + 前缀生成稳定 `session_id`，同一 sender 的消息自动归到同一会话。

---

## 💬 斜杠命令

任何前端里消息以 `/` 开头都走 `CommandDispatcher`，不进入 LLM：

| 命令 | 说明 |
| --- | --- |
| `/help` | 列出全部命令 |
| `/status` | 模型 · Skills · 会话 · 当前配置 |
| `/new [标题]` | 新建并切换到新会话 |
| `/sessions` | 列出最近 10 个会话 |
| `/reset` | 清空当前会话历史 |
| `/compact [keep=N]` | 摘要前文 + 保留最近 N 条 |
| `/think low\|medium\|high` | 调整推理深度 |
| `/verbose on\|off` | 是否展示工具调用链 |
| `/trace on\|off` | 是否透出 LangGraph 内部事件 |
| `/usage off\|tokens\|full` | 切换 usage 粒度 |
| `/activation mention\|always` | 激活策略（mention 下需 `@agent-name`） |
| `/bash_confirm on\|off` | bash 写命令是否二次确认 |
| `/config [key] [value]` | 查看 / 修改 yaml（持久化） |
| `/model [id]` | 查看 / 切换主模型（持久化 + 重建 LLM） |
| `/reload` | 重读 yaml + 重建 LLM + 清 Agent 缓存 |
| `/edit` | 在 `$EDITOR` 打开 yaml，退出后自动 reload |
| `/restart` | 重建 Skills 索引 + reload workspace |
| `/memories [kind=... \| clear yes]` | 查看/清空长期记忆 |
| `/recall <query>` | 检索长期记忆 |
| `/recall_at <ts> [subject=... predicate=...]` | 时间三元组查询 |
| `/forget <id>` | 软删某条记忆 |
| `/plugin list\|show\|enable\|disable\|install\|uninstall\|doctor` | 插件管理 |
| `/computer_use [on\|off\|status]` | AI 操控电脑总开关 |
| `/computer_use <desktop\|browser\|loop> on\|off` | 分层开关 |
| `/clear` | 清屏（仅 TUI 本地） |
| `/quit` \| `/exit` \| `/q` | 退出（仅 TUI 本地） |

---

## 🖼️ 多模态（VLM）

独立的 `vlm` / `vl_embedding` 配置，未配置即回退到主 `model` / `embedding`，即便只配了一个 `gpt-4o` 也能直接吃图。

### 用法

**Web UI**：输入框夹子按钮 / 粘贴截图 / 拖拽文件 → 图片自动 `POST /api/upload`，拿到公开 URL 后随消息发送给 VLM。

**REST**：

```bash
# 1) 先上传拿 URL
curl -F file=@cat.png -F describe=true http://localhost:18789/api/upload
# {"ok":true,
#  "url":"http://localhost:18789/uploads/20260419/abcd-cat.png",
#  "description":"一只虎斑猫卧在窗台上...",
#  "memory_id":7,
#  "embedding_used":true}

# 2) 带图聊天
curl -X POST http://localhost:18789/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"分析这张图","images":["http://localhost:18789/uploads/20260419/abcd-cat.png"]}'
```

**WebSocket**：payload 加 `images: ["..."]` 字段即可走 VLM 流式。

### 关键点

- `images` 非空 → Agent 走 `self.vlm`（未配置时与 `self.llm` 同一实例，零开销）
- `HumanMessage.content` 切换为 OpenAI 多部分格式 `[{type:text}, {type:image_url}]`
- `/api/upload?describe=true` 后台调 VLM 生成一句描述，用 VL-embedding 向量化后写入 MemoryStore（`kind=episodic, subject=upload`），日后 `/recall` 能检索到
- 上传走服务端（非 base64 直传），URL 对所有主流 VLM provider 兼容
- mime/ext 双白名单 + 10MB 上限，保存在 `data/uploads/<yyyymmdd>/<hash>-<name>`

---

## 🤖 Computer Use（AI 操控电脑）

三层完整的 computer-use 能力栈：**桌面原生**（鼠标/键盘/截图）+ **浏览器自动化**（Playwright）+ **VLM 任务级 Loop Agent**。**默认全关** — 启用需显式 yaml 开关或 `/computer_use on` 斜杠命令。

### 安装 + 授权

```bash
pip install pyautogui mss playwright
playwright install chromium              # ~200MB
```

macOS 首次需授权：**系统设置 → 隐私与安全 → 辅助功能 / 屏幕录制** → 允许 Python / Terminal。

### 开启

```yaml
agent:
  computer_use:
    enabled: true                        # 总开关
    desktop_enabled: true                # 鼠标/键盘/截图
    browser_enabled: true                # Playwright
    loop_agent_enabled: false            # VLM 循环（最敏感，默认关）

    browser_mode: launch                 # launch（全新 Chromium）或 cdp（接管已开 Chrome）
    browser_headless: false              # 默认 headful 便于观察

    app_denylist: ["com.apple.Terminal", "com.apple.keychainaccess"]
    domain_denylist: ["*bank*", "*.alipay.com"]
```

或直接在 TUI 里：

```
/computer_use on                         # 总开关
/computer_use desktop on                 # 分层
/computer_use browser on
/computer_use loop on                    # 开 VLM 循环
/computer_use status                     # 查看状态
```

### 18 个工具

| 层 | 工具 |
| --- | --- |
| Desktop (8) | `screen_capture` / `screen_info` / `mouse_move` / `mouse_click` / `mouse_drag` / `mouse_scroll` / `keyboard_type` / `keyboard_press` |
| Browser (9) | `browser_open` / `click` / `type` / `scroll` / `screenshot` / `extract` / `wait_for` / `press` / `close` |
| Loop Agent (1) | `computer_task(goal, max_steps=20)` — VLM 观察-决策-执行闭环 |

### 使用示例

**桌面操作**（LLM 自己决定调用时机）：

```
你：截图看看现在哪个窗口在前台
Agent：[调 screen_capture] [调 screen_info] 当前是 VSCode，分辨率 1920x1080 …
```

**浏览器任务**：

```
你：打开 https://news.ycombinator.com，抽取前 5 条标题
Agent：[browser_open] [browser_wait_for "tr.athing"] [browser_extract "tr.athing .titleline"]
       → 返回 5 条标题
```

**任务级 Loop Agent**（最敏感，默认关）：

```
你：computer_task(goal="在淘宝搜羽绒服并截图前 3 条")
Agent 内部：
  step1  screen_capture → VLM 看截图 → 决定 {"action":"browser_open","url":"taobao.com"}
  step2  screen_capture → VLM → {"action":"browser_type","selector":"input[name=q]","text":"羽绒服"}
  step3  ...
  step7  {"action":"done","result":"已截图并抽取前 3 条"}
```

### 安全四层兜底

| 层 | 作用 |
| --- | --- |
| 总开关 + 分层开关 | `enabled=false` 时工具根本不挂 LLM |
| Allowlist / Denylist | app（macOS 走 `NSWorkspace.frontmostApplication`）+ URL domain |
| 高危硬拒绝 | `cmd+shift+q` 注销、`com.apple.keychainaccess` 钥匙串等 |
| 全程审计 | `data/computer_use.jsonl` + 动作前后缩略图 |

**默认没有 confirm 弹窗** —— 靠 allowlist + 日志 + 高危硬拒绝兜底。若要二次确认，可关 `loop_agent_enabled` 让每个工具由 Agent 逐次决定。

### 浏览器双模式

| 模式 | 何时用 | 怎么起 |
| --- | --- | --- |
| `launch` | 通用脚本 / 隔离环境 | 默认，`browser_mode: launch` |
| `cdp` | 要用你**已登录的 Chrome profile**（省重新登录） | 手动 `open -a 'Google Chrome' --args --remote-debugging-port=9222`，yaml 改 `browser_mode: cdp` |

### 审计日志样例

```jsonl
{"ts":"2026-04-20T00:30:12+08:00","tool":"mouse_click","args":{"x":100,"y":200,"button":"'left'","clicks":1},"ok":true,"duration_ms":12.3}
{"ts":"2026-04-20T00:30:13+08:00","tool":"browser_open","args":{"url":"'https://example.com'"},"ok":true,"result":"{'ok': True, 'url': ...}"}
```

缩略图存 `data/computer_use/thumbs/<date>/` PNG。

---

## 🔌 插件生态（OpenClaw 兼容）

JimiAgent 是 OpenClaw 插件生态的**静态宿主**：直接识别社区里的 `openclaw.plugin.json` / `.claude-plugin/plugin.json` / `.cursor-plugin/plugin.json` / `.codex-plugin/plugin.json`，把其中的 skills / slots / commandAliases / channels 合并进本地 agent。**不执行插件里的 TS/JS 代码**，避免把 Node runtime 当作硬依赖。

### 安装 / 管理

```bash
# 本地目录 / 软链（开发用）
python cli.py plugins install ./my-plugin
python cli.py plugins install -l ./my-plugin

# Git（支持 owner/repo 简写）
python cli.py plugins install https://github.com/openclaw/voice-call.git
python cli.py plugins install openclaw/voice-call

# npm（需 node+npm；默认 --ignore-scripts --omit=dev）
python cli.py plugins install @openclaw/voice-call

# 查看 / 诊断
python cli.py plugins list
python cli.py plugins inspect voice-call
python cli.py plugins doctor

# 聊天里也能用
/plugin list
/plugin install ./my-plugin
/plugin show voice-call
```

### 发现路径优先级

1. yaml `plugins.load.paths`
2. `<workspace>/.openclaw/plugins/<id>/`
3. `~/.openclaw/plugins/<id>/`（CLI 默认安装位置）
4. `~/.openclaw/npm_store/node_modules/@openclaw/*`

### SKILL.md frontmatter（OpenClaw 风格）

```yaml
---
name: todoist-cli
description: Manage Todoist tasks
metadata:
  openclaw:
    requires:
      env: [TODOIST_API_KEY]
      bins: [curl]
    primaryEnv: TODOIST_API_KEY
    exec:
      command: bash
      args: ["-c", "echo hi"]
      timeoutSeconds: 10
---
```

支持 `bash / sh / python / node / deno / go / ruby / perl / pwsh` 子进程 skill。安全策略：

- `asyncio.create_subprocess_exec` 绝不走 shell
- 静态 + 运行时二次黑名单扫描
- 默认 10s timeout · 64KB 输出截断

### 插件 slots（替换内置实现）

`plugins.slots.memory`：

- `memory-core`（默认）或空 → 内置 `MemoryStore`
- `none` → 禁用长期记忆
- `<plugin-id>` → 插件 `runtime.py` 暴露 `create_memory_store(settings)` 工厂

同理 `plugins.slots.contextEngine`。任一步失败都优雅回退内置实现。

### REST

```
GET  /api/plugins                    # list
GET  /api/plugins/{id}               # inspect
POST /api/plugins/{id}/enable
POST /api/plugins/{id}/disable
POST /api/plugins/rescan
```

---

## 🧠 长期记忆

三类记忆，独立 SQLite（`data/memory_store.db`）+ FTS5 + 可选向量化：

| 类型 | 用途 | 注入位置 |
| --- | --- | --- |
| `semantic` | 事实（"Jim 住在杭州"） | `<memories>` 块 |
| `episodic` | 事件（"20260419 上传了一张猫的照片"） | `<memories>` 块 |
| `procedural` | 规则（"回答时先分点再总结"） | `<rules>` 块（system prompt 顶部） |

### 抽取模式（`memory.extract_mode`）

- `off` — 不抽取
- `on_compact` — 仅在自动 compact 时抽
- `per_turn` — 每轮后台异步抽（默认），10s 超时硬上限，失败/超时静默

### 时间知识图（Triple Store）

每条 semantic 记忆可拆成 `(subject, predicate, object, valid_from, valid_until)`。写入同 `(s,p)` 新 object 时，旧条目 `valid_until` 推进 → 形成"时间线"。

查询：

```python
store.triples_at("2026-01-01", subject="jim", predicate="work_at")
# → [("jim", "work_at", "ACME", "2025-06-01", "2026-03-01")]
```

CLI 斜杠：`/recall_at 2026-01-01 subject=jim predicate=work_at`

REST：`GET /api/memories/triples?ts=2026-01-01&subject=jim`

### Namespace 隔离

`namespace: auto` 用 `agent_name` 做强隔离 —— 不同 agent 的记忆天然不可见，多 agent 场景不串。

### Evolver 反哺

`procedural` 记忆 `hits≥5` 时，Evolver 会生成 `procedural_usage` signal，提议把高频规则固化到 `workspace/AGENTS.md`。详见 `python cli.py evolver run|status|signals|genes`。

---

## 🛠️ 编写新 Skill

在 `workspace/skills/` 下新建目录 `<your_skill>/`：

**SKILL.md**

```markdown
---
name: your_skill
description: 一句话功能描述（会参与语义召回）
version: "1.0"
dependencies: []
---

# Your Skill

## 使用场景
- 用户问 X 的时候

## 使用方法
调用 `your_skill` 工具，传入 xxx 参数。
```

**`your_skill.py`**（可选）：每个顶层 async/sync 函数会按签名+docstring 注册为一个 Tool：

```python
async def do_something(param: str) -> str:
    """一句话说明（成为 Tool description）"""
    return f"result: {param}"
```

保存后 `POST /api/skills/rebuild` 或 `/restart` 斜杠命令重建索引。

---

## 🌐 API 端点

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| GET | `/api/status` | 系统状态 |
| POST | `/api/chat` | 同步聊天 |
| POST | `/api/events` | SSE 流式聊天 |
| WS | `/ws/chat` | WebSocket 流式聊天 |
| POST | `/api/upload` | 图片上传（可选 `describe=true` 写记忆） |
| GET | `/api/sessions` | 会话列表 |
| POST | `/api/sessions/new` | 新建会话 |
| DELETE | `/api/sessions/{id}` | 删除会话 |
| GET | `/api/skills` | 技能列表 |
| POST | `/api/skills/rebuild` | 重建 Skills 索引 |
| POST | `/api/webhook/{name}` | Webhook 入站 |
| GET | `/api/memories` | 记忆检索 `?q=&k=&kind=` |
| POST | `/api/memories` | 写记忆 |
| DELETE | `/api/memories/{id}` | 软删记忆 |
| GET | `/api/memories/triples` | 时间三元组查询 |
| GET | `/api/scheduler/jobs` | 定时任务列表 |
| POST | `/api/scheduler/jobs/{id}/run` | 立即触发 |
| GET | `/api/plugins` | 插件列表 |
| GET | `/api/plugins/{id}` | 插件详情 |
| POST | `/api/plugins/{id}/enable` | 启用 |
| POST | `/api/plugins/{id}/disable` | 禁用 |
| POST | `/api/plugins/rescan` | 重扫 |
| POST | `/api/evolver/run\|apply\|status\|genes\|signals` | 自进化 |
| GET | `/docs` · `/redoc` | Swagger / ReDoc |
| 静态 | `/uploads/...` | 上传图片 |

---

## 🚢 部署

### 开发

```bash
python cli.py start --verbose       # DEBUG 日志 + reload 关
```

### 生产（裸机 uvicorn + systemd）

**`/etc/systemd/system/jimiagent.service`**：

```ini
[Unit]
Description=JimiAgent Gateway
After=network.target

[Service]
Type=simple
User=jim
WorkingDirectory=/home/jim/JimiAgent
Environment="JIMI_LOG_LEVEL=INFO"
Environment="JIMI_LOG_OUTPUT=both"
ExecStart=/home/jim/miniconda3/envs/jimiAgent312/bin/python main.py
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now jimiagent
sudo systemctl status jimiagent
journalctl -u jimiagent -f
```

### 生产（macOS launchd）

```bash
python cli.py onboard --install-daemon
# 生成 ~/Library/LaunchAgents/com.jimiagent.plist，不自动拉起，只打印手动命令
```

### 反向代理（可选）

放 nginx / caddy 后面挂个 HTTPS。注意 `/ws/chat` 要带 `Upgrade` 头：

```nginx
location /ws/ {
    proxy_pass http://127.0.0.1:18789;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;
}

location / {
    proxy_pass http://127.0.0.1:18789;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
}
```

生产环境另外：

- 把 `cors_origins` 从 `["*"]` 改成白名单
- `channels.webhook.secret` 必设
- `agent.bash.enabled` 若开启务必收紧 `allowlist`

---

## 🧰 CLI 速查

```bash
# 基本
python cli.py start                    # 启动 Gateway
python cli.py chat                     # 进入 TUI
python cli.py chat -s <id>             # 恢复指定会话
python cli.py status                   # 打印系统状态
python cli.py doctor                   # 健康检查

# 配置
python cli.py config path              # 当前 yaml 路径
python cli.py config list              # 可修改字段 + 当前值
python cli.py config get <key>         # 读取 dotted key
python cli.py config set <key> <val>   # 写入（自动识别 bool/int/float/str）
python cli.py config edit              # $EDITOR 打开 yaml

# 会话
python cli.py sessions list
python cli.py sessions delete <id>

# 调度器
python cli.py scheduler list
python cli.py scheduler run <job_id>

# 插件
python cli.py plugins list
python cli.py plugins install <spec>   # 本地路径 / Git URL / owner/repo / npm 包
python cli.py plugins uninstall <id>
python cli.py plugins doctor

# 自进化
python cli.py evolver run [--review]
python cli.py evolver signals
python cli.py evolver genes

# 配对（DM pairing 模式）
python cli.py pairing list
python cli.py pairing approve <code>
python cli.py pairing add <sender_id>

# 迁移 / onboard
python cli.py migrate-env              # 旧 .env → yaml
python cli.py onboard                  # 交互式初始化向导
python cli.py onboard --install-daemon # 生成 launchd / systemd 单元
```

---

## 🩺 故障排查

**Q: 启动时报 `OpenAIEmbedding ... is not a valid OpenAIEmbeddingModelType`**

A: 不是错。OpenAI SDK 的 enum 只认 `text-embedding-3-*` / `ada-002`。配置 `qwen3-embedding-8b` 等 DashScope 自定义名时，`SkillRetriever` 会自动用占位名构造 + patch 运行时真实名。日志降级为 DEBUG 后不再打扰。

**Q: `/api/upload` 报 `python-multipart not installed`**

A: `pip install python-multipart>=0.0.9`（已写入 `requirements.txt`）。

**Q: WebSocket 连不上 / 反代 502**

A: 检查 nginx / caddy 是否转发了 `Upgrade` / `Connection: upgrade` 头；`proxy_read_timeout` 至少 600s。

**Q: Skills 召回没命中**

A: 首先确认 `retrieval_mode` —— 默认 `hybrid`（BM25 + Vector + RRF），对中英混合 / 专有名词 / 短查询最稳：

1. 检查 `data/skill_index/` 目录是否存在（vector 侧产物）
2. `POST /api/skills/rebuild` 或 `/restart` 斜杠命令重建索引
3. 若仍漏掉关键 skill，把 query 里提到的关键词补进 `SKILL.md` 的 `description`（hybrid 的 BM25 侧对 description 加权 2×）
4. 普适工具（datetime / file_ops 这类）可在 SKILL.md frontmatter 加 `always: true` → 不占 top_k 名额总是加载
5. `python cli.py doctor` 看 `llama-index-retrievers-bm25` 是否装上 + embedding 是否可达

**Q: 记忆里有重复条目**

A: 看 `memory.dedup_threshold`（默认 0.92）。数值越大越不容易判重；想更激进去重调到 0.85 左右。

**Q: 对话卡死 / 模型不回应**

A: 1) `python cli.py doctor` 看 API Key 可达；2) `/reload` 重建 LLM；3) 查 `logs/jimi.log` / `journalctl -u jimiagent -f`；4) 配 `model.fallbacks` 做 failover。

**Q: TUI `/` 不弹补全**

A: 终端至少留 8 行可见区域。非常窄的终端会被 `reserve_space_for_menu` 吃掉。或者确认 `prompt_toolkit>=3.0.0`。

**Q: Computer Use 开了但鼠标不动 / 截图是黑的**

A: macOS 权限问题：**系统设置 → 隐私与安全**：
1. **辅助功能** 允许你运行 Python 的终端 / IDE（鼠标键盘必须）
2. **屏幕录制** 允许同一个程序（截图必须，黑图一般是权限缺失）
重启终端后再试。`python cli.py doctor` 可查 pyautogui / mss 版本。

**Q: Playwright 报 `Executable doesn't exist`**

A: 只装了 Python 包，还没装 Chromium。执行 `playwright install chromium`（~200MB）。

**Q: Computer Use 工具被 LLM 无限调用**

A: `loop_agent_enabled` 默认 false；通常 Agent 在普通对话里只会按需用几次。若出现失控：
1. `/computer_use off` 立刻关
2. 检查 `data/computer_use.jsonl` 的动作记录定位问题
3. 把 `max_actions_per_task` / `action_delay_ms` 调大降速
4. 把怀疑被滥用的 app 加到 `app_denylist`

---

## 📜 License

MIT — 详见 `LICENSE`。

---

> 有问题开 issue，PR 欢迎。
