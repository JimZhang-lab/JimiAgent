# Changelog

本项目全部重要改动记录于此。格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，版本号遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/)。

> 约定：Added（新增）· Changed（变更）· Fixed（修复）· Security（安全）· Removed（移除）

---

## [Unreleased]

### M 组 · Computer Use — AI 操控电脑

三层完整的 computer-use 能力栈：桌面原生控制（鼠标/键盘/截图）+ 浏览器自动化（Playwright launch/CDP 双模）+ VLM 任务级 Loop Agent。**默认全关**，启用需显式 yaml 开关或 `/computer_use on` 斜杠命令。

#### Added

- **`ComputerUseConfig`**（`server/config/settings.py`）：总开关 + 分层开关（desktop/browser/loop_agent）+ app/URL allowlist/denylist + 高危快捷键硬拒绝 + 审计路径 + 性能参数 + Playwright launch/cdp 模式切换，共 22 个字段。yaml 段 `agent.computer_use.*` 支持嵌套 `browser.*` / `audit.*` 结构。
- **`server/core/computer_use/` 新包**（~1100 行）：
  - `safety.py` — allowlist/denylist fnmatch 匹配、`check_app()` / `check_url()` / `check_dangerous_keypress()`、macOS 前台 app 识别（`NSWorkspace.frontmostApplication`）、`ComputerUsePermissionError` 异常。
  - `audit.py` — `Auditor` 写 `data/computer_use.jsonl`（时间戳 + 工具名 + args + 结果 + 前后缩略图 ref）+ `save_thumbnail()` PIL 缩略图。
  - `desktop.py` — `DesktopController`：mss 截图 + pyautogui 鼠标键盘。8 个工具 `screen_capture` / `screen_info` / `mouse_move` / `mouse_click` / `mouse_drag` / `mouse_scroll` / `keyboard_type` / `keyboard_press`。lazy import，不启用时完全不加载 pyautogui。
  - `browser.py` — `BrowserController`：Playwright async API，`launch` 模式启动独立 Chromium，`cdp` 模式 `connect_over_cdp()` 接管已开 Chrome。9 个工具 `browser_open` / `click` / `type` / `scroll` / `screenshot` / `extract` / `wait_for` / `press` / `close`。
  - `loop_agent.py` — `ComputerTaskRunner`：VLM 观察-决策-执行闭环。每步截图 → VLM 返回 JSON action → `_dispatch` 分发到 desktop/browser → 再截图 → 直到 `status=done` 或 `max_steps` 超限或 VLM 超时。支持 `done` / `wait` / 所有 desktop+browser 动作。
  - `tools.py` — `build_computer_use_tools(agent)` 按分层开关聚合 18 个 `StructuredTool`（8 desktop + 9 browser + 1 loop），`_safe_call` / `_safe_acall` 统一异常捕获（`ComputerUsePermissionError` → `{"error":"permission_denied"}` 给 LLM）+ 审计写入；`shutdown(agent)` 关 Playwright 实例。
- **`/computer_use` 斜杠命令**（`server/core/commands.py`）：`status` 查看状态；`on|off` 切总开关；`<desktop|browser|loop> <on|off>` 切分层；全部持久化到 yaml + 清 Agent cache。
- **Agent 接线**（`server/core/agent.py`）：`_builtin_tools` 新增 `*build_computer_use_tools(self)` 一行，按 `computer_use.enabled` 动态挂载。
- **Gateway shutdown hook**（`server/api/gateway.py`）：lifespan 结束时遍历所有 agent 调 `cu_shutdown` 关 Playwright。
- **TUI 补全**：`SLASH_COMMANDS` 加 `/computer_use` + `display_meta`；`/help` 表同步。
- **MUTABLE_KEYS**（`server/config/config_io.py`）：加 `agent.computer_use.enabled` / `desktop_enabled` / `browser_enabled` / `loop_agent_enabled` / `browser_mode` / `browser_headless` / `audit_thumbnails` 7 条，供 `/config` 和 `/computer_use` 写 yaml。
- **依赖**（`requirements.txt`）：`pyautogui>=0.9.54` / `mss>=9.0.1` / `playwright>=1.42.0`。首次需 `playwright install chromium` + macOS 授权辅助功能 / 屏幕录制。
- **cli.py doctor**：新增 pyautogui / mss / playwright 版本检查。
- **env 覆盖**：`JIMI_COMPUTER_USE_ENABLED` / `JIMI_BROWSER_MODE` / `JIMI_BROWSER_HEADLESS`。

#### Security

四层兜底（基于用户选择，**不带 confirm 层**；靠 allowlist + 日志 + 高危硬拒绝）：

1. **总开关 + 分层开关** — `enabled=false` 时 `build_computer_use_tools` 返回空列表，工具根本不挂 LLM
2. **Allowlist 前置检查** — 每个写类工具（click/type/scroll/press）先 `safety.check_app()` / `check_url()`，未通过抛 `ComputerUsePermissionError`
3. **高危硬拒绝** — `dangerous_keypress_deny` 归一化后硬拒（`shift+cmd+q` == `cmd+shift+q`）；`app_denylist` 默认含 Terminal / Keychain / 系统偏好设置；`domain_denylist` 默认含 `*bank*` / `*.alipay.com` / `*.icloud.com`
4. **全程审计** — 每次动作前后缩略图 + jsonl 记录到 `data/computer_use.jsonl`，`thumbnails: true` 时缩略图存 `data/computer_use/thumbs/<date>/`

#### 刻意不做（Non-Goals）

- 不做 confirm 弹窗模式（用户明确不要，靠 allowlist + 日志兜底）
- 不做 OCR（VLM 直读屏幕文字已够用）
- 不做 Windows 权限指引（macOS/Linux 优先）
- 不做多屏独立坐标系（先支持 primary display）
- 不做支付/银行 domain 操控（硬拒绝）
- 不自动导入 Chrome profile cookies（cdp 模式让用户自己启 `--remote-debugging-port=9222`）

#### 回归

17/17 通过（safety 分层开关 + allowlist/denylist + 归一化高危键 8 条 · desktop 截图/点击/快捷键/URL 4 条 · browser launch/cdp lazy init + extract 3 条 · loop done 退出 + max_steps 超限 2 条）。

---

### N 组 · Skills Hybrid 检索升级

在纯 `VectorStoreIndex` 召回基础上叠加 BM25 + RRF 融合，并带上精确名短路 / TTL LRU 缓存 / `always=true` 常驻 / name+description 加权四项低成本优化。总开销变化 < 1%，中英混合 / 专有名词 / 短查询精度显著提升；30s 内重复 query 省掉 embedding API 调用（实测 30 倍+ 加速）。

#### Added

- **Hybrid 检索主路径**（`server/core/skill_retriever.py`）：`BM25Retriever + QueryFusionRetriever(mode=RECIPROCAL_RANK)`；`num_queries=1` 不做 LLM query expansion；两路各取 `top_k × 2` 候选给融合层，最终截 `top_k`。
- **CJK 预分词方案**：因 `llama-index-retrievers-bm25>=0.7` 把 `tokenizer=` 参数降为 no-op，改走"索引端预分词（CJK 单字空格隔开）+ `token_pattern=r"(?u)\S+"` + 子类 `_CJKBM25Retriever` 在 query 端同样预分词"。中文 `读取文件` 类查询正确命中 `file_ops`。
- **精确名短路**：`_exact_name_matches(query)` 检查 query 是否含完整 `skill_name` 或 `skill_key`（`len≥3`，避免单字误匹配）；命中则前置加入结果，保证 TUI 里直接称呼 skill 名秒返。
- **TTL LRU 缓存**：`_TTLCache(maxsize=128, ttl=30s)`，缓存的是 `list[NodeWithScore]`；一次命中同时省 embedding + BM25 + 融合三步。
- **`always: true` 常驻**：`SkillMeta.always` 字段（K1 已定义但未使用）现在真正生效 —— 这类 skill 不占 `top_k` 名额，永远被加载，适合 `datetime_info` / `file_ops` 这类普适工具。
- **BM25 侧加权**：`bm25_weight_name=3` / `bm25_weight_description=2` 通过分词后 token 序列重复实现 tf 放大；name 命中的权重远高于 instructions。Vector 侧不加权（语义向量靠自身处理）。
- **降级链**：`hybrid → vector_only → bm25_only → 全量加载`；`llama-index-retrievers-bm25` 未装 / embedding 不可达 / 两路同时挂都能优雅降级。
- **配置**（`SkillsConfig` 6 个新字段）：`retrieval_mode` / `query_cache_ttl_seconds` / `query_cache_size` / `exact_match_short_circuit` / `bm25_weight_name` / `bm25_weight_description`。

#### Changed

- `retrieve_skills()` 重构为 5-step 流程：`always → exact match → cache → hybrid retrieve → 阈值过滤+去重`。
- `similarity_threshold` 只在 `vector_only` 模式下生效（RRF score 在 [0, ~0.03] 区间，与原阈值语义不兼容）。
- 保底规则"至少返回最相关一条"改为"非-always 匹配为 0 时才触发"（`always` skill 不算作 retrieval 命中）。
- `rebuild_index()` 同步清空 query 缓存。
- `requirements.txt` 新增 `llama-index-retrievers-bm25>=0.5.0`。
- `cli.py doctor` 新增该依赖版本检查。

#### Performance

| 场景 | 原 | 新 |
| --- | --- | --- |
| 首次查询 | 200-500ms | +1-3ms（`<1%`） |
| 30s 内重复 | 200-500ms | ~0.01ms（`30×+` 加速） |
| `grep 一下日志` | 向量语义模糊命中 `web_search` | BM25 通过英文 "grep" 稳定命中 `shell_exec` |
| `读取文件 /tmp/foo.txt` | 向量返 `web_search` | hybrid 返 `file_ops` |

#### 回归

25/25 通过（CJK tokenizer 4 · TTLCache 4 · 精确匹配 4 · cache 开关 · BM25 加权 3 · `always` 永驻 2 · 缓存零重入 2 · 阈值 2 · 降级 1 · rebuild 清理 2）。

### Changed

- **TUI 斜杠补全**：换成带 `display_meta` 描述的自定义 `SlashCommandCompleter`（原 `WordCompleter` 在 `/` 非 word 字符前缀下触发不稳定）。打 `/` 即弹出候选菜单，每条带简介；菜单样式与项目深色主题一致。
- **日志降噪**：`OpenAIEmbedding` 对非标准 embedding 名走"占位名 + patch 真实名"兼容路径本就是预期行为，之前用 `logger.info` 提示让人误以为配错 —— 降级为 `logger.debug`，措辞也改成"不在 OpenAI 官方 enum 中，走兼容模式"。

### Fixed

- TUI 某些终端下打 `/` 不弹补全（根因：`WordCompleter` 默认 `WORD=False`，`/` 被当非 word 字符返回空前缀，不同终端触发时机不同）。

---

## [0.1.0] — 2026-04-19

### L 组 · 多模态 VLM + VL-Embedding + 图片记忆

独立的 VLM / VL-Embedding 配置，未配置即回退到主 LLM / 文本 Embedding。聊天可带图、上传端点产公开 URL、图片可选描述并向量化写入 MemoryStore、Web UI 支持粘贴 / 拖拽 / 文件选择。

#### Added

- **`VLMConfig` / `VLEmbeddingConfig`**（`server/config/settings.py`）：`Settings.vlm` / `Settings.vl_embedding` + `effective_vlm` / `effective_vl_embedding` 回退 property；yaml 段 `agent.vlm` / `agent.vl_embedding`；env 覆盖 `JIMI_VLM_{API_KEY,BASE_URL,MODEL}` / `JIMI_VL_EMBEDDING_{API_KEY,BASE_URL,MODEL}`。
- **Agent 双模型**（`server/core/agent.py`）：启动时实例化 `self.vlm`；未显式配置时 `_create_vlm` 直接返回 `self.llm`（同引用，零开销）。`_build_agent(..., use_vlm=bool)` cache key 加 `llm|vlm` 维度。
- **图文消息**（`server/api/models.py`）：`ChatRequest.images: list[str]`（http(s) / data URI）。`_build_input_messages` 在有图时把 `HumanMessage.content` 切成多部分列表 `[{type:text}, {type:image_url}]`。`chat` / `chat_stream` 两条通路都穿传 `images`，REST / SSE / WS 全通。
- **`POST /api/upload`**（`server/api/routes.py`）：multipart 上传、SHA-256 短哈希命名、按日期目录 `data/uploads/<yyyymmdd>/<hash>-<name>`、10MB 上限、mime/ext 双白名单。可选 `describe=true` → VLM 生成描述 → VL-embedding → 写 `MemoryStore`（失败不影响上传）。Gateway 挂载 `/uploads` 静态目录。
- **多模态记忆通路**（`server/core/multimodal.py` + `memory_store.py`）：`describe_image()` / `describe_and_remember()` 三步走。`MemoryStore.add()` 新增 `embedding=` 参数，外部已算好 vector 直接写，跳过内部 embed。
- **Web UI 拖拽 / 粘贴上传**（`server/static/*`）：附件栏 + 附加按钮；`pendingImages` 状态；上传 + 缩略图 + 移除；`sendMessage` 带 `images[]`；消息气泡渲染图片；点击 / 粘贴 / 拖拽入口；拖拽高亮 + pending spinner 样式。

#### Changed

- `requirements.txt` 新增 `python-multipart>=0.0.9`（FastAPI 解析 multipart 上传必需）。

#### 未做

- 视频 / 音频多模态（需 provider realtime 支持）。
- 上传目录 GC 清理（用户自行 `rm -rf data/uploads`）。
- 前端 base64 直传（走服务端上传 + 公开 URL 对所有主流 VLM provider 更通用）。

#### 回归

34/34 通过（L1 配置默认 + 回退 + yaml + env；L3 多部分 content；L4 TestClient 验证 200/415/400；L5 外部 embedding 保真写入）。

---

### K 组 · OpenClaw 插件生态兼容层

把 JimiAgent 升级为 OpenClaw 插件生态的**静态宿主**：解析 `openclaw.plugin.json` / `.claude-plugin` / `.cursor-plugin` / `.codex-plugin`；SKILL.md frontmatter 对齐 `metadata.openclaw.*`；exec skill runtime；插件 slots 可替换内置实现。**绝不执行插件里的 TS/JS 代码**，避免引入 Node runtime 依赖。

#### Added

- **SKILL.md frontmatter 升级**（`server/core/skill_loader.py`）：扩展 `SkillMeta` 含 `requires_env` / `bins` / `any_bins` / `config`、`primary_env`、`always`、`skill_key`、`emoji`、`homepage`、`os_list`、`install_specs`；解析 `metadata.openclaw.*`（别名 `clawdbot` / `clawdis`）；`diagnose_skill()` 检查缺环境变量 / 缺 bin / OS 不匹配。老 SKILL.md 完全向后兼容。
- **exec skill runtime**（`server/core/skill_exec.py`）：支持 bash / sh / python / node / deno / go / ruby / perl / pwsh 子进程；`asyncio.create_subprocess_exec` 绝不走 shell；静态 args 危险模式扫描 + 运行时二次扫描；64KB 输出截断；默认 10s timeout。`build_tool_from_skill` 优先走 exec，失败回退脚本 / 指令。
- **插件 manifest 静态解析**（`server/core/plugin_manifest.py`）：解析 `openclaw.plugin.json`（id / kind / skills / providers / channels / commandAliases / activation / configSchema / uiHints / providerAuthEnvVars / legacyPluginIds）；`.claude-plugin` / `.cursor-plugin` / `.codex-plugin` 作为 bundle 备源自动识别；`validate_manifest` 非阻断诊断。
- **插件注册表 + CLI / REST / 斜杠**（`server/core/plugin_registry.py`）：四层发现路径（yaml `load.paths` → `<ws>/.openclaw/plugins` → `~/.openclaw/plugins` → `npm_registry/node_modules/@openclaw`）；`allow` / `deny` / `entries` 三路开关；`/plugin list|show|enable|disable|install|uninstall|doctor`；`python cli.py plugins …` 同名子命令；REST 全套（`GET /api/plugins`、`GET /api/plugins/{id}`、`POST /api/plugins/{id}/(enable|disable)`、`POST /api/plugins/rescan`）。
- **插件 slots（memory / contextEngine）**（`server/core/ports/` + `plugin_slots.py`）：`MemoryPort` / `ContextEnginePort` Protocol；从插件 `runtime.py` 动态导入 `create_memory_store(settings)` / `create_context_engine(settings)` 工厂；任一步失败自动回退内置实现；`slots.memory="none"` 完全禁用长期记忆。
- **Bundle 目录兼容**：`_BUNDLE_CANDIDATES` 覆盖 4 种 manifest 位置；没写 `skills` 字段时约定优于配置（默认取 `./skills`）。
- **插件安装器**（`server/core/plugin_installer.py`）：统一 `install(spec)` 路由三类 —— 本地路径（copy / symlink）、Git（`git clone --depth 1`，支持 `owner/repo` 简写）、npm 包（`--ignore-scripts --omit=dev`；缺 node 明确报错）。失败清理半成品；`uninstall` 支持 `keep_files`。
- **`PluginsConfig` + yaml**（`server/config/settings.py`）：`plugins.enabled` / `allow` / `deny` / `load.paths` / `slots` / `entries` / `install_registry` / `npm_registry` 全套字段。
- **Agent 接线**（`server/core/agent.py`）：启动时扫描 `PluginRegistry`，把插件贡献的 skill 目录合并进 `SkillRetriever.extra_skill_dirs`；按 `plugins.slots.memory` 选择 MemoryStore 实现。

#### 未做

- 不执行 manifest 里的 TS/JS `register(api)` 代码（不引 jiti / Node runtime）。
- 不自动化 `install specs`（brew / uv / node / go），仅诊断提示。
- 不做 ClawHub 服务端协议 / marketplace.json。
- 不做 LSP server 注入。

#### 回归

47/47 通过（frontmatter 扩展 + 向后兼容、exec runtime 正路径 + 危险拒绝、manifest 三种 bundle + 坏 JSON、registry allow/deny、skill 目录合并、slots 四种分支、installer 本地安装 / 覆盖 / 卸载 / 无 manifest 拒绝）。

---

### J 组 · 记忆系统 v2

在 I 组基础上补齐 4 项扩展，`per_turn 默认 + namespace=agent_name` 策略。

#### Added

- **Triple Store**：`memories` 表加 `predicate` / `object` / `valid_from` / `valid_until` 四列 + 自动 ALTER 迁移；`add_triple(s, p, o)` 支持时间覆盖语义（同 `(S,P)` 写入新 object 会把旧条目 `valid_until` 推进至新 `valid_from`）。
- **时间查询**：`triples_at(ts, subject?, predicate?)` API + `recall_at` LLM 工具 + `/recall_at` 斜杠命令（支持 `YYYY-MM-DD` 短式）+ `GET|POST /api/memories/triples` REST。
- **Hot-path per-turn 抽取**：`extract_mode: "per_turn"` 默认开启；`chat` / `chat_stream` 末尾 `asyncio.create_task(_hot_extract(...))` 后台触发；`_extract_locks` per-session 锁避免重叠；10s 超时硬上限；失败 / 超时静默。
- **Procedural → system prompt 顶部**：`build_rules_block()` 把 procedural 记忆拼成独立 `<rules>...</rules>` 块，注入在 `<memories>` 前；`recall_max_inject` 与 `procedural_inject_max` 分开控制。
- **Evolver 反哺**：`EvolutionEngine` 接受 `memory_store` 注入，`scan_signals` 把 `hits≥5` 的 procedural 记忆生成 `procedural_usage` signal；新增 `gene.procedural_promotion` 默认种子（提议把高频规则固化到 `workspace/AGENTS.md`）；`_ensure_seed_assets` 对已存在 `genes.json` 做增量合并。
- **Namespace 隔离**：yaml `namespace: "auto"` 自动使用 `agent_name`；所有 CRUD / 搜索 / triple 查询都加 `namespace=?` 过滤；`MemoryStore(..., namespace_override=)` 支持显式穿透；v1 旧数据默认归入 `'default'`。

#### Changed

- `_create_schema` 通过 `PRAGMA table_info` 检测 + `ALTER TABLE ADD COLUMN`，支持 v1 → v2 无痛升级。
- `scan_signals` 重构：三源（tool_log / events / memory_store）解耦，任一源缺失不影响其他源。

#### 回归

46/46 通过（migration、triple 时间覆盖、namespace 隔离、rules 块、hot-path 去重、triple 抽取、evolver procedural_usage 端到端）。

---

### I 组 · 跨会话长期记忆

调研对齐：LangMem（三分法）/ Mem0（去重）/ Letta（core-block）/ Graphiti（留扩展点给 v2）。

#### Added

- **MemoryStore 核心**（`server/core/memory_store.py`）：独立 SQLite + FTS5 + 可选向量化。
  - 3 类记忆：`semantic`（事实）/ `episodic`（事件）/ `procedural`（规则）。
  - Embedding 自动复用 `agent.embedding` 配置；失败降级为 FTS5。
  - **CJK 兜底**：FTS5 `unicode61` 对中文分不出词时自动切 `LIKE %token%` 子串搜索。
  - Cosine 去重：相似度 > `dedup_threshold`（默认 0.92）的新事实仅 touch 不新插。
  - 软删除 + `hits` 计数 + `kind` 过滤。
- **Background 抽取**：`compact_session` 成功后自动调 `extract_and_store(llm, messages)`，LLM 结构化 JSON 抽事实入库（失败不影响 compact）。
- **Hot-path 注入**：`chat` / `chat_stream` 前 `build_recall_block(user_message, k=5)` → `SystemMessage <memories>...</memories>` 注入。
- **3 个内置工具**：`remember(text, kind)` / `recall(query, k)` / `forget(memory_id)`，让 LLM 自己决定记 / 查 / 删。
- **4 个斜杠命令**：`/memories [kind=... | clear yes]` / `/recall <query>` / `/forget <id>` / `/memory` 别名。
- **REST 端点**：`GET /api/memories?q=&k=&kind=` / `POST /api/memories` / `DELETE /api/memories/{id}`。
- **配置开关**：yaml `agent.memory.{enabled,extract_on_compact,recall_max_inject,dedup_threshold,embedding_model}`；`enabled=false` 时完全回退到旧行为。

#### 回归

35/35 通过（cosine、JSON 解析、FTS / 向量 / LIKE 三路径、去重、软删、抽取 + 容错）。

---

### H 轮 · 安全 / 健壮性加固

#### Security

- **bash 注入防护加深**：`_DANGEROUS_PATTERNS` 新增 `;` / `&&` / `||` / `|` / 反引号 / `$(` / `>` / `<` / `&` 等 shell 组合字符。即便 allowlist 首 token 命中，也无法用这些拼接绕过（修 F3 allowlist 只校验第一个 token 的注入空位）。

#### Fixed

- `sessions_history` / `send` / `spawn` 事件循环兼容：抽出 `_run_coro` 辅助，同时处理"threadpool worker 线程（无 loop）"和"主 loop 线程（有 loop）"两种场景，避免 `RuntimeError: loop already running`。

---

### G 组 · Skill 自我修正与进化（EvoMap 适配）

#### Added

- **Evolver 引擎**（`server/core/evolver.py`）：GEP 协议 Python 精简实现。
  - `Gene` / `Capsule` / `EvolutionEvent` / `PersonalityState` 完整数据模型。
  - 4 个种子 Gene：`skill_failure_repair` / `missing_skill_innovate` / `stagnation_breaker` / `error_pattern_generic`。
  - 4 种 strategy：`balanced` / `innovate` / `harden` / `repair-only`。
  - `Signal.fingerprint()` 24h 级去重；stagnation 自动升级为 harden。
  - 单轮最多 5 个 event，避免淹没。
  - **只生成 prompt 不改代码**：坚持 EvoMap 中立定位；`apply` 最激进动作是派生新 session 让 Agent 走 bash / `edit_file` 白名单尝试修复。
- **Tool-call 日志**：`on_tool_start/end/error` 追加写 `data/tool_log.jsonl`，作为 signal 主来源。
- **CLI 子命令**：`python cli.py evolver run|status|signals|genes`；`--review` 人机交互 `y` / `n` / `a` / `q` 逐条确认应用。
- **REST 端点**：`/api/evolver/run|apply|status|genes|signals`。
- **Scheduler 打通**：cron job 新增 `kind: evolver` 字段，周期性跑 `evolver.run_cycle()` 不走 LLM。
- **审计轨迹**：所有 event 持久化到 `data/gep/events.jsonl`，重启不丢。

---

### F 组 · OpenClaw 深度对齐

#### Added

- `sessions_list` 工具别名：OpenClaw 命名兼容。
- `doctor` 新增 DM 安全策略审计：自动识别 `dmPolicy=open + secret 空 + allowFrom 含 *` 等风险组合。
- `edit_file` 内置工具：要求 `old_text` 唯一出现，避免误改多处。
- `bash` / `process_list` 内置工具（默认关闭）：四层防护 = allowlist + 危险模式黑名单 + 二次确认 + cwd/timeout/截断；`/bash_confirm on|off` 切换确认。
- `onboard --install-daemon`：macOS 生成 launchd plist，Linux 生成 systemd user service，不自动拉起只打印手动命令；`--uninstall-daemon` 打印清除命令。
- `PairingStore` 支持 yaml `allowFrom` 合并 + fnmatch 通配符（`admin_*` 等）。

---

### E 组 · OpenClaw 对齐

#### Added

- `/trace on|off` 斜杠命令：透出 LangGraph 内部事件，供调试。
- `/activation mention|always`：mention 模式下消息必须含 `@agent_name` 才处理。
- **DM Pairing / Allowlist**：webhook 入站安全策略（`open` / `pairing` / `closed`），配对码 + CLI `pairing list|approve|revoke|add`。
- **Session 工具扩展**：`sessions_history`（跨会话读消息）/ `sessions_send`（跨会话推送）/ `sessions_spawn`（派生子任务）。
- **文件工具**：`read_file` / `write_file` / `list_dir`，严格路径白名单，防穿越。
- **Multi-Agent Router**：`AgentRegistry` + `RoutingRule`，yaml `agents` 段声明多 agent，按 channel / sender 路由。
- **CLI `message send|stream`**：不经过 Gateway 直接本地调用 Agent。
- **CLI `onboard`**：交互式配置向导 —— 首次设置 API Key / 模型 / Workspace。

---

### 基础设施（0.1.0 首发共同完成）

#### 核心架构

- LangGraph `create_react_agent` 驱动的 ReAct Agent。
- `AsyncSqliteSaver` 持久化记忆，每会话独立 `thread_id`。
- LlamaIndex Skills 语义召回，Top-K 动态挂载工具。
- Workspace 四层 prompt（`SOUL` / `AGENTS` / `TOOLS` / `USER`）。
- 斜杠命令分发器 `CommandDispatcher`。
- 自动 compact：阈值触发 LLM 摘要 + 保留最近 N 条。

#### 前端

- FastAPI Gateway：REST + WebSocket 流式 + SSE + Webhook。
- 深色 Web UI（`server/static/`）。
- 终端 TUI：`prompt_toolkit` 斜杠补全 / 历史 / 多行 + `rich.live` 流式 Markdown 面板 + 工具调用面板 + 紧凑状态行。

#### 功能

- Session 内置工具：`list_recent_sessions` / `get_session_info` / `create_new_session`。
- 模型 Failover：基于 `Runnable.with_fallbacks` 的主 → 备选链。
- Webhook 通用入站：`POST /api/webhook/{name}`，`hmac.compare_digest` 校验密钥，`sender_id` 稳定哈希映射会话。
- SSE 流式聊天：`POST /api/events`，含客户端断连检测。
- Cron 定时任务：`server/core/scheduler.py`，REST `/api/scheduler/jobs`，CLI `scheduler list|run`。
- 日志系统：`console` / `file` / `both` 三种输出，`RotatingFileHandler` 轮转，rich / plain 格式切换。
- 终端可访问 / 配置方案：CLI `config get|set|list|path|edit` + TUI `/config` / `/model` / `/reload` / `/edit`，`ruamel.yaml` roundtrip 保注释。

#### Fixed

- `compact_session` 的 `aupdate_state` Ambiguous（`as_node="__start__"`）。
- `compact` 后消息顺序错（全删后按 `[summary, *kept]` 重新写入）。
- `OpenAIEmbedding` 对非标准模型名的 enum 校验（占位名构造 + patch `_query_engine` / `_text_engine`）。
- `SessionManager.touch` + `increment_message_count` 对话后自动递增。
- `SessionManager.create_session_with_id` 幂等 API 替代路由里对 `_sessions` 的脏改。

#### 工程

- 24 个 `.py` 文件统一头注。
- 全项目去 emoji + 去 AI 味文案。
- `.gitignore` 屏蔽 `data/` / `*.db*` / `config/agent_config.yaml` / `workspace/USER.md` / `.DS_Store`。
- `config/agent_config.example.yaml` 作为公开模板。
- 健康检查 `python cli.py doctor` 加 `prompt_toolkit` / `croniter` / `ruamel.yaml` 版本 + 日志配置检查。

---

[Unreleased]: https://github.com/JimZhang-lab/JimiAgent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JimZhang-lab/JimiAgent/releases/tag/v0.1.0
