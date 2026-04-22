'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/server/config/settings.py
Description: 配置加载。

'''
import os
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Optional

import yaml


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class ModelConfig:
    provider: str = "openai"
    model_id: str = "qwen3-32B"
    temperature: float = 0.7
    max_tokens: int = 4096
    streaming: bool = True
    api_key: str = ""
    base_url: str = ""
    # 备选模型
    fallbacks: list = field(default_factory=list)


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-ada-002"
    api_key: str = ""
    base_url: str = ""


@dataclass
class VLMConfig:
    """视觉 LLM（Vision-Language Model）配置"""
    provider: str = ""
    model_id: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096
    streaming: bool = True
    api_key: str = ""
    base_url: str = ""


@dataclass
class VLEmbeddingConfig:
    """视觉-语言 Embedding 配置"""
    model: str = ""
    api_key: str = ""
    base_url: str = ""


@dataclass
class SkillsConfig:
    """Skills 召回配置"""
    top_k: int = 3
    similarity_threshold: float = 0.5
    index_persist_dir: str = "./data/skill_index"

    # Hybrid 检索
    retrieval_mode: str = "hybrid"           # hybrid / vector_only / bm25_only
    query_cache_ttl_seconds: int = 30
    query_cache_size: int = 128
    exact_match_short_circuit: bool = True
    bm25_weight_name: int = 3
    bm25_weight_description: int = 2


@dataclass
class MemoryConfig:
    """记忆配置"""
    backend: str = "sqlite"
    sqlite_path: str = "./data/memory.db"
    enabled: bool = True
    store_path: str = "./data/memory_store.db"
    extract_on_compact: bool = True
    recall_max_inject: int = 5
    dedup_threshold: float = 0.92
    embedding_model: str = "auto"  # auto / off / <custom name>（预留）

    # J 组扩展
    # off / on_compact / per_turn
    extract_mode: str = "per_turn"
    # per_turn 单次超时
    extract_timeout_seconds: int = 10
    # auto 时沿用 agent_name
    namespace: str = "auto"
    # procedural 注入上限
    procedural_inject_max: int = 10


@dataclass
class SessionConfig:
    max_history_messages: int = 50
    compact_threshold: int = 40
    default_think_level: str = "medium"  # low / medium / high


@dataclass
class GatewayConfig:
    host: str = "0.0.0.0"
    port: int = 18789
    cors_origins: list = field(default_factory=lambda: ["*"])


@dataclass
class LoggingConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "rich"
    output: str = "console"
    file_path: str = "./data/logs/jimi.log"
    rotate_mb: int = 10
    backup_count: int = 5


@dataclass
class SafetyConfig:
    """全盘文件访问 + 命令执行的安全层配置。

    - `enabled`: 关则退化为 workspace-only（兼容旧行为）
    - `extra_path_deny` / `extra_cmd_deny`: 用户扩展黑名单（fnmatch / regex）
    - `extra_safe_cmd_heads`: 用户扩展的 safe 命令头（如 `kubectl`）
    - `confirm_mode`:
        * "ui"      — dangerous 操作走 LangGraph interrupt，等前端/TUI confirm
        * "llm"     — 返回 [PENDING CONFIRM]，让 LLM 询问用户后带 confirm=true 再调
        * "auto"    — 优先 ui；runtime 无 interrupt 支持时退化为 llm
        * "off"     — 全部 allow（不推荐，等价放开所有）
    - `auto_approve_workspace_writes`: workspace 内写默认视为 safe（无需 confirm）
    """
    enabled: bool = True
    extra_path_deny: list = field(default_factory=list)
    extra_cmd_deny: list = field(default_factory=list)
    extra_safe_cmd_heads: list = field(default_factory=list)
    confirm_mode: str = "auto"
    auto_approve_workspace_writes: bool = True
    # 等 confirm 的秒数，超时默认拒绝
    confirm_timeout_seconds: int = 120


@dataclass
class BashConfig:
    """bash 工具配置。"""
    enabled: bool = False
    allowlist: list = field(default_factory=lambda: [
        "git", "ls", "cat", "pwd", "python", "pip",
        "node", "npm", "echo", "head", "tail", "wc",
    ])
    # 二次确认前允许的只读前缀
    read_only_prefixes: list = field(default_factory=lambda: [
        "ls", "cat", "pwd", "echo", "head", "tail", "wc",
        "git status", "git log", "git diff", "git branch",
        "pip list", "pip show", "npm list",
    ])
    timeout_seconds: int = 30
    max_output_bytes: int = 4096


@dataclass
class ChannelsConfig:
    """渠道配置。"""
    raw: dict = field(default_factory=dict)

    def is_enabled(self, name: str) -> bool:
        cfg = self.raw.get(name) or {}
        return bool(cfg.get("enabled", False))

    def get(self, name: str) -> dict:
        return self.raw.get(name) or {}


@dataclass
class SchedulerConfig:
    """定时任务配置"""
    enabled: bool = False
    tick_seconds: int = 30
    jobs: list = field(default_factory=list)


@dataclass
class AgentsConfig:
    """多 Agent 路由配置。"""
    default_agent: str = "main"
    agent_list: list = field(default_factory=lambda: [])  # [{name, workspace, model: {...}}]
    routing: list = field(default_factory=lambda: [])  # [{match: {...}, agent: str}]


@dataclass
class ComputerUseConfig:
    """Computer Use 配置。"""
    # 开关
    enabled: bool = False                          # 总开关
    desktop_enabled: bool = True                   # 鼠标/键盘/截图分层
    browser_enabled: bool = True                   # Playwright
    loop_agent_enabled: bool = False               # VLM 任务级循环（最敏感）

    # Allowlist / Denylist
    # 空表示全开；denylist 优先
    app_allowlist: list = field(default_factory=list)
    app_denylist: list = field(default_factory=lambda: [
        "com.apple.keychainaccess",
        "com.apple.systempreferences",
        "com.apple.Terminal",                      # 防 loop agent 自杀
    ])
    domain_allowlist: list = field(default_factory=list)
    domain_denylist: list = field(default_factory=lambda: [
        "*bank*", "*.alipay.com", "*.icloud.com",
    ])

    # 高危硬拒绝
    dangerous_keypress_deny: list = field(default_factory=lambda: [
        "cmd+shift+q",                             # 注销
        "ctrl+alt+delete",
    ])

    # 审计
    audit_log_path: str = "./data/computer_use.jsonl"
    audit_thumbnails: bool = True
    audit_thumbnail_dir: str = "./data/computer_use/thumbs"

    # 性能
    screenshot_max_width: int = 1920               # 给 VLM 前缩放宽度
    action_delay_ms: int = 100                     # 每动作间 sleep
    max_actions_per_task: int = 50                 # loop agent 最大步数
    loop_step_timeout_seconds: int = 30

    # Browser
    browser_mode: str = "launch"                   # launch / cdp
    browser_headless: bool = False
    browser_cdp_url: str = "http://localhost:9222"
    browser_user_data_dir: str = ""
    browser_downloads_dir: str = "./data/downloads"

    # 前台 app 识别失败时的策略
    safety_fail_policy: str = "deny"               # deny / allow


@dataclass
class PluginsConfig:
    """OpenClaw 插件生态兼容层配置"""
    enabled: bool = True
    allow: list = field(default_factory=list)
    deny: list = field(default_factory=list)
    load_paths: list = field(default_factory=list)
    slots: dict = field(default_factory=lambda: {
        "memory": "memory-core",
        "contextEngine": "none",
    })
    entries: dict = field(default_factory=dict)
    install_registry: str = "~/.openclaw/plugins"
    # npm 插件目录
    npm_registry: str = "~/.openclaw/npm_store"


@dataclass
class Settings:
    """全局配置单例"""
    agent_name: str = "jimi"  # agent 标识，用于 /activation mention 模式匹配 @agent_name
    model: ModelConfig = field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    vl_embedding: VLEmbeddingConfig = field(default_factory=VLEmbeddingConfig)
    workspace_path: str = "./workspace"
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    session: SessionConfig = field(default_factory=SessionConfig)
    gateway: GatewayConfig = field(default_factory=GatewayConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    channels: ChannelsConfig = field(default_factory=ChannelsConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    agents: AgentsConfig = field(default_factory=AgentsConfig)
    bash: BashConfig = field(default_factory=BashConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    plugins: PluginsConfig = field(default_factory=PluginsConfig)
    computer_use: ComputerUseConfig = field(default_factory=ComputerUseConfig)

    # 兼容旧 API
    @property
    def api_key(self) -> str:
        return self.model.api_key

    @property
    def base_url(self) -> str:
        return self.model.base_url

    # VLM / VL-Embedding 有效配置

    @property
    def effective_vlm(self) -> VLMConfig:
        """返回补全后的 VLMConfig。"""
        m = self.model
        v = self.vlm
        return VLMConfig(
            provider=v.provider or m.provider,
            model_id=v.model_id or m.model_id,
            temperature=v.temperature if v.model_id else m.temperature,
            max_tokens=v.max_tokens if v.model_id else m.max_tokens,
            streaming=v.streaming,
            api_key=v.api_key or m.api_key,
            base_url=v.base_url or m.base_url,
        )

    @property
    def effective_vl_embedding(self) -> EmbeddingConfig:
        """返回补全后的 EmbeddingConfig。"""
        e = self.embedding
        ve = self.vl_embedding
        return EmbeddingConfig(
            model=ve.model or e.model,
            api_key=ve.api_key or e.api_key,
            base_url=ve.base_url or e.base_url,
        )

    @property
    def workspace_abs_path(self) -> Path:
        p = Path(self.workspace_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()

    @property
    def memory_abs_path(self) -> Path:
        p = Path(self.memory.sqlite_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()

    @property
    def skill_index_abs_path(self) -> Path:
        p = Path(self.skills.index_persist_dir)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()

    @property
    def log_file_abs_path(self) -> Path:
        """日志文件绝对路径（若 file_path 为相对路径则相对 PROJECT_ROOT）"""
        p = Path(self.logging.file_path)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.resolve()


def _parse_config(raw: dict) -> Settings:
    """从原始 YAML dict 解析为 Settings 对象"""
    agent_cfg = raw.get("agent") or {}
    gateway_cfg = raw.get("gateway") or {}
    logging_cfg = raw.get("logging") or {}
    channels_cfg = raw.get("channels") or {}
    scheduler_cfg = raw.get("scheduler") or {}

    # model
    model_raw = dict(agent_cfg.get("model") or {})
    fallbacks = model_raw.pop("fallbacks", []) or []
    model = ModelConfig(**model_raw, fallbacks=fallbacks)

    # embedding
    embedding = EmbeddingConfig(**(agent_cfg.get("embedding") or {}))

    # vlm / vl_embedding
    vlm = VLMConfig(**(agent_cfg.get("vlm") or {}))
    vl_embedding = VLEmbeddingConfig(
        **(agent_cfg.get("vl_embedding") or agent_cfg.get("vl-embedding") or {})
    )

    # 其他
    skills = SkillsConfig(**(agent_cfg.get("skills") or {}))
    memory = MemoryConfig(**(agent_cfg.get("memory") or {}))
    session = SessionConfig(**(agent_cfg.get("session") or {}))
    gateway = GatewayConfig(
        host=gateway_cfg.get("host", "0.0.0.0"),
        port=gateway_cfg.get("port", 18789),
        cors_origins=gateway_cfg.get("cors_origins", ["*"]),
    )
    log = LoggingConfig(**logging_cfg) if logging_cfg else LoggingConfig()

    scheduler = SchedulerConfig(
        enabled=bool(scheduler_cfg.get("enabled", False)),
        tick_seconds=int(scheduler_cfg.get("tick_seconds", 30)),
        jobs=list(scheduler_cfg.get("jobs") or []),
    )

    # multi-agent
    agents_raw = raw.get("agents") or {}
    agents = AgentsConfig(
        default_agent=str(agents_raw.get("default_agent", "main")),
        agent_list=list(agents_raw.get("list") or []),
        routing=list(agents_raw.get("routing") or []),
    )

    # bash
    bash_raw = dict(agent_cfg.get("bash") or {})
    bash = BashConfig(**bash_raw) if bash_raw else BashConfig()

    # safety（全盘文件访问 + 命令执行）
    safety_raw = dict(agent_cfg.get("safety") or {})
    safety_valid = {f.name for f in fields(SafetyConfig)}
    safety = SafetyConfig(
        **{k: v for k, v in safety_raw.items() if k in safety_valid}
    )

    # plugins
    plugins_raw = raw.get("plugins") or {}
    load_paths: list = []
    load_block = plugins_raw.get("load") or {}
    if isinstance(load_block, dict):
        load_paths = list(load_block.get("paths") or [])
    plugins = PluginsConfig(
        enabled=bool(plugins_raw.get("enabled", True)),
        allow=list(plugins_raw.get("allow") or []),
        deny=list(plugins_raw.get("deny") or []),
        load_paths=load_paths,
        slots=dict(plugins_raw.get("slots") or {
            "memory": "memory-core",
            "contextEngine": "none",
        }),
        entries=dict(plugins_raw.get("entries") or {}),
        install_registry=str(
            plugins_raw.get("install_registry", "~/.openclaw/plugins")
        ),
        npm_registry=str(
            plugins_raw.get("npm_registry", "~/.openclaw/npm_store")
        ),
    )

    # computer_use
    cu_raw = dict(agent_cfg.get("computer_use") or {})
    cu_browser_raw = dict(cu_raw.pop("browser", None) or {})
    cu_audit_raw = dict(cu_raw.pop("audit", None) or {})
    # 兼容嵌套段
    cu_kwargs = dict(cu_raw)
    for k, v in cu_browser_raw.items():
        cu_kwargs[f"browser_{k}"] = v
    for k, v in cu_audit_raw.items():
        if k == "log_path":
            cu_kwargs["audit_log_path"] = v
        elif k == "thumbnails":
            cu_kwargs["audit_thumbnails"] = v
        elif k == "thumbnail_dir":
            cu_kwargs["audit_thumbnail_dir"] = v
    # 过滤掉未识别字段
    valid = {f.name for f in fields(ComputerUseConfig)}
    computer_use = ComputerUseConfig(
        **{k: v for k, v in cu_kwargs.items() if k in valid}
    )

    return Settings(
        agent_name=str(agent_cfg.get("name", "jimi")),
        model=model,
        embedding=embedding,
        vlm=vlm,
        vl_embedding=vl_embedding,
        workspace_path=agent_cfg.get("workspace", "./workspace"),
        skills=skills,
        memory=memory,
        session=session,
        gateway=gateway,
        logging=log,
        channels=ChannelsConfig(raw=channels_cfg),
        scheduler=scheduler,
        agents=agents,
        bash=bash,
        safety=safety,
        plugins=plugins,
        computer_use=computer_use,
    )


def _apply_env_overrides(s: Settings) -> Settings:
    """用 JIMI_* 环境变量覆盖 yaml 里的对应字段（可选）"""
    mapping = {
        "JIMI_API_KEY": ("model", "api_key"),
        "JIMI_BASE_URL": ("model", "base_url"),
        "JIMI_MODEL": ("model", "model_id"),
        "JIMI_EMBEDDING_API_KEY": ("embedding", "api_key"),
        "JIMI_EMBEDDING_BASE_URL": ("embedding", "base_url"),
        "JIMI_EMBEDDING_MODEL": ("embedding", "model"),
        # VLM / VL-Embedding 显式覆盖
        "JIMI_VLM_API_KEY": ("vlm", "api_key"),
        "JIMI_VLM_BASE_URL": ("vlm", "base_url"),
        "JIMI_VLM_MODEL": ("vlm", "model_id"),
        "JIMI_VL_EMBEDDING_API_KEY": ("vl_embedding", "api_key"),
        "JIMI_VL_EMBEDDING_BASE_URL": ("vl_embedding", "base_url"),
        "JIMI_VL_EMBEDDING_MODEL": ("vl_embedding", "model"),
        "JIMI_GATEWAY_HOST": ("gateway", "host"),
        "JIMI_LOG_LEVEL": ("logging", "level"),
        "JIMI_LOG_OUTPUT": ("logging", "output"),
        "JIMI_LOG_FILE": ("logging", "file_path"),
    }
    for env_key, (section, field_name) in mapping.items():
        v = os.getenv(env_key)
        if v:
            setattr(getattr(s, section), field_name, v)

    # 数值型
    port_env = os.getenv("JIMI_GATEWAY_PORT")
    if port_env and port_env.isdigit():
        s.gateway.port = int(port_env)

    # Computer Use 环境变量覆盖
    cu_env = os.getenv("JIMI_COMPUTER_USE_ENABLED")
    if cu_env is not None:
        s.computer_use.enabled = cu_env.lower() in ("1", "true", "yes", "on")
    bm = os.getenv("JIMI_BROWSER_MODE")
    if bm in ("launch", "cdp"):
        s.computer_use.browser_mode = bm
    bh = os.getenv("JIMI_BROWSER_HEADLESS")
    if bh is not None:
        s.computer_use.browser_headless = bh.lower() in ("1", "true", "yes", "on")

    # Embedding 缺省时回退到 model
    if not s.embedding.api_key:
        s.embedding.api_key = s.model.api_key
    if not s.embedding.base_url:
        s.embedding.base_url = s.model.base_url

    return s


def load_settings(config_path: Optional[str] = None) -> Settings:
    """加载 yaml 并返回 Settings 实例（允许 JIMI_CONFIG 覆盖路径）"""
    if config_path is None:
        config_path = os.getenv("JIMI_CONFIG") or (
            PROJECT_ROOT / "config" / "agent_config.yaml"
        )
    config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        settings = _parse_config(raw)
    else:
        # 仅在同级模板存在时提示
        example = config_path.with_name("agent_config.example.yaml")
        if example.exists():
            import warnings
            warnings.warn(
                f"未找到 {config_path}，但检测到模板 {example}。\n"
                f"请先执行:  cp {example} {config_path}\n"
                f"再填入真实 api_key / base_url。\n"
                f"当前先用内置默认值启动，部分功能（LLM、Embedding）将不可用。",
                RuntimeWarning,
                stacklevel=2,
            )
        settings = Settings()

    return _apply_env_overrides(settings)


# 全局单例
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """获取全局配置单例"""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reload_settings() -> Settings:
    """强制重新加载配置（测试/热更用）"""
    global _settings
    _settings = load_settings()
    return _settings
