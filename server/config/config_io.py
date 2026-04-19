'''
Author: JimZhang
Date: 2026-04-19 15:15:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 15:15:00
FilePath: /JimiAgent/server/config/config_io.py
Description: YAML 配置读写工具。

'''
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Optional

from ruamel.yaml import YAML

from server.config.settings import PROJECT_ROOT


# 点分路径白名单
# 结构：dotted.path -> Python 类型
# 复杂结构请直接 `config edit`
MUTABLE_KEYS: dict[str, type] = {
    # 模型
    "agent.model.provider": str,
    "agent.model.model_id": str,
    "agent.model.temperature": float,
    "agent.model.max_tokens": int,
    "agent.model.streaming": bool,
    "agent.model.api_key": str,
    "agent.model.base_url": str,
    # Embedding
    "agent.embedding.model": str,
    "agent.embedding.api_key": str,
    "agent.embedding.base_url": str,
    # Skills
    "agent.skills.top_k": int,
    "agent.skills.similarity_threshold": float,
    # 会话
    "agent.session.max_history_messages": int,
    "agent.session.compact_threshold": int,
    "agent.session.default_think_level": str,
    # Gateway
    "gateway.host": str,
    "gateway.port": int,
    # 日志
    "logging.level": str,
    "logging.format": str,
    "logging.output": str,
    "logging.file_path": str,
    "logging.rotate_mb": int,
    "logging.backup_count": int,
    # Scheduler
    "scheduler.enabled": bool,
    "scheduler.tick_seconds": int,
    # Channels
    "channels.webchat.enabled": bool,
    "channels.terminal.enabled": bool,
    "channels.webhook.enabled": bool,
    "channels.webhook.secret": str,
    "channels.webhook.session_prefix": str,
    # Computer Use
    "agent.computer_use.enabled": bool,
    "agent.computer_use.desktop_enabled": bool,
    "agent.computer_use.browser_enabled": bool,
    "agent.computer_use.loop_agent_enabled": bool,
    "agent.computer_use.browser_mode": str,            # launch / cdp
    "agent.computer_use.browser_headless": bool,
    "agent.computer_use.audit_thumbnails": bool,
    # Hybrid 检索
    "agent.skills.retrieval_mode": str,
    "agent.skills.query_cache_ttl_seconds": int,
    "agent.skills.exact_match_short_circuit": bool,
}


_yaml = YAML(typ="rt")  # round-trip 保注释
_yaml.preserve_quotes = True
_yaml.indent(mapping=2, sequence=4, offset=2)


def get_config_path() -> Path:
    """按 JIMI_CONFIG 环境变量 > 默认 config/agent_config.yaml 解析"""
    env = os.getenv("JIMI_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    return (PROJECT_ROOT / "config" / "agent_config.yaml").resolve()


def load_raw() -> Any:
    """加载原始 yaml（roundtrip，保留注释）"""
    path = get_config_path()
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return _yaml.load(f) or {}


def dump_raw(doc: Any) -> None:
    """写回 yaml（保留注释）"""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        _yaml.dump(doc, f)


def _navigate(doc: Any, parts: list[str], create: bool = False) -> Any:
    """按 dotted path 逐层深入。create=True 时缺失节点自动补 dict。"""
    cur = doc
    for p in parts:
        if isinstance(cur, dict) and p in cur:
            cur = cur[p]
        elif create and isinstance(cur, dict):
            cur[p] = {}
            cur = cur[p]
        else:
            raise KeyError(".".join(parts))
    return cur


def get_value(key: str) -> Any:
    """读取 dotted path 对应的配置值；缺失则抛 KeyError"""
    if not key:
        return load_raw()
    doc = load_raw()
    parts = key.split(".")
    parent = _navigate(doc, parts[:-1]) if len(parts) > 1 else doc
    if not isinstance(parent, dict) or parts[-1] not in parent:
        raise KeyError(key)
    return parent[parts[-1]]


def _coerce(value: str, target_type: type) -> Any:
    """把 CLI / TUI 传入的字符串 value 转成目标类型"""
    if target_type is bool:
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes", "on"):
            return True
        if lowered in ("false", "0", "no", "off"):
            return False
        raise ValueError(f"布尔值只接受 true/false，收到 {value!r}")
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    return value  # str


def set_value(key: str, value: Any) -> Any:
    """按 dotted path 写入，持久化到 yaml"""
    if key not in MUTABLE_KEYS:
        raise PermissionError(
            f"key {key!r} 不在可修改白名单。可用 `config list` 查看，"
            "或 `config edit` 打开编辑器直接改 yaml"
        )

    target_type = MUTABLE_KEYS[key]
    if isinstance(value, str) and target_type is not str:
        value = _coerce(value, target_type)
    elif target_type is str and not isinstance(value, str):
        value = str(value)

    doc = load_raw()
    parts = key.split(".")
    if len(parts) == 1:
        doc[parts[0]] = value
    else:
        parent = _navigate(doc, parts[:-1], create=True)
        if not isinstance(parent, dict):
            raise KeyError(f"{'.'.join(parts[:-1])} 不是 mapping")
        parent[parts[-1]] = value
    dump_raw(doc)
    return value


def list_keys() -> list[tuple[str, str]]:
    """列出所有可修改 key 与当前值"""
    doc = load_raw()
    out: list[tuple[str, str]] = []
    for key in MUTABLE_KEYS:
        parts = key.split(".")
        try:
            parent = _navigate(doc, parts[:-1]) if len(parts) > 1 else doc
            if isinstance(parent, dict) and parts[-1] in parent:
                val = parent[parts[-1]]
                out.append((key, repr(val)))
            else:
                out.append((key, "(未设置)"))
        except KeyError:
            out.append((key, "(未设置)"))
    return out


def render_doc(doc: Optional[Any] = None) -> str:
    """序列化 yaml 文档为字符串（用于 TUI 打印）"""
    buf = io.StringIO()
    _yaml.dump(doc if doc is not None else load_raw(), buf)
    return buf.getvalue()
