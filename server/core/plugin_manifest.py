'''K3 - 插件 manifest 解析。'''
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---- dataclass ----

@dataclass
class PluginManifest:
    """静态 manifest 数据结构"""
    id: str
    name: str = ""
    description: str = ""
    version: str = ""
    source: Path = Path()          # 插件根目录（含 openclaw.plugin.json 的目录）
    manifest_file: Path = Path()   # 实际读到的 manifest 文件路径
    manifest_kind: str = "openclaw"  # openclaw / claude / cursor / codex
    enabled_by_default: bool = True
    status: str = "ok"
    error: str = ""

    # runtime metadata
    kind: str = ""                 # "" / "memory" / "context-engine"
    skills: list[str] = field(default_factory=list)       # 相对/绝对路径
    providers: list[str] = field(default_factory=list)
    cli_backends: list[str] = field(default_factory=list)
    channels: list[str] = field(default_factory=list)
    legacy_plugin_ids: list[str] = field(default_factory=list)

    # activation
    on_providers: list[str] = field(default_factory=list)
    on_commands: list[str] = field(default_factory=list)
    on_channels: list[str] = field(default_factory=list)
    on_routes: list[str] = field(default_factory=list)
    on_capabilities: list[str] = field(default_factory=list)

    # auth
    provider_auth_env_vars: dict[str, list[str]] = field(default_factory=dict)
    channel_env_vars: dict[str, list[str]] = field(default_factory=dict)

    # command aliases
    command_aliases: list[dict] = field(default_factory=list)

    # config
    config_schema: dict = field(default_factory=dict)
    ui_hints: dict = field(default_factory=dict)

    # 原始 JSON
    raw: dict = field(default_factory=dict)

    def resolved_skill_dirs(self) -> list[Path]:
        """解析并返回有效的 skill 绝对路径列表"""
        out: list[Path] = []
        for s in self.skills:
            p = Path(s)
            if not p.is_absolute():
                p = (self.source / p).resolve()
            if p.exists() and p.is_dir():
                out.append(p)
        return out


# ---- 路径 ----

_BUNDLE_CANDIDATES: list[tuple[str, str]] = [
    # (相对路径, manifest_kind)
    ("openclaw.plugin.json", "openclaw"),
    (".claude-plugin/plugin.json", "claude"),
    (".cursor-plugin/plugin.json", "cursor"),
    (".codex-plugin/plugin.json", "codex"),
]


def find_manifest_file(plugin_dir: Path) -> tuple[Optional[Path], str]:
    """在目录中查找首个支持的 manifest 文件"""
    if not plugin_dir.exists() or not plugin_dir.is_dir():
        return None, ""
    for rel, kind in _BUNDLE_CANDIDATES:
        p = plugin_dir / rel
        if p.exists() and p.is_file():
            return p, kind
    return None, ""


# ---- 解析 ----

def _as_str_list(v) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if x is not None]
    if isinstance(v, str) and v:
        return [v]
    return []


def _as_str_list_dict(v) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    if isinstance(v, dict):
        for k, vv in v.items():
            out[str(k)] = _as_str_list(vv)
    return out


def parse_manifest(plugin_dir: Path) -> Optional[PluginManifest]:
    """解析并返回目录下的 manifest 数据"""
    if not plugin_dir.exists() or not plugin_dir.is_dir():
        return None

    mf_path, kind = find_manifest_file(plugin_dir)
    if mf_path is None:
        return None

    try:
        raw = json.loads(mf_path.read_text(encoding="utf-8"))
    except Exception as e:
        return PluginManifest(
            id=plugin_dir.name,
            source=plugin_dir,
            manifest_file=mf_path,
            manifest_kind=kind,
            status="invalid",
            error=f"JSON 解析失败: {type(e).__name__}: {e}",
        )

    if not isinstance(raw, dict):
        return PluginManifest(
            id=plugin_dir.name,
            source=plugin_dir,
            manifest_file=mf_path,
            manifest_kind=kind,
            status="invalid",
            error="manifest 顶层不是 object",
            raw={"_root": raw},
        )

    # 缺 id/name 时用目录名兜底
    pid = str(raw.get("id") or raw.get("name") or plugin_dir.name).strip()
    if not pid:
        return PluginManifest(
            id=plugin_dir.name,
            source=plugin_dir,
            manifest_file=mf_path,
            manifest_kind=kind,
            status="invalid",
            error="manifest 缺少 id/name",
            raw=raw,
        )

    activation = raw.get("activation") or {}
    if not isinstance(activation, dict):
        activation = {}

    manifest = PluginManifest(
        id=pid,
        name=str(raw.get("name", pid)),
        description=str(raw.get("description", "")),
        version=str(raw.get("version", "")),
        source=plugin_dir,
        manifest_file=mf_path,
        manifest_kind=kind,
        enabled_by_default=bool(raw.get("enabledByDefault", True)),
        kind=str(raw.get("kind", "")),
        skills=_as_str_list(raw.get("skills")),
        providers=_as_str_list(raw.get("providers")),
        cli_backends=_as_str_list(raw.get("cliBackends")),
        channels=_as_str_list(raw.get("channels")),
        legacy_plugin_ids=_as_str_list(raw.get("legacyPluginIds")),
        on_providers=_as_str_list(activation.get("onProviders")),
        on_commands=_as_str_list(activation.get("onCommands")),
        on_channels=_as_str_list(activation.get("onChannels")),
        on_routes=_as_str_list(activation.get("onRoutes")),
        on_capabilities=_as_str_list(activation.get("onCapabilities")),
        provider_auth_env_vars=_as_str_list_dict(raw.get("providerAuthEnvVars")),
        channel_env_vars=_as_str_list_dict(raw.get("channelEnvVars")),
        command_aliases=[
            c for c in (raw.get("commandAliases") or []) if isinstance(c, dict)
        ],
        config_schema=raw.get("configSchema") or {},
        ui_hints=raw.get("uiHints") or {},
        raw=raw,
    )

    # 缺 skills 时默认使用 ./skills
    if kind in ("claude", "cursor", "codex") and not manifest.skills:
        default_skills = plugin_dir / "skills"
        if default_skills.is_dir():
            manifest.skills = ["./skills"]

    return manifest


def validate_manifest(manifest: PluginManifest) -> list[str]:
    """进行非阻断检查并返回告警列表"""
    import os as _os
    warnings: list[str] = []

    if manifest.kind in ("memory", "context-engine"):
        rt = manifest.source / "runtime.py"
        if not rt.exists():
            warnings.append(
                f"kind={manifest.kind} 但缺 runtime.py，slots 将无法装载"
            )

    for s in manifest.skills:
        p = Path(s)
        if not p.is_absolute():
            p = manifest.source / p
        if not p.exists():
            warnings.append(f"skills 路径 {s} 不存在")

    for prov, envs in manifest.provider_auth_env_vars.items():
        missing = [e for e in envs if not _os.environ.get(e)]
        if missing:
            warnings.append(
                f"provider {prov} 缺环境变量: {missing}"
            )

    return warnings
