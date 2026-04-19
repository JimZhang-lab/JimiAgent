'''K4 - 插件注册表。'''
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from server.config.settings import PluginsConfig, Settings
from server.core.plugin_manifest import (
    PluginManifest,
    find_manifest_file,
    parse_manifest,
)

logger = logging.getLogger(__name__)


@dataclass
class PluginState:
    """合并清单信息与当前运行开关等状态"""
    manifest: PluginManifest
    enabled: bool = True
    reason_disabled: str = ""  # allow/deny/entries/slots-none 导致禁用
    warnings: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.manifest.id

    @property
    def status(self) -> str:
        """对外三态：ok / disabled / invalid"""
        if self.manifest.status != "ok":
            return self.manifest.status
        return "ok" if self.enabled else "disabled"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.manifest.name,
            "version": self.manifest.version,
            "description": self.manifest.description,
            "kind": self.manifest.kind,
            "source": str(self.manifest.source),
            "manifest_file": str(self.manifest.manifest_file),
            "manifest_kind": self.manifest.manifest_kind,
            "status": self.status,
            "enabled": self.enabled,
            "reason_disabled": self.reason_disabled,
            "error": self.manifest.error,
            "warnings": list(self.warnings),
            "skills": list(self.manifest.skills),
            "providers": list(self.manifest.providers),
            "channels": list(self.manifest.channels),
            "command_aliases": list(self.manifest.command_aliases),
        }


class PluginRegistry:
    """插件发现与开关聚合注册表"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._plugins: dict[str, PluginState] = {}
        self._scanned = False

    # ---- 发现路径 ----

    def _discovery_paths(self) -> list[Path]:
        """按优先级返回所有应扫描的目录列表。"""
        cfg: PluginsConfig = self.settings.plugins
        paths: list[Path] = []

        # 1. yaml 显式路径
        for p in (cfg.load_paths or []):
            try:
                paths.append(Path(p).expanduser().resolve())
            except Exception:
                continue

        # 2. workspace 插件目录
        ws = self.settings.workspace_abs_path
        ws_plug = (ws / ".openclaw" / "plugins").resolve()
        if ws_plug.exists():
            paths.extend(
                [p for p in ws_plug.iterdir() if p.is_dir()]
            )

        # 3. 全局 ~/.openclaw/plugins
        try:
            install_root = Path(cfg.install_registry).expanduser().resolve()
            if install_root.exists():
                paths.extend(
                    [p for p in install_root.iterdir() if p.is_dir()]
                )
        except Exception:
            pass

        # 4. npm_store/node_modules
        try:
            npm_root = Path(cfg.npm_registry).expanduser().resolve()
            nm = npm_root / "node_modules"
            if nm.exists():
                # @openclaw scope
                scope = nm / "@openclaw"
                if scope.exists():
                    paths.extend(
                        [p for p in scope.iterdir() if p.is_dir()]
                    )
                # 顶层 openclaw-*
                for p in nm.iterdir():
                    if p.is_dir() and p.name.startswith("openclaw-"):
                        paths.append(p)
        except Exception:
            pass

        return paths

    # ---- 扫描 ----

    def scan(self) -> None:
        """扫描所有目录进行插件加载解析并更新状态"""
        cfg: PluginsConfig = self.settings.plugins
        self._plugins.clear()

        if not cfg.enabled:
            logger.info("Plugins disabled by config，跳过扫描")
            self._scanned = True
            return

        allow = set(cfg.allow or [])
        deny = set(cfg.deny or [])
        entries = cfg.entries or {}

        for d in self._discovery_paths():
            if not d.exists() or not d.is_dir():
                continue
            mf_file, _kind = find_manifest_file(d)
            if mf_file is None:
                continue  # 目录里没 manifest，忽略

            manifest = parse_manifest(d)
            if manifest is None:
                continue

            if manifest.id in self._plugins:
                continue  # 先到先得

            # allow/deny/entries 决定最终 enabled
            enabled = manifest.enabled_by_default
            reason = ""
            if allow and manifest.id not in allow:
                enabled = False
                reason = f"allow 白名单不含 {manifest.id}"
            if manifest.id in deny:
                enabled = False
                reason = "被 deny 列表排除"

            entry = entries.get(manifest.id) or {}
            if isinstance(entry, dict) and "enabled" in entry:
                # yaml 显式覆盖
                override = bool(entry.get("enabled"))
                enabled = override
                if not override:
                    reason = reason or "yaml entries 显式 disabled"

            # invalid 插件不参与 enabled
            if manifest.status != "ok":
                enabled = False

            # 运行告警
            warnings = _validate_state(manifest)

            self._plugins[manifest.id] = PluginState(
                manifest=manifest,
                enabled=enabled,
                reason_disabled=reason,
                warnings=warnings,
            )

        self._scanned = True
        logger.info(
            f"PluginRegistry 扫描完成：{len(self._plugins)} 个插件 "
            f"(enabled={sum(1 for s in self._plugins.values() if s.enabled)})"
        )

    # ---- 查询 ----

    def list(self) -> list[PluginState]:
        if not self._scanned:
            self.scan()
        return sorted(self._plugins.values(), key=lambda s: s.id)

    def get(self, plugin_id: str) -> Optional[PluginState]:
        if not self._scanned:
            self.scan()
        return self._plugins.get(plugin_id)

    def enabled_plugins(self) -> list[PluginState]:
        return [s for s in self.list() if s.enabled]

    # ---- 开关（仅内存） ----

    def enable(self, plugin_id: str) -> bool:
        state = self.get(plugin_id)
        if state is None:
            return False
        state.enabled = True
        state.reason_disabled = ""
        return True

    def disable(self, plugin_id: str) -> bool:
        state = self.get(plugin_id)
        if state is None:
            return False
        state.enabled = False
        state.reason_disabled = "手动 disable"
        return True

    # ---- Skill 目录 ----

    def active_skill_dirs(self) -> list[Path]:
        """获取所有已启用插件的 skill 目录绝对路径"""
        dirs: list[Path] = []
        for s in self.enabled_plugins():
            for sd in s.manifest.resolved_skill_dirs():
                if sd not in dirs:
                    dirs.append(sd)
        return dirs


# ---- helpers ----

def _validate_state(manifest: PluginManifest) -> list[str]:
    from server.core.plugin_manifest import validate_manifest
    try:
        return validate_manifest(manifest)
    except Exception as e:
        return [f"validate 抛异常: {e}"]
