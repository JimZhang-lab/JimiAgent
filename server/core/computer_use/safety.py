"""Computer Use 安全检查。"""
from __future__ import annotations

import fnmatch
import logging
import re
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from server.config.settings import ComputerUseConfig

logger = logging.getLogger(__name__)


class ComputerUsePermissionError(PermissionError):
    """Computer Use 安全拒绝时抛出。"""


def _match_any(patterns: list[str], value: str) -> bool:
    """fnmatch 风格匹配。"""
    if not patterns or not value:
        return False
    v = value.lower()
    for pat in patterns:
        if fnmatch.fnmatchcase(v, pat.lower()):
            return True
    return False


def get_frontmost_app() -> Optional[dict]:
    """返回前台 app 信息。"""
    try:
        # macOS 依赖通常由 pyautogui 带上
        from AppKit import NSWorkspace
        ws = NSWorkspace.sharedWorkspace()
        active = ws.frontmostApplication()
        if active is None:
            return None
        return {
            "name": str(active.localizedName() or ""),
            "bundle_id": str(active.bundleIdentifier() or ""),
        }
    except Exception as e:
        logger.debug(f"get_frontmost_app 失败: {e}")
        return None


def check_app(cfg: "ComputerUseConfig", action: str = "") -> None:
    """检查前台 app 是否允许被操控。"""
    app = get_frontmost_app()
    if app is None:
        # 识别失败时走 fail_policy
        if cfg.safety_fail_policy == "allow":
            return
        raise ComputerUsePermissionError(
            f"无法识别前台 app（safety_fail_policy=deny）→ 拒绝 {action or '操作'}"
        )

    name = app.get("name", "")
    bid = app.get("bundle_id", "")

    # 1) denylist 优先
    for candidate in (name, bid):
        if _match_any(cfg.app_denylist, candidate):
            raise ComputerUsePermissionError(
                f"app {name!r} ({bid}) 在 denylist 中 → 拒绝 {action or '操作'}"
            )

    # 2) allowlist 非空时必须命中
    if cfg.app_allowlist:
        hit = any(
            _match_any(cfg.app_allowlist, c) for c in (name, bid) if c
        )
        if not hit:
            raise ComputerUsePermissionError(
                f"app {name!r} ({bid}) 不在 allowlist 中 → 拒绝 {action or '操作'}"
            )


def check_url(cfg: "ComputerUseConfig", url: str, action: str = "") -> None:
    """检查目标 URL 是否允许。"""
    if not url:
        return
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        host = ""
    if not host:
        return  # 相对路径和 data: URL 不拦

    # 1) denylist 优先
    if _match_any(cfg.domain_denylist, host):
        raise ComputerUsePermissionError(
            f"domain {host!r} 在 denylist 中 → 拒绝 {action or '浏览器操作'}"
        )

    # 2) allowlist 非空必须命中
    if cfg.domain_allowlist and not _match_any(cfg.domain_allowlist, host):
        raise ComputerUsePermissionError(
            f"domain {host!r} 不在 allowlist 中 → 拒绝 {action or '浏览器操作'}"
        )


# 高危键组合归一化
_KEY_SEP = re.compile(r"[\s+]+")


def _normalize_keys(keys: str) -> str:
    parts = sorted(p.strip().lower() for p in _KEY_SEP.split(keys or "") if p.strip())
    return "+".join(parts)


def check_dangerous_keypress(cfg: "ComputerUseConfig", keys: str) -> None:
    """硬拒绝高危快捷键。"""
    norm = _normalize_keys(keys)
    for banned in cfg.dangerous_keypress_deny:
        if _normalize_keys(banned) == norm:
            raise ComputerUsePermissionError(
                f"快捷键 {keys!r} 在 dangerous_keypress_deny 中 → 硬拒绝"
            )


def is_enabled(cfg: "ComputerUseConfig", layer: Optional[str] = None) -> bool:
    """层级开关检查。"""
    if not cfg.enabled:
        return False
    if layer == "desktop":
        return cfg.desktop_enabled
    if layer == "browser":
        return cfg.browser_enabled
    if layer == "loop":
        return cfg.loop_agent_enabled
    return True
