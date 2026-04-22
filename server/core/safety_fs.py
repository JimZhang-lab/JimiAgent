"""文件系统与 shell 安全判定。"""
from __future__ import annotations

import fnmatch
import logging
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# 系统硬拒路径：命中后任何 confirm 都无效。
SYSTEM_DENY_PATTERNS: tuple[str, ...] = (
    # macOS 系统目录
    "/System/*",
    "/Library/Keychains/*",
    "/private/etc/*",
    "/private/var/db/*",
    # Unix 通用
    "/bin/*", "/sbin/*", "/usr/bin/*", "/usr/sbin/*", "/usr/local/bin/*",
    "/etc/*",
    # 用户凭据目录
    "*/.ssh/*",
    "*/.gnupg/*",
    "*/Library/Keychains/*",
    # 项目凭据
    "*/credentials*",
    "*/.aws/credentials",
    "*/.kube/config",
)


# 命令硬拒模式，比 file_tools 里的老常量更精确。
COMMAND_DENY_PATTERNS: tuple[str, ...] = (
    r"\brm\s+-rf\s+/\s*$",
    r"\brm\s+-rf\s+/\*",
    r"\brm\s+-rf\s+~\s*$",
    r"\brm\s+-rf\s+\$HOME",
    r"\bmkfs\.",
    r"\bdd\s+if=.*of=/dev/",
    r":\(\)\s*\{",              # fork bomb
    r"\bchmod\s+-R\s+777\s+/",
    r">\s*/dev/sd[a-z]",
    r"\bshutdown\b",
    r"\breboot\b",
    r"\bhalt\b",
    r"\bpoweroff\b",
)


# 读类命令前缀：命中首 token 即视为 safe。
COMMAND_SAFE_HEADS: frozenset[str] = frozenset({
    "ls", "pwd", "cat", "head", "tail", "wc", "file", "stat",
    "echo", "which", "whoami", "uname", "date", "hostname",
    "find", "grep", "rg", "ripgrep", "fd", "tree",
    "ps", "top", "df", "du", "free", "uptime",
    "env", "printenv",
    "git",   # 写子命令后面单独处理
})

# git 写子命令需要 confirm
_GIT_WRITE_SUBCMDS: frozenset[str] = frozenset({
    "commit", "push", "pull", "merge", "rebase", "reset", "clean",
    "checkout", "switch", "branch", "tag", "stash", "revert",
    "cherry-pick", "rm", "mv", "add",
})


@dataclass(frozen=True)
class SafetyVerdict:
    """安全判定结果。"""
    decision: str  # "allow" | "confirm" | "deny"
    reason: str = ""
    # confirm 时给 UI 展示的摘要
    summary: str = ""

    @property
    def is_allow(self) -> bool:
        return self.decision == "allow"

    @property
    def needs_confirm(self) -> bool:
        return self.decision == "confirm"

    @property
    def is_deny(self) -> bool:
        return self.decision == "deny"


def _match_any(patterns, value: str) -> bool:
    if not patterns or not value:
        return False
    v = value.lower() if not value.startswith("/") else value
    for pat in patterns:
        if fnmatch.fnmatchcase(v, pat.lower() if not pat.startswith("/") else pat):
            return True
    return False


def _resolve_path(path_str: str) -> Path:
    """统一 expanduser + resolve。resolve 不要 strict，允许不存在的目标。"""
    return Path(path_str).expanduser().resolve()


def _is_subpath(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def is_path_denied(path_str: str, extra_patterns: Optional[list[str]] = None) -> tuple[bool, str]:
    """判断路径是否命中硬拒黑名单。

    返回 `(denied, reason)`。`reason` 空表示允许。
    """
    if not path_str:
        return False, ""
    try:
        p = _resolve_path(path_str)
    except Exception as e:
        return True, f"路径解析失败：{e}"

    # 合并内置与配置扩展
    patterns = list(SYSTEM_DENY_PATTERNS)
    if extra_patterns:
        patterns.extend(extra_patterns)

    p_str = str(p)
    for pat in patterns:
        # 展开 ~ 再匹配
        expanded = str(Path(pat.rstrip("*")).expanduser().resolve())
        # 同时支持 fnmatch 和 prefix
        if fnmatch.fnmatchcase(p_str, pat):
            return True, f"路径 {p_str} 命中系统保护黑名单: {pat}"
        if pat.endswith("/*") and (p_str == expanded or _is_subpath(p, Path(expanded))):
            return True, f"路径 {p_str} 位于系统保护目录: {pat}"
    return False, ""


def classify_path_write(
    path_str: str,
    workspace: Optional[Path] = None,
    extra_deny: Optional[list[str]] = None,
) -> SafetyVerdict:
    """对**写**类路径操作做分类。

    workspace 内 → allow；workspace 外 → confirm；系统黑名单 → deny。
    """
    denied, reason = is_path_denied(path_str, extra_deny)
    if denied:
        return SafetyVerdict("deny", reason=reason)

    try:
        p = _resolve_path(path_str)
    except Exception as e:
        return SafetyVerdict("deny", reason=f"路径解析失败：{e}")

    if workspace is not None:
        try:
            ws = Path(workspace).expanduser().resolve()
            if _is_subpath(p, ws):
                return SafetyVerdict(
                    "allow",
                    summary=f"写入 workspace 内: {p}",
                )
        except Exception:
            pass

    return SafetyVerdict(
        "confirm",
        reason="workspace 外写入",
        summary=f"写入: {p}",
    )


def classify_path_read(
    path_str: str,
    extra_deny: Optional[list[str]] = None,
) -> SafetyVerdict:
    """对**读**类路径操作做分类。读默认 allow；仅硬拒系统黑名单。"""
    denied, reason = is_path_denied(path_str, extra_deny)
    if denied:
        return SafetyVerdict("deny", reason=reason)
    try:
        p = _resolve_path(path_str)
    except Exception as e:
        return SafetyVerdict("deny", reason=f"路径解析失败：{e}")
    return SafetyVerdict("allow", summary=f"读取: {p}")


def classify_command(
    command: str,
    workspace: Optional[Path] = None,
    extra_deny: Optional[list[str]] = None,
    extra_safe_heads: Optional[list[str]] = None,
) -> SafetyVerdict:
    """对 shell 命令做分类。

    - 硬拒：毁灭性模式（rm -rf /, mkfs, fork bomb 等）
    - safe：read-only 前缀 + git 非写子命令
    - confirm：其他所有
    """
    if not command or not command.strip():
        return SafetyVerdict("deny", reason="空命令")

    cmd_l = command.strip()

    # 先看硬拒模式
    for pat in COMMAND_DENY_PATTERNS:
        if re.search(pat, cmd_l):
            return SafetyVerdict(
                "deny",
                reason=f"命令命中硬拒模式: {pat}",
            )

    # 再看额外 deny
    if extra_deny:
        for pat in extra_deny:
            if re.search(pat, cmd_l):
                return SafetyVerdict(
                    "deny",
                    reason=f"命令命中用户 denylist: {pat}",
                )

    # 管道、重定向和组合符优先走 confirm，避免误放行。
    if any(sym in cmd_l for sym in ("|", ">", "<", ";", "&&", "||", "`", "$(")):
        return SafetyVerdict(
            "confirm",
            reason="含管道 / 重定向 / 组合符",
            summary=f"执行: {cmd_l}",
        )

    # 拆第一个 token
    try:
        tokens = shlex.split(cmd_l)
    except ValueError:
        # 奇怪引号等解析失败时交给人工确认
        return SafetyVerdict(
            "confirm",
            reason="命令解析失败，需人工确认",
            summary=f"执行: {cmd_l}",
        )
    if not tokens:
        return SafetyVerdict("deny", reason="空命令")

    head = Path(tokens[0]).name.lower()
    safe_heads = set(COMMAND_SAFE_HEADS)
    if extra_safe_heads:
        safe_heads.update(h.lower() for h in extra_safe_heads)

    # git 写子命令单独处理
    if head == "git" and len(tokens) > 1:
        sub = tokens[1].lower()
        if sub in _GIT_WRITE_SUBCMDS:
            return SafetyVerdict(
                "confirm",
                reason=f"git 写子命令: {sub}",
                summary=f"git {sub} ...",
            )
        return SafetyVerdict("allow", summary=f"git {sub}")

    if head in safe_heads:
        return SafetyVerdict("allow", summary=f"read-only: {head} ...")

    # 其余未知命令默认 confirm
    return SafetyVerdict(
        "confirm",
        reason=f"非白名单命令: {head}",
        summary=f"执行: {cmd_l}",
    )
