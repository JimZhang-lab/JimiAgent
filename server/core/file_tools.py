'''
Author: JimZhang
LastEditors: Cascade
LastEditTime: 2026-04-21
FilePath: /JimiAgent/server/core/file_tools.py
Description: 全盘文件与 Shell 内置工具。

安全架构：
- `settings.safety.enabled=True`（默认）：走 `safety_fs` 分类
    - safe    → 直接执行
    - confirm → 依 `confirm_mode` 处理：
        * "llm"  → 返回 `[PENDING CONFIRM]`，等 LLM 询问用户后再带 `confirm=True` 重试
        * "ui"   → 调 LangGraph `interrupt(...)` 暂停 graph，前端/TUI 弹框（Stage 2）
        * "auto" → ui 可用则 ui，否则退化 llm
    - deny    → 硬拒，不允许 confirm
- `settings.safety.enabled=False`：兼容老行为（workspace-only 白名单）
'''
import logging
import os
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from langchain_core.tools import StructuredTool

from server.core.safety_fs import (
    classify_command,
    classify_path_read,
    classify_path_write,
    is_path_denied,
)

if TYPE_CHECKING:
    from server.core.agent import JimiAgent

logger = logging.getLogger(__name__)

# 旧危险 token 常量，skill_exec.py 仍 import
_DANGEROUS_PATTERNS = (
    "rm -rf /",
    "rm -rf /*",
    "rm -rf ~",
    "sudo",
    "mkfs",
    ":(){",  # fork bomb
    "dd if=",
    "> /dev/sda",
    "chmod -R 777 /",
    # shell 注入/组合字符
    ";",
    "&&",
    "||",
    "|",
    "`",
    "$(",
    ">",
    "<",
    "&",
)


# ================ helpers ================

def _safety_cfg(agent: "JimiAgent"):
    return getattr(agent.settings, "safety", None)


def _confirm_mode(agent: "JimiAgent") -> str:
    """返回实际生效的 confirm 策略：off / ui / llm。"""
    cfg = _safety_cfg(agent)
    if cfg is None:
        return "llm"
    mode = (cfg.confirm_mode or "auto").lower()
    # "auto" 默认走 ui（LangGraph interrupt）；不处理方需用 agent.resume() 恢复
    if mode == "auto":
        return "ui"
    if mode not in ("off", "ui", "llm"):
        return "ui"
    return mode


def _ui_interrupt_confirm(payload: dict) -> bool:
    """调用 LangGraph interrupt 暂停 graph，等 UI 层 resume。

    resume value 约定：
    - True / "approve" / "yes" → 批准
    - 其它 → 拒绝
    只能在 graph 执行上下文（tool 内部）调用；
    外部直接调会 RuntimeError。
    """
    try:
        from langgraph.types import interrupt
    except ImportError:
        # langgraph 版本过低，退化为拒绝
        logger.warning("langgraph.types.interrupt 不可用，视为拒绝")
        return False
    resp = interrupt(payload)
    if resp is True:
        return True
    if isinstance(resp, str):
        return resp.strip().lower() in ("true", "yes", "y", "approve", "ok")
    if isinstance(resp, dict):
        return bool(resp.get("approve") or resp.get("approved"))
    return False


def _workspace(agent: "JimiAgent") -> Path:
    return agent.settings.workspace_abs_path


def user_cwd() -> Path:
    """返回用户启动 `jimi chat` 时的 cwd（由 cli.py 透传 JIMI_USER_CWD）。

    若环境变量缺失（例如通过 `python cli.py start` 跑 gateway 或测试），
    则回退到进程 os.getcwd()。对 TUI 下的 `list_dir(".")` 与"当前目录"
    语义至关重要——否则 LLM 会把 workspace 误当作用户 CWD。
    """
    env = os.environ.get("JIMI_USER_CWD")
    if env:
        p = Path(env)
        if p.exists() and p.is_dir():
            return p.resolve()
    return Path(os.getcwd()).resolve()


def _extra_path_deny(agent: "JimiAgent") -> list[str]:
    cfg = _safety_cfg(agent)
    return list(cfg.extra_path_deny) if cfg else []


def _extra_cmd_deny(agent: "JimiAgent") -> list[str]:
    cfg = _safety_cfg(agent)
    return list(cfg.extra_cmd_deny) if cfg else []


def _extra_safe_heads(agent: "JimiAgent") -> list[str]:
    cfg = _safety_cfg(agent)
    return list(cfg.extra_safe_cmd_heads) if cfg else []


def _pending_confirm_message(summary: str, retry_hint: str) -> str:
    """返回给 LLM 的「需确认」提示。

    LLM 应把 summary 告诉用户、得到同意后带 confirm=True 重试。
    """
    return (
        f"[PENDING CONFIRM] {summary}\n"
        f"请把以上操作告知用户，等用户同意后使用相同参数并附加 `confirm=true` 重试。\n"
        f"若用户拒绝，请告知用户操作已取消。\n"
        f"重试示例: {retry_hint}"
    )


# ================ legacy fallback（safety disabled） ================

def _resolve_roots_legacy(agent: "JimiAgent") -> list[Path]:
    """safety 关闭时的 allowed roots（仅 workspace + extra）"""
    roots = [_workspace(agent)]
    extra = getattr(agent.settings, "_file_tools_roots", None)
    if extra:
        for r in extra:
            p = Path(r)
            if not p.is_absolute():
                from server.config.settings import PROJECT_ROOT
                p = PROJECT_ROOT / p
            roots.append(p.resolve())
    return roots


def _safe_resolve_legacy(path_str: str, roots: list[Path]) -> Path:
    """safety 关闭时的白名单校验（老逻辑）"""
    p = Path(path_str).expanduser().resolve()
    for root in roots:
        try:
            p.relative_to(root)
            return p
        except ValueError:
            continue
    raise PermissionError(
        f"路径 {path_str} 不在允许的目录内。"
        f"允许的根目录: {[str(r) for r in roots]}"
    )


# ================ 新架构：统一 resolve + 安全判定 ================

def _check_path_read(agent: "JimiAgent", path_str: str) -> tuple[bool, str, Optional[Path]]:
    """读路径检查。返回 (allowed, err_message, resolved_path)。"""
    cfg = _safety_cfg(agent)
    if cfg is None or not cfg.enabled:
        # legacy
        try:
            p = _safe_resolve_legacy(path_str, _resolve_roots_legacy(agent))
            return True, "", p
        except PermissionError as e:
            return False, str(e), None

    v = classify_path_read(path_str, extra_deny=_extra_path_deny(agent))
    if v.is_deny:
        return False, f"[REJECTED] {v.reason}", None
    return True, "", Path(path_str).expanduser().resolve()


def _check_path_write(
    agent: "JimiAgent", path_str: str, confirm: bool,
) -> tuple[str, str, Optional[Path]]:
    """写路径检查。返回 (status, message, resolved_path)。

    status:
      - "allow"  → 允许直接执行
      - "deny"   → 硬拒；message 为原因
      - "pending"→ 需 confirm；message 为给 LLM 的 PENDING 提示
    """
    cfg = _safety_cfg(agent)
    if cfg is None or not cfg.enabled:
        # legacy
        try:
            p = _safe_resolve_legacy(path_str, _resolve_roots_legacy(agent))
            return "allow", "", p
        except PermissionError as e:
            return "deny", str(e), None

    if _confirm_mode(agent) == "off":
        # 仅硬拒黑名单
        denied, reason = is_path_denied(path_str, _extra_path_deny(agent))
        if denied:
            return "deny", f"[REJECTED] {reason}", None
        return "allow", "", Path(path_str).expanduser().resolve()

    workspace = _workspace(agent) if cfg.auto_approve_workspace_writes else None
    v = classify_path_write(
        path_str, workspace=workspace,
        extra_deny=_extra_path_deny(agent),
    )
    resolved = Path(path_str).expanduser().resolve()
    if v.is_deny:
        return "deny", f"[REJECTED] {v.reason}", None
    if v.is_allow:
        return "allow", "", resolved
    # confirm required
    if confirm:
        return "allow", "", resolved
    return "pending", v.summary or f"写入 {resolved}", resolved


def _check_command(agent: "JimiAgent", command: str, confirm: bool) -> tuple[str, str]:
    """命令检查。返回 (status, message)。status 同 _check_path_write。"""
    cfg = _safety_cfg(agent)
    if cfg is None or not cfg.enabled:
        # legacy：只在 bash.enabled 路径里用，这里不会被调
        return "allow", ""

    if _confirm_mode(agent) == "off":
        # 仅硬拒黑名单
        from server.core.safety_fs import COMMAND_DENY_PATTERNS
        import re
        for pat in COMMAND_DENY_PATTERNS:
            if re.search(pat, command):
                return "deny", f"[REJECTED] 命令命中硬拒模式: {pat}"
        for pat in _extra_cmd_deny(agent):
            if re.search(pat, command):
                return "deny", f"[REJECTED] 命令命中用户 denylist: {pat}"
        return "allow", ""

    v = classify_command(
        command,
        extra_deny=_extra_cmd_deny(agent),
        extra_safe_heads=_extra_safe_heads(agent),
    )
    if v.is_deny:
        return "deny", f"[REJECTED] {v.reason}"
    if v.is_allow:
        return "allow", ""
    if confirm:
        return "allow", ""
    return "pending", v.summary or f"执行命令: {command}"


# ================ 文件工具 ================

def build_file_tools(agent: "JimiAgent") -> list[StructuredTool]:
    """构建文件操作工具集（全盘 + 安全层）"""

    def read_file(path: str, offset: int = 0, limit: int = 200) -> str:
        """读取指定文件的内容（文本文件）。

        支持全盘路径（如 /Users/xxx、~/Desktop/x）。系统保护路径会被拒绝。
        offset/limit 控制行范围。
        """
        ok, err, p = _check_path_read(agent, path)
        if not ok:
            return err
        if not p.exists():
            return f"文件不存在: {path}"
        if not p.is_file():
            return f"{path} 不是文件"
        try:
            text = p.read_text("utf-8")
        except UnicodeDecodeError:
            return f"{path} 不是文本文件（无法 UTF-8 解码）"
        lines = text.splitlines()
        selected = lines[offset: offset + limit]
        header = (
            f"文件: {p}（{len(lines)} 行，显示 "
            f"{offset+1}~{offset+len(selected)}）\n"
        )
        return header + "\n".join(
            f"{i+offset+1:4d}| {line}" for i, line in enumerate(selected)
        )

    def write_file(path: str, content: str, confirm: bool = False) -> str:
        """写入内容到指定文件（会覆盖原有内容）。目录不存在会自动创建。

        支持全盘路径。workspace 内写入无需 confirm；workspace 外默认触发用户
        confirm（UI 或 LLM 模式，取决于 safety.confirm_mode）；系统保护路径硬拒。
        """
        status, msg, p = _check_path_write(agent, path, confirm=confirm)
        if status == "deny":
            return msg
        if status == "pending":
            # UI 模式：调 interrupt 等用户点击确认
            if _confirm_mode(agent) == "ui":
                approved = _ui_interrupt_confirm({
                    "kind": "confirm",
                    "tool": "write_file",
                    "summary": msg,
                    "detail": {
                        "path": str(p),
                        "size": len(content),
                        "preview": content[:200],
                    },
                })
                if not approved:
                    return "[CANCELLED] 用户取消写入"
                # 批准 → 落盘
            else:
                # LLM 中转模式：返回 PENDING 让 LLM 询问用户
                retry = f"write_file(path={path!r}, content=..., confirm=True)"
                return _pending_confirm_message(msg, retry)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, "utf-8")
        return f"已写入 {p}（{len(content)} 字符）"

    def edit_file(
        path: str, old_text: str, new_text: str, confirm: bool = False,
    ) -> str:
        """对指定文件做一次精确替换（查找 old_text 替换为 new_text）。

        要求 old_text 在文件中仅出现一次；否则返回错误提示。支持全盘路径。
        workspace 外编辑需 confirm=true。
        """
        status, msg, p = _check_path_write(agent, path, confirm=confirm)
        if status == "deny":
            return msg
        if status == "pending":
            if _confirm_mode(agent) == "ui":
                approved = _ui_interrupt_confirm({
                    "kind": "confirm",
                    "tool": "edit_file",
                    "summary": msg,
                    "detail": {
                        "path": str(p),
                        "old_preview": old_text[:120],
                        "new_preview": new_text[:120],
                    },
                })
                if not approved:
                    return "[CANCELLED] 用户取消编辑"
            else:
                retry = f"edit_file(path={path!r}, old_text=..., new_text=..., confirm=True)"
                return _pending_confirm_message(msg, retry)
        if not p.exists() or not p.is_file():
            return f"文件不存在或不是普通文件: {path}"
        try:
            text = p.read_text("utf-8")
        except UnicodeDecodeError:
            return f"{path} 不是文本文件，无法 edit"
        count = text.count(old_text)
        if count == 0:
            return f"未找到匹配片段（长度 {len(old_text)} 字符）"
        if count > 1:
            return (
                f"old_text 出现 {count} 次，无法确定目标。"
                "请提供更多上下文让它唯一。"
            )
        new_content = text.replace(old_text, new_text, 1)
        p.write_text(new_content, "utf-8")
        return f"已更新 {p}（+{len(new_text) - len(old_text)} 字符）"

    def list_dir(path: str = ".", max_items: int = 50) -> str:
        """列出目录中的文件和子目录。

        - `path="."` 或不传：默认"用户当前目录"（启动 `jimi chat` 的 cwd，
          由 JIMI_USER_CWD 透传），对应用户直觉；不是 workspace！
        - 其它：支持全盘路径，受 safety 规则限制。
        """
        if path in (".", "", None):
            p = user_cwd()
        else:
            ok, err, p = _check_path_read(agent, path)
            if not ok:
                return err
        if not p.exists():
            return f"目录不存在: {path}"
        if not p.is_dir():
            return f"{path} 不是目录"

        items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines = [f"目录: {p}（共 {len(items)} 项）"]
        for item in items[:max_items]:
            if item.is_dir():
                try:
                    count = sum(1 for _ in item.iterdir())
                except Exception:
                    count = "?"
                lines.append(f"  [DIR]  {item.name}/ ({count} items)")
            else:
                try:
                    size = item.stat().st_size
                except Exception:
                    size = 0
                lines.append(f"  [FILE] {item.name} ({size} bytes)")
        if len(items) > max_items:
            lines.append(f"  ... 还有 {len(items) - max_items} 项未显示")
        return "\n".join(lines)

    def get_cwd() -> str:
        """返回用户的当前工作目录。

        即启动 `jimi chat` 时所在的目录。用户说「当前文件夹」「这里」「此目录」
        时都指这个，**不是 workspace**。在不确定路径时优先调用本工具确认。
        """
        return str(user_cwd())

    return [
        StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description=(
                "读取指定路径的文本文件内容（支持行偏移和限制）。"
                "支持全盘路径；系统保护目录（/etc、~/.ssh 等）会被拒绝。"
            ),
        ),
        StructuredTool.from_function(
            func=write_file,
            name="write_file",
            description=(
                "写入内容到指定文件（覆盖，自动建目录）。支持全盘路径。"
                "workspace 外写入会先返回 [PENDING CONFIRM]，需先询问用户同意、"
                "再传 confirm=true 重试。系统保护路径硬拒。"
            ),
        ),
        StructuredTool.from_function(
            func=edit_file,
            name="edit_file",
            description=(
                "对文件做一次精确替换：把 old_text 替换为 new_text。"
                "要求 old_text 在文件中唯一出现。支持全盘路径。"
                "workspace 外编辑需 confirm=true。"
            ),
        ),
        StructuredTool.from_function(
            func=list_dir,
            name="list_dir",
            description=(
                "列出目录中的文件和子目录。"
                "**path='.' 或不传 = 用户当前目录**（启动 jimi 时的 cwd）；"
                "可传任意路径。系统保护目录会被拒绝。"
            ),
        ),
        StructuredTool.from_function(
            func=get_cwd,
            name="get_cwd",
            description=(
                "返回用户的当前工作目录（启动 jimi chat 时的 cwd）。"
                "用户说「当前文件夹」「这里」「此目录」时指这个，不是 workspace。"
            ),
        ),
    ]


# ================ bash 工具 ================

def build_bash_tools(agent: "JimiAgent") -> list[StructuredTool]:
    """构建 bash 工具。bash.enabled 必须打开。"""
    bash_cfg = getattr(agent.settings, "bash", None)
    if bash_cfg is None or not bash_cfg.enabled:
        return []

    default_cwd = _workspace(agent)
    safety_on = _safety_cfg(agent) is not None and _safety_cfg(agent).enabled

    def _run_blocking(cmd: str, cwd: Path) -> tuple[int, str, str]:
        import subprocess
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(cwd),
                capture_output=True,
                text=True,
                timeout=bash_cfg.timeout_seconds,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"[TIMEOUT after {bash_cfg.timeout_seconds}s]"
        except Exception as e:
            return -2, "", f"[ERROR: {type(e).__name__}: {e}]"

    def _truncate(text: str) -> str:
        if len(text) <= bash_cfg.max_output_bytes:
            return text
        return (
            text[: bash_cfg.max_output_bytes]
            + f"\n...[truncated, {len(text)} bytes total]"
        )

    # legacy allowlist（safety 关闭时用）
    def _in_allowlist(cmd: str) -> bool:
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            return False
        if not tokens:
            return False
        head = Path(tokens[0]).name
        return head in bash_cfg.allowlist

    def _is_legacy_dangerous(cmd: str) -> bool:
        low = cmd.strip().lower()
        return any(pat in low for pat in _DANGEROUS_PATTERNS)

    def _is_read_only(cmd: str) -> bool:
        s = cmd.strip()
        return any(s.startswith(pref) for pref in bash_cfg.read_only_prefixes)

    def bash(command: str, confirm: bool = False, cwd: str = "") -> str:
        """执行一条 shell 命令。

        - 读类命令（ls/cat/pwd/git status 等）直接执行
        - 写类 / 未知命令需 confirm=true；先告知用户后重试
        - 毁灭性命令（rm -rf /、mkfs、dd 等）硬拒
        - cwd 可指定工作目录（默认 workspace）；若指向系统保护目录会被拒绝
        """
        # cwd 校验
        work_dir = default_cwd
        if cwd:
            ok, err, p = _check_path_read(agent, cwd)
            if not ok:
                return f"[REJECTED cwd] {err}"
            if not p.is_dir():
                return f"cwd 不是目录: {cwd}"
            work_dir = p

        if safety_on:
            # 新 safety_fs 判定
            status, msg = _check_command(agent, command, confirm=confirm)
            if status == "deny":
                return msg
            if status == "pending":
                if _confirm_mode(agent) == "ui":
                    approved = _ui_interrupt_confirm({
                        "kind": "confirm",
                        "tool": "bash",
                        "summary": msg,
                        "detail": {
                            "command": command,
                            "cwd": str(work_dir),
                        },
                    })
                    if not approved:
                        return "[CANCELLED] 用户取消命令执行"
                else:
                    retry = f"bash(command={command!r}, confirm=True)"
                    return _pending_confirm_message(msg, retry)
            # allow
        else:
            # 兼容老 allowlist 模式
            if _is_legacy_dangerous(command):
                return f"[REJECTED] 命令含危险模式: {command!r}"
            if not _in_allowlist(command):
                return (
                    f"[REJECTED] 命令不在 allowlist 中。当前 allowlist: "
                    f"{bash_cfg.allowlist}"
                )
            allow_run = _is_read_only(command) or confirm
            if not allow_run:
                return (
                    f"[PENDING CONFIRM] 该命令具有副作用，"
                    f"请再次调用并传 confirm=true。\n命令: {command!r}"
                )

        rc, out, err = _run_blocking(command, work_dir)
        return (
            f"[cwd: {work_dir}]\n"
            f"exit={rc}\n"
            f"--- stdout ---\n{_truncate(out)}\n"
            f"--- stderr ---\n{_truncate(err)}"
        )

    def process_list(limit: int = 20) -> str:
        """列出当前系统进程（安全只读）。"""
        rc, out, _err = _run_blocking("ps -ef", default_cwd)
        if rc != 0:
            return "[无法获取进程列表]"
        lines = out.splitlines()[: int(limit) + 1]
        return "\n".join(lines)

    desc = (
        "执行 shell 命令。支持全盘操作。"
        "读类命令直接执行；写/未知命令需先告知用户并 confirm=true 重试。"
        "毁灭性命令（rm -rf /、mkfs 等）硬拒。"
        "cwd 可选，默认 workspace。"
        if safety_on
        else
        "在 workspace 中执行 shell 命令。受 allowlist 限制，"
        "写类命令需 confirm=true。危险模式直接拒绝。"
    )
    return [
        StructuredTool.from_function(
            func=bash, name="bash", description=desc,
        ),
        StructuredTool.from_function(
            func=process_list,
            name="process_list",
            description="列出当前系统进程（只读快照）。",
        ),
    ]
