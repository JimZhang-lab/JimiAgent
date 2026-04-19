'''
Author: JimZhang
Date: 2026-04-19 18:10:00
LastEditors: JimZhang
LastEditTime: 2026-04-19 18:10:00
FilePath: /JimiAgent/server/core/file_tools.py
Description: 文件与 Shell 内置工具。
'''
import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

if TYPE_CHECKING:
    from server.core.agent import JimiAgent

logger = logging.getLogger(__name__)

# 危险 token
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
    "|",  # 禁管道，避免 `git log | sh`
    "`",
    "$(",
    ">",  # 重定向
    "<",
    "&",  # 后台执行
)


def _resolve_roots(agent: "JimiAgent") -> list[Path]:
    """从 settings 解析 allowed roots。"""
    roots = [agent.settings.workspace_abs_path]
    # 支持额外 roots
    extra = getattr(agent.settings, "_file_tools_roots", None)
    if extra:
        for r in extra:
            p = Path(r)
            if not p.is_absolute():
                from server.config.settings import PROJECT_ROOT
                p = PROJECT_ROOT / p
            roots.append(p.resolve())
    return roots


def _safe_resolve(path_str: str, roots: list[Path]) -> Path:
    """解析路径并校验白名单。"""
    p = Path(path_str).resolve()
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


def build_file_tools(agent: "JimiAgent") -> list[StructuredTool]:
    """构建文件操作工具集"""
    roots = _resolve_roots(agent)

    def read_file(path: str, offset: int = 0, limit: int = 200) -> str:
        """读取指定文件的内容（文本文件）。"""
        try:
            p = _safe_resolve(path, roots)
        except PermissionError as e:
            return str(e)
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
        header = f"文件: {p.name}（{len(lines)} 行，显示 {offset+1}~{offset+len(selected)}）\n"
        return header + "\n".join(
            f"{i+offset+1:4d}| {line}" for i, line in enumerate(selected)
        )

    def write_file(path: str, content: str) -> str:
        """写入内容到指定文件（会覆盖原有内容）。目录不存在会自动创建。"""
        try:
            p = _safe_resolve(path, roots)
        except PermissionError as e:
            return str(e)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, "utf-8")
        return f"已写入 {p}（{len(content)} 字符）"

    def edit_file(path: str, old_text: str, new_text: str) -> str:
        """对指定文件做一次精确替换（查找 old_text 替换为 new_text）。

        要求 old_text 在文件中仅出现一次；否则返回错误提示，
        让调用者补上更多上下文再重试（避免意外改到多处）。
        """
        try:
            p = _safe_resolve(path, roots)
        except PermissionError as e:
            return str(e)
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
        """列出目录中的文件和子目录。"""
        if path == ".":
            p = roots[0]  # 默认 workspace
        else:
            try:
                p = _safe_resolve(path, roots)
            except PermissionError as e:
                return str(e)
        if not p.exists():
            return f"目录不存在: {path}"
        if not p.is_dir():
            return f"{path} 不是目录"

        items = sorted(p.iterdir(), key=lambda x: (not x.is_dir(), x.name))
        lines = [f"目录: {p}（共 {len(items)} 项）"]
        for item in items[:max_items]:
            if item.is_dir():
                count = sum(1 for _ in item.iterdir()) if item.exists() else 0
                lines.append(f"  [DIR]  {item.name}/ ({count} items)")
            else:
                size = item.stat().st_size
                lines.append(f"  [FILE] {item.name} ({size} bytes)")
        if len(items) > max_items:
            lines.append(f"  ... 还有 {len(items) - max_items} 项未显示")
        return "\n".join(lines)

    return [
        StructuredTool.from_function(
            func=read_file,
            name="read_file",
            description=(
                "读取指定路径的文本文件内容（支持行偏移和限制）。"
                "用于查看 workspace 中的文件。"
            ),
        ),
        StructuredTool.from_function(
            func=write_file,
            name="write_file",
            description=(
                "写入内容到指定文件（覆盖）。"
                "用于创建或更新 workspace 中的文件。"
            ),
        ),
        StructuredTool.from_function(
            func=edit_file,
            name="edit_file",
            description=(
                "对文件做一次精确替换：把 old_text 替换为 new_text。"
                "要求 old_text 在文件中唯一出现；否则请带更多上下文重试。"
            ),
        ),
        StructuredTool.from_function(
            func=list_dir,
            name="list_dir",
            description=(
                "列出目录中的文件和子目录。默认列出 workspace 根目录。"
            ),
        ),
    ]


def build_bash_tools(agent: "JimiAgent") -> list[StructuredTool]:
    """构建 bash 工具。"""
    bash_cfg = getattr(agent.settings, "bash", None)
    if bash_cfg is None or not bash_cfg.enabled:
        return []

    workspace = agent.settings.workspace_abs_path

    def _is_dangerous(cmd: str) -> bool:
        low = cmd.strip().lower()
        return any(pat in low for pat in _DANGEROUS_PATTERNS)

    def _is_read_only(cmd: str) -> bool:
        s = cmd.strip()
        return any(s.startswith(pref) for pref in bash_cfg.read_only_prefixes)

    def _in_allowlist(cmd: str) -> bool:
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            return False
        if not tokens:
            return False
        # 取第一个 token 的 basename
        head = Path(tokens[0]).name
        return head in bash_cfg.allowlist

    def _get_confirm_mode(session_id: str) -> str:
        sess = agent.session_mgr.get_session(session_id)
        return (sess.metadata.get("bash_confirm", "on") if sess else "on")

    def _run_blocking(cmd: str) -> tuple[int, str, str]:
        """同步执行（在线程池内调用）"""
        import subprocess
        try:
            proc = subprocess.run(
                cmd,
                shell=True,
                cwd=str(workspace),
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
        return text[: bash_cfg.max_output_bytes] + f"\n...[truncated, {len(text)} bytes total]"

    def bash(command: str, confirm: bool = False) -> str:
        """在 workspace 内执行一条 shell 命令。"""
        # 同步工具里取不到 session_id，这里退回最近会话
        sessions = agent.session_mgr.list_sessions()
        confirm_mode = _get_confirm_mode(sessions[0].id) if sessions else "on"

        if _is_dangerous(command):
            return f"[REJECTED] 命令含危险模式，拒绝执行: {command!r}"
        if not _in_allowlist(command):
            return (
                f"[REJECTED] 命令不在 allowlist 中。当前 allowlist: "
                f"{bash_cfg.allowlist}"
            )
        # 只读或已确认时放行
        allow_run = (
            _is_read_only(command)
            or confirm is True
            or confirm_mode == "off"
        )
        if not allow_run:
            return (
                f"[PENDING CONFIRM] 该命令具有副作用，请再次调用并传 confirm=true。"
                f"\n命令: {command!r}"
            )

        # StructuredTool 会把阻塞调用丢到 executor
        rc, out, err = _run_blocking(command)
        return (
            f"exit={rc}\n"
            f"--- stdout ---\n{_truncate(out)}\n"
            f"--- stderr ---\n{_truncate(err)}"
        )

    def process_list(limit: int = 20) -> str:
        """列出当前系统进程（安全只读）。"""
        rc, out, _err = _run_blocking("ps -ef")
        if rc != 0:
            return "[无法获取进程列表]"
        lines = out.splitlines()[: int(limit) + 1]  # +1 包含表头
        return "\n".join(lines)

    return [
        StructuredTool.from_function(
            func=bash,
            name="bash",
            description=(
                "在 workspace 中执行 shell 命令。受 allowlist 限制，"
                "写类命令需 confirm=true。危险模式直接拒绝。"
            ),
        ),
        StructuredTool.from_function(
            func=process_list,
            name="process_list",
            description="列出当前系统进程（只读快照）。",
        ),
    ]
