'''
Author: JimZhang
LastEditors: Cascade
LastEditTime: 2026-04-21
FilePath: /JimiAgent/workspace/skills/shell_exec/shell.py
Description: Shell Skill。

与 builtin `bash` 工具等价；safety 关闭时退回旧黑名单。
'''
import asyncio
import re

# 保留老常量，向后兼容其它 import 方
BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/\s*$",
    r"rm\s+-rf\s+/\*",
    r"mkfs\.",
    r"dd\s+if=",
    r":\(\)\s*\{",
    r">\s*/dev/sd",
    r"chmod\s+-R\s+777\s+/\s*$",
]

MAX_OUTPUT_LENGTH = 5000
COMMAND_TIMEOUT = 30


def _load_settings():
    from server.config.settings import get_settings
    return get_settings()


def _is_safety_on() -> bool:
    try:
        cfg = getattr(_load_settings(), "safety", None)
        return bool(cfg and cfg.enabled)
    except Exception:
        return False


def _pending_msg(summary: str, retry_hint: str) -> str:
    return (
        f"[PENDING CONFIRM] {summary}\n"
        f"请把以上操作告知用户，得到同意后使用相同参数并附加 `confirm=true` 重试。\n"
        f"若用户拒绝，请告知用户操作已取消。\n"
        f"重试示例: {retry_hint}"
    )


def is_command_safe(command: str) -> tuple[bool, str]:
    """legacy 黑名单：safety 关闭时用。"""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return False, f"命令被阻止: 匹配危险模式 '{pattern}'"
    return True, ""


async def execute(command: str, confirm: bool = False) -> str:
    """执行 Shell 命令并返回输出。

    safety=on 时走分类器；safety=off 时退化黑名单。
    """
    if _is_safety_on():
        from server.core.safety_fs import classify_command
        s = _load_settings()
        v = classify_command(
            command,
            extra_deny=list(s.safety.extra_cmd_deny),
            extra_safe_heads=list(s.safety.extra_safe_cmd_heads),
        )
        if v.is_deny:
            return f"[REJECTED] {v.reason}"
        if v.needs_confirm and not confirm:
            retry = f"execute(command={command!r}, confirm=True)"
            return _pending_msg(v.summary, retry)
    else:
        is_safe, reason = is_command_safe(command)
        if not is_safe:
            return f"[blocked] {reason}"

    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=COMMAND_TIMEOUT,
        )

        output_parts = []
        if stdout:
            decoded = stdout.decode("utf-8", errors="replace").strip()
            if decoded:
                output_parts.append(decoded)
        if stderr:
            decoded = stderr.decode("utf-8", errors="replace").strip()
            if decoded:
                output_parts.append(f"[STDERR]\n{decoded}")

        output = "\n".join(output_parts) if output_parts else "(无输出)"

        if len(output) > MAX_OUTPUT_LENGTH:
            output = (
                output[:MAX_OUTPUT_LENGTH]
                + f"\n\n... [输出已截断, 总长度: {len(output)} 字符]"
            )

        exit_info = f"[退出码: {process.returncode}]"
        return f"{output}\n{exit_info}"

    except asyncio.TimeoutError:
        return f"[timeout] 命令执行超时 ({COMMAND_TIMEOUT}秒): {command}"
    except Exception as e:
        return f"[error] 命令执行出错: {str(e)}"
