'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:25:00
FilePath: /JimiAgent/workspace/skills/shell_exec/shell.py
Description: Shell Skill。

'''
import asyncio
import re

# 危险命令模式（绝对禁止）
BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/\s*$",
    r"rm\s+-rf\s+/\*",
    r"mkfs\.",
    r"dd\s+if=",
    r":\(\)\s*\{",
    r">\s*/dev/sd",
    r"chmod\s+-R\s+777\s+/\s*$",
]

# 输出截断长度
MAX_OUTPUT_LENGTH = 5000

# 命令超时（秒）
COMMAND_TIMEOUT = 30


def is_command_safe(command: str) -> tuple[bool, str]:
    """检查命令是否安全"""
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return False, f"命令被阻止: 匹配危险模式 '{pattern}'"
    return True, ""


async def execute(command: str) -> str:
    """执行 Shell 命令并返回输出"""
    # 安全检查
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

        # 截断过长输出
        if len(output) > MAX_OUTPUT_LENGTH:
            output = output[:MAX_OUTPUT_LENGTH] + f"\n\n... [输出已截断, 总长度: {len(output)} 字符]"

        exit_info = f"[退出码: {process.returncode}]"
        return f"{output}\n{exit_info}"

    except asyncio.TimeoutError:
        return f"[timeout] 命令执行超时 ({COMMAND_TIMEOUT}秒): {command}"
    except Exception as e:
        return f"[error] 命令执行出错: {str(e)}"
