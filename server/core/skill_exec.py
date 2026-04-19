'''K2 - Exec Skill 运行时。'''
import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import StructuredTool

from server.core.file_tools import _DANGEROUS_PATTERNS

if TYPE_CHECKING:
    from server.core.skill_loader import SkillMeta

logger = logging.getLogger(__name__)


# 预设命令白名单；value 为附加前缀。
_EXEC_PRESETS: dict[str, list[str]] = {
    "bash": [],
    "sh": [],
    "python": [],
    "python3": [],
    "node": [],
    "deno": [],
    "go": ["run"],
    "ruby": [],
    "perl": [],
    "pwsh": [],
    "powershell": [],
}

_OUTPUT_TRUNC = 64 * 1024  # 64 KB
_DEFAULT_TIMEOUT = 10


def _scan_dangerous(args: list[str]) -> list[str]:
    """扫 args 中是否混入 _DANGEROUS_PATTERNS；返回命中列表。"""
    hits: list[str] = []
    for a in args:
        for pat in _DANGEROUS_PATTERNS:
            if pat and pat in a:
                hits.append(f"参数 {a!r} 命中危险模式 {pat!r}")
    return hits


async def _run_subprocess(
    cmd: str,
    args: list[str],
    cwd: Path,
    env: dict,
    timeout: int,
) -> str:
    """受控子进程执行。返回合并后的 stdout/stderr（已截断）。"""
    # 先继承父 env，再用 skill 覆盖
    merged_env = dict(os.environ)
    merged_env.update({str(k): str(v) for k, v in (env or {}).items()})

    try:
        proc = await asyncio.create_subprocess_exec(
            cmd,
            *args,
            cwd=str(cwd),
            env=merged_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError:
        return f"(错误: 命令 `{cmd}` 不在 PATH)"
    except Exception as e:
        return f"(子进程启动失败: {type(e).__name__}: {e})"

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except Exception:
            pass
        return f"(超时: skill 运行超过 {timeout} 秒)"

    out = (stdout or b"").decode("utf-8", errors="replace")
    err = (stderr or b"").decode("utf-8", errors="replace")
    combined = out
    if err.strip():
        combined += f"\n[stderr]\n{err}"

    if len(combined) > _OUTPUT_TRUNC:
        combined = combined[:_OUTPUT_TRUNC] + f"\n…(已截断，共 {len(combined)} 字节)"

    if proc.returncode and proc.returncode != 0:
        combined += f"\n[exit code: {proc.returncode}]"

    return combined or "(skill 无输出)"


def build_exec_tool(skill: "SkillMeta") -> StructuredTool | None:
    """把声明了 `metadata.openclaw.exec` 的 skill 包成 StructuredTool"""
    cmd = (skill.exec_command or "").strip().lower()
    if not cmd:
        return None

    if cmd not in _EXEC_PRESETS:
        logger.warning(
            f"Skill {skill.name}: 不支持的 exec.command `{cmd}`，跳过"
        )
        return None

    # 尽量解析绝对路径；失败时保留原命令名
    resolved_cmd = shutil.which(cmd) or cmd

    # skill 目录作为 cwd
    skill_dir = skill.skill_dir
    if skill_dir is None or not skill_dir.exists():
        logger.warning(f"Skill {skill.name}: skill_dir 不存在，跳过 exec 工具")
        return None

    # 静态 args = preset + frontmatter
    preset = list(_EXEC_PRESETS[cmd])
    static_args = preset + list(skill.exec_args or [])

    # 静态 args 命中危险模式则拒绝注册
    hits = _scan_dangerous(static_args)
    if hits:
        logger.error(
            f"Skill {skill.name}: 静态 args 命中危险模式，拒绝注册: {hits}"
        )
        return None

    timeout = int(skill.exec_timeout or _DEFAULT_TIMEOUT)
    env = dict(skill.exec_env or {})

    description = (
        skill.description
        + "\n\n（exec skill：可选 input 会以 JSON 追加到 argv 末尾。）"
    )

    async def _exec_tool(input: str = "") -> str:
        """exec skill 运行时入口"""
        run_args = list(static_args)
        if input:
            # 作为单个 argv 追加，不再走 shell 解析
            run_args.append(input if isinstance(input, str) else json.dumps(input))
        # 运行前再扫一遍参数
        hits_rt = _scan_dangerous(run_args)
        if hits_rt:
            return "(拒绝执行：运行时参数命中危险模式: " + "; ".join(hits_rt) + ")"
        return await _run_subprocess(
            cmd=resolved_cmd,
            args=run_args,
            cwd=skill_dir,
            env=env,
            timeout=timeout,
        )

    return StructuredTool.from_function(
        coroutine=_exec_tool,
        name=skill.skill_key or skill.name,
        description=description,
    )
