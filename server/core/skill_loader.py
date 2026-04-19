'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/server/core/skill_loader.py
Description: Skill 加载器。

'''
import importlib.util
import re
import asyncio
from pathlib import Path
from typing import Any, Callable, Optional
from dataclasses import dataclass, field

import yaml
from langchain_core.tools import tool, StructuredTool


@dataclass
class SkillMeta:
    """从 SKILL.md frontmatter 解析出的 Skill 元信息。"""
    name: str
    description: str
    version: str = "1.0"
    dependencies: list[str] = field(default_factory=list)
    instructions: str = ""              # SKILL.md 正文指令
    script_path: Optional[Path] = None  # 关联的 Python 脚本
    skill_dir: Optional[Path] = None

    # ---- OpenClaw metadata ----
    requires_env: list[str] = field(default_factory=list)   # requires.env
    requires_bins: list[str] = field(default_factory=list)  # requires.bins
    requires_any_bins: list[str] = field(default_factory=list)  # requires.anyBins
    requires_config: list[str] = field(default_factory=list)  # requires.config
    primary_env: Optional[str] = None
    always: bool = False
    skill_key: Optional[str] = None
    emoji: Optional[str] = None
    homepage: Optional[str] = None
    os_list: list[str] = field(default_factory=list)  # ["macos","linux"]
    install_specs: list[dict] = field(default_factory=list)  # [{kind,formula,bins,...}]

    # ---- exec runtime ----
    exec_command: Optional[str] = None       # bash|node|python|go|deno|…
    exec_args: list[str] = field(default_factory=list)
    exec_env: dict = field(default_factory=dict)
    exec_timeout: int = 10                   # 秒
    # 原始 frontmatter（供高级场景/调试；不做契约）
    raw_frontmatter: dict = field(default_factory=dict)


# metadata.openclaw 别名
_OPENCLAW_META_ALIASES = ("openclaw", "clawdbot", "clawdis")


def _extract_openclaw_meta(frontmatter: dict) -> dict:
    """提取 metadata.<alias> 配置。"""
    meta = frontmatter.get("metadata")
    if not isinstance(meta, dict):
        return {}
    for key in _OPENCLAW_META_ALIASES:
        block = meta.get(key)
        if isinstance(block, dict):
            return block
    return {}


def parse_skill_md(skill_dir: Path) -> Optional[SkillMeta]:
    """解析 SKILL.md，返回 frontmatter 和正文指令。"""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        # 兼容小写 skill.md
        alt = skill_dir / "skill.md"
        if alt.exists():
            skill_file = alt
        else:
            return None

    content = skill_file.read_text(encoding="utf-8")

    # 解析 YAML frontmatter
    frontmatter: dict = {}
    instructions = content
    fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
    if fm_match:
        try:
            parsed = yaml.safe_load(fm_match.group(1))
            if isinstance(parsed, dict):
                frontmatter = parsed
        except yaml.YAMLError:
            pass
        instructions = fm_match.group(2).strip()

    name = frontmatter.get("name", skill_dir.name)
    description = frontmatter.get("description", f"Skill: {name}")

    # 查找关联 Python 脚本
    script_path = None
    for py_file in skill_dir.glob("*.py"):
        if py_file.name != "__init__.py":
            script_path = py_file
            break

    # ---- 解析 OpenClaw metadata ----
    ocmeta = _extract_openclaw_meta(frontmatter)
    requires = ocmeta.get("requires") or {}
    if not isinstance(requires, dict):
        requires = {}

    def _as_str_list(v) -> list[str]:
        if isinstance(v, list):
            return [str(x) for x in v if x is not None]
        if isinstance(v, str) and v:
            return [v]
        return []

    exec_block = ocmeta.get("exec") or {}
    if not isinstance(exec_block, dict):
        exec_block = {}

    return SkillMeta(
        name=name,
        description=description,
        version=str(frontmatter.get("version", "1.0")),
        dependencies=_as_str_list(frontmatter.get("dependencies")),
        instructions=instructions,
        script_path=script_path,
        skill_dir=skill_dir,
        # OpenClaw runtime metadata
        requires_env=_as_str_list(requires.get("env")),
        requires_bins=_as_str_list(requires.get("bins")),
        requires_any_bins=_as_str_list(requires.get("anyBins")),
        requires_config=_as_str_list(requires.get("config")),
        primary_env=(ocmeta.get("primaryEnv") or None),
        always=bool(ocmeta.get("always", False)),
        skill_key=(ocmeta.get("skillKey") or None),
        emoji=(ocmeta.get("emoji") or None),
        homepage=(ocmeta.get("homepage") or None),
        os_list=_as_str_list(ocmeta.get("os")),
        install_specs=[
            s for s in (ocmeta.get("install") or [])
            if isinstance(s, dict)
        ],
        exec_command=(exec_block.get("command") or None),
        exec_args=_as_str_list(exec_block.get("args")),
        exec_env=(
            exec_block.get("env")
            if isinstance(exec_block.get("env"), dict) else {}
        ),
        exec_timeout=int(
            exec_block.get("timeoutSeconds")
            or exec_block.get("timeout")
            or 10
        ),
        raw_frontmatter=frontmatter,
    )


# ---- 环境检查 ----

def diagnose_skill(skill: SkillMeta) -> list[str]:
    """检查 skill 依赖，返回告警列表。"""
    import os as _os
    import shutil as _sh
    import sys as _sys
    warnings: list[str] = []

    for ev in skill.requires_env:
        if not _os.environ.get(ev):
            warnings.append(f"环境变量 {ev} 未设置")

    for b in skill.requires_bins:
        if not _sh.which(b):
            warnings.append(f"命令 `{b}` 不在 PATH")

    if skill.requires_any_bins:
        if not any(_sh.which(b) for b in skill.requires_any_bins):
            warnings.append(
                f"anyBins {skill.requires_any_bins} 全部不在 PATH"
            )

    if skill.os_list:
        plat = _sys.platform
        cur = (
            "macos" if plat == "darwin"
            else "linux" if plat.startswith("linux")
            else "windows" if plat.startswith("win")
            else plat
        )
        if cur not in skill.os_list:
            warnings.append(
                f"OS {cur} 不在 skill 支持列表 {skill.os_list}"
            )

    return warnings


def _load_script_functions(script_path: Path) -> dict[str, Callable]:
    """动态加载 Python 脚本中的函数"""
    spec = importlib.util.spec_from_file_location(
        f"skill_script_{script_path.stem}", script_path
    )
    if spec is None or spec.loader is None:
        return {}

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    functions = {}
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        attr = getattr(module, attr_name)
        if callable(attr) and hasattr(attr, "__code__"):
            functions[attr_name] = attr
    return functions


def build_tool_from_skill(skill: SkillMeta) -> list[StructuredTool]:
    """
    将一个 Skill 转换为 LangChain Tool(s)

    优先级：
    1. 声明了 `metadata.openclaw.exec` → 走子进程 runtime（K2）
    2. 有 `*.py` → 动态导入所有公共函数（老行为）
    3. 纯指令型 → 返回指令内容的信息 tool
    """
    tools: list[StructuredTool] = []

    # K2: exec runtime 优先
    if skill.exec_command:
        from server.core.skill_exec import build_exec_tool
        t = build_exec_tool(skill)
        if t is not None:
            tools.append(t)
            return tools
        # exec tool 构建失败时回退

    if skill.script_path and skill.script_path.exists():
        # 有脚本时为公共函数生成 Tool
        functions = _load_script_functions(skill.script_path)
        for func_name, func in functions.items():
            tool_name = f"{skill.name}_{func_name}" if len(functions) > 1 else skill.name
            doc = func.__doc__ or skill.description

            if asyncio.iscoroutinefunction(func):
                t = StructuredTool.from_function(
                    coroutine=func,
                    name=tool_name,
                    description=doc.strip(),
                )
            else:
                t = StructuredTool.from_function(
                    func=func,
                    name=tool_name,
                    description=doc.strip(),
                )
            tools.append(t)
    else:
        # 纯指令型 Skill 返回信息 Tool
        instructions = skill.instructions

        def _info_tool() -> str:
            return instructions

        t = StructuredTool.from_function(
            func=_info_tool,
            name=skill.name,
            description=skill.description,
        )
        tools.append(t)

    return tools


def load_all_skills(skills_dir: Path) -> tuple[list[SkillMeta], list[StructuredTool]]:
    """加载指定目录下的所有 Skills"""
    all_metas = []
    all_tools = []

    if not skills_dir.exists():
        return all_metas, all_tools

    for skill_subdir in sorted(skills_dir.iterdir()):
        if not skill_subdir.is_dir():
            continue

        meta = parse_skill_md(skill_subdir)
        if meta is None:
            continue

        all_metas.append(meta)
        tools = build_tool_from_skill(meta)
        all_tools.extend(tools)

    return all_metas, all_tools
