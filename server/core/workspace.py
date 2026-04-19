'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/server/core/workspace.py
Description: Workspace 管理器。

'''
from pathlib import Path
from typing import Optional

from server.config.settings import get_settings


# Workspace 核心 Markdown 文件
PROMPT_FILES = [
    ("SOUL.md", "人格定义"),
    ("AGENTS.md", "行为规则"),
    ("TOOLS.md", "环境信息"),
    ("USER.md", "用户上下文"),
]


class Workspace:
    """管理 workspace 配置文件和 skills。"""

    def __init__(self, workspace_path: Optional[Path] = None):
        if workspace_path is None:
            workspace_path = get_settings().workspace_abs_path
        self.path = Path(workspace_path)
        self._prompt_cache: Optional[str] = None

    def _read_file(self, filename: str) -> str:
        """读取 workspace 中的文件"""
        filepath = self.path / filename
        if filepath.exists():
            return filepath.read_text(encoding="utf-8").strip()
        return ""

    def build_system_prompt(self) -> str:
        """按顺序组装完整的 system prompt。"""
        if self._prompt_cache is not None:
            return self._prompt_cache

        sections = []
        for filename, label in PROMPT_FILES:
            content = self._read_file(filename)
            if content:
                sections.append(f"<!-- {label}: {filename} -->\n{content}")

        self._prompt_cache = "\n\n---\n\n".join(sections)
        return self._prompt_cache

    def reload(self):
        """清除缓存，重新加载文件"""
        self._prompt_cache = None

    def get_skills_dir(self) -> Path:
        """返回 skills 目录路径"""
        skills_dir = self.path / "skills"
        skills_dir.mkdir(parents=True, exist_ok=True)
        return skills_dir

    def list_skills(self) -> list[str]:
        """列出所有已安装的 skill 名称"""
        skills_dir = self.get_skills_dir()
        return [
            d.name for d in skills_dir.iterdir()
            if d.is_dir() and (d / "SKILL.md").exists()
        ]

    def get_skill_path(self, skill_name: str) -> Optional[Path]:
        """获取指定 skill 的路径"""
        skill_path = self.get_skills_dir() / skill_name
        if skill_path.exists() and (skill_path / "SKILL.md").exists():
            return skill_path
        return None
