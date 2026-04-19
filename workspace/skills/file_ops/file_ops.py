'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:25:00
FilePath: /JimiAgent/workspace/skills/file_ops/file_ops.py
Description: 文件操作 Skill。

'''
from pathlib import Path


def read_file(file_path: str) -> str:
    """读取文件内容"""
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        return f"文件不存在: {p}"
    if not p.is_file():
        return f"路径不是文件: {p}"
    try:
        content = p.read_text(encoding="utf-8")
        if len(content) > 10000:
            return f"文件内容（前10000字符）:\n{content[:10000]}\n\n... [文件已截断, 总长度: {len(content)} 字符]"
        return content
    except Exception as e:
        return f"读取文件出错: {str(e)}"


def write_file(file_path: str, content: str) -> str:
    """写入内容到文件"""
    p = Path(file_path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"已成功写入文件: {p} ({len(content)} 字符)"
    except Exception as e:
        return f"写入文件出错: {str(e)}"


def list_directory(dir_path: str = ".") -> str:
    """列出目录内容"""
    p = Path(dir_path).expanduser().resolve()
    if not p.exists():
        return f"目录不存在: {p}"
    if not p.is_dir():
        return f"路径不是目录: {p}"

    items = []
    try:
        for item in sorted(p.iterdir()):
            if item.name.startswith("."):
                continue
            prefix = "[dir]" if item.is_dir() else "[file]"
            size = ""
            if item.is_file():
                size_bytes = item.stat().st_size
                if size_bytes < 1024:
                    size = f" ({size_bytes} B)"
                elif size_bytes < 1024 * 1024:
                    size = f" ({size_bytes / 1024:.1f} KB)"
                else:
                    size = f" ({size_bytes / (1024 * 1024):.1f} MB)"
            items.append(f"{prefix} {item.name}{size}")

        if not items:
            return f"目录为空: {p}"
        return f"目录: {p}\n" + "\n".join(items)
    except Exception as e:
        return f"列出目录出错: {str(e)}"
