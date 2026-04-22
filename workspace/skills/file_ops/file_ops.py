'''
Author: JimZhang
LastEditors: Cascade
LastEditTime: 2026-04-21
FilePath: /JimiAgent/workspace/skills/file_ops/file_ops.py
Description: 文件操作 Skill。

与 builtin 文件工具等价，保留 skill 形态兼容旧 workflow。
'''
from pathlib import Path


def _load_settings():
    """延迟 import，避免 skill 加载时触发循环依赖。"""
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


def read_file(file_path: str) -> str:
    """读取文件内容（任意路径；系统保护目录会被拒绝）。"""
    if _is_safety_on():
        from server.core.safety_fs import classify_path_read
        s = _load_settings()
        v = classify_path_read(
            file_path,
            extra_deny=list(s.safety.extra_path_deny),
        )
        if v.is_deny:
            return f"[REJECTED] {v.reason}"

    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        return f"文件不存在: {p}"
    if not p.is_file():
        return f"路径不是文件: {p}"
    try:
        content = p.read_text(encoding="utf-8")
        if len(content) > 10000:
            return (
                f"文件内容（前10000字符）:\n{content[:10000]}\n\n"
                f"... [文件已截断, 总长度: {len(content)} 字符]"
            )
        return content
    except Exception as e:
        return f"读取文件出错: {str(e)}"


def write_file(file_path: str, content: str, confirm: bool = False) -> str:
    """写入内容到文件（覆盖；自动建目录）。

    workspace 内写入直接放行；workspace 外需 confirm=true。
    系统保护路径硬拒。
    """
    if _is_safety_on():
        from server.core.safety_fs import classify_path_write
        s = _load_settings()
        workspace = (
            s.workspace_abs_path
            if s.safety.auto_approve_workspace_writes
            else None
        )
        v = classify_path_write(
            file_path,
            workspace=workspace,
            extra_deny=list(s.safety.extra_path_deny),
        )
        if v.is_deny:
            return f"[REJECTED] {v.reason}"
        if v.needs_confirm and not confirm:
            retry = f"write_file(file_path={file_path!r}, content=..., confirm=True)"
            return _pending_msg(v.summary, retry)

    p = Path(file_path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"已成功写入文件: {p} ({len(content)} 字符)"
    except Exception as e:
        return f"写入文件出错: {str(e)}"


def list_directory(dir_path: str = ".") -> str:
    """列出目录内容（任意路径；系统保护目录会被拒绝）。"""
    if _is_safety_on():
        from server.core.safety_fs import classify_path_read
        s = _load_settings()
        v = classify_path_read(
            dir_path,
            extra_deny=list(s.safety.extra_path_deny),
        )
        if v.is_deny:
            return f"[REJECTED] {v.reason}"

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
