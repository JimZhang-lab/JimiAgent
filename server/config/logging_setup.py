'''
Author: JimZhang
Date: 2026-04-19 15:00:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 15:00:00
FilePath: /JimiAgent/server/config/logging_setup.py
Description: logging 初始化工具。

'''
from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path
from typing import Optional

from server.config.settings import LoggingConfig, Settings


_PLAIN_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# 压低第三方日志噪音
_NOISY_LOGGERS = ("httpx", "httpcore", "openai", "urllib3", "websockets")


def _make_console_handler(fmt_mode: str) -> logging.Handler:
    """构造控制台 handler。"""
    if fmt_mode == "rich":
        try:
            from rich.logging import RichHandler

            handler = RichHandler(
                rich_tracebacks=True,
                show_time=True,
                show_level=True,
                show_path=False,
                markup=False,
            )
            # RichHandler 自带时间和级别
            handler.setFormatter(logging.Formatter("%(message)s"))
            return handler
        except Exception:
            # rich 不可用则退化为 plain
            pass

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(_PLAIN_FORMAT, datefmt=_DATE_FMT))
    return handler


def _make_file_handler(
    file_path: Path,
    rotate_mb: int,
    backup_count: int,
) -> logging.Handler:
    """构造轮转文件 handler。"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        filename=str(file_path),
        mode="a",
        maxBytes=max(1, rotate_mb) * 1024 * 1024,
        backupCount=max(0, backup_count),
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(_PLAIN_FORMAT, datefmt=_DATE_FMT))
    return handler


def setup_logging(
    cfg: LoggingConfig,
    log_file_abs: Optional[Path] = None,
) -> None:
    """按 LoggingConfig 初始化 logging。"""
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    root = logging.getLogger()
    # 清理旧 handler，避免重复输出
    for h in list(root.handlers):
        root.removeHandler(h)
        try:
            h.close()
        except Exception:
            pass
    root.setLevel(level)

    output = (cfg.output or "console").lower()
    if output not in ("console", "file", "both"):
        output = "console"

    if output in ("console", "both"):
        root.addHandler(_make_console_handler(cfg.format or "rich"))

    if output in ("file", "both"):
        file_path = log_file_abs or Path(cfg.file_path)
        try:
            root.addHandler(
                _make_file_handler(file_path, cfg.rotate_mb, cfg.backup_count)
            )
        except Exception as e:
            # 文件 handler 失败时回退控制台
            fallback = _make_console_handler(cfg.format or "plain")
            root.addHandler(fallback)
            root.warning(
                f"无法创建日志文件 {file_path} ({e})，回退到控制台输出"
            )

    # 降低第三方库噪音
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    root.debug(
        f"日志已初始化: level={cfg.level} output={output} "
        f"format={cfg.format} file={cfg.file_path}"
    )


def setup_from_settings(settings: Settings) -> None:
    """从 Settings 初始化日志。"""
    setup_logging(settings.logging, log_file_abs=settings.log_file_abs_path)
