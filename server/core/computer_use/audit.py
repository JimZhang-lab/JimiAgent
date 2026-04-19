"""Computer Use 审计日志。"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from server.config.settings import ComputerUseConfig

logger = logging.getLogger(__name__)


def _abs_path(project_root: Path, p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = project_root / path
    return path


class Auditor:
    """记录 Computer Use 动作和缩略图。"""

    def __init__(self, cfg: "ComputerUseConfig", project_root: Path):
        self.cfg = cfg
        self.root = project_root
        self._log_path = _abs_path(project_root, cfg.audit_log_path)
        self._thumb_dir = _abs_path(project_root, cfg.audit_thumbnail_dir)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        if cfg.audit_thumbnails:
            self._thumb_dir.mkdir(parents=True, exist_ok=True)

    # 事件写入

    def write(
        self,
        tool: str,
        args: dict,
        result: Any = None,
        error: Optional[str] = None,
        before_thumb: Optional[str] = None,
        after_thumb: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> None:
        """追加一条动作记录。"""
        rec = {
            "ts": _dt.datetime.now(_dt.timezone.utc).isoformat(),
            "tool": tool,
            "args": self._truncate_args(args),
            "ok": error is None,
        }
        if error:
            rec["error"] = error[:500]
        if result is not None:
            rec["result"] = self._truncate_repr(result)
        if duration_ms is not None:
            rec["duration_ms"] = round(duration_ms, 2)
        if before_thumb:
            rec["before_thumb"] = before_thumb
        if after_thumb:
            rec["after_thumb"] = after_thumb
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Computer Use 审计写入失败: {e}")

    # 缩略图

    def save_thumbnail(self, image, tag: str = "snap") -> Optional[str]:
        """保存缩略图并返回相对路径。"""
        if not self.cfg.audit_thumbnails:
            return None
        try:
            from PIL import Image  # Pillow（已是依赖）
        except ImportError:
            return None
        try:
            img = image if hasattr(image, "save") else Image.open(str(image))
            img = img.copy()
            img.thumbnail((320, 320))
            stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            fname = f"{stamp}_{tag}.png"
            dest = self._thumb_dir / _dt.date.today().isoformat() / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest, format="PNG", optimize=True)
            try:
                return str(dest.relative_to(self.root))
            except ValueError:
                return str(dest)
        except Exception as e:
            logger.debug(f"保存缩略图失败: {e}")
            return None

    # 工具函数

    @staticmethod
    def _truncate_args(args: dict, max_value: int = 200) -> dict:
        out = {}
        for k, v in (args or {}).items():
            s = repr(v)
            out[k] = s if len(s) <= max_value else s[:max_value] + "…"
        return out

    @staticmethod
    def _truncate_repr(value: Any, max_len: int = 500) -> Any:
        s = repr(value)
        return s if len(s) <= max_len else s[:max_len] + "…"


class _Timer:
    """with 风格计时器，返回 ms"""
    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000


def timer() -> _Timer:
    return _Timer()
