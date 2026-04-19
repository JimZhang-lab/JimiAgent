"""Computer Use 桌面控制。"""
from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from server.core.computer_use.safety import (
    check_app,
    check_dangerous_keypress,
)

if TYPE_CHECKING:
    from server.config.settings import ComputerUseConfig
    from server.core.computer_use.audit import Auditor

logger = logging.getLogger(__name__)


_KEY_SPLIT = re.compile(r"[\s+]+")


class DesktopController:
    """桌面控制器。写操作先过安全检查。"""

    def __init__(
        self,
        cfg: "ComputerUseConfig",
        auditor: "Auditor",
        project_root: Path,
        screenshot_dir: Optional[Path] = None,
    ):
        self.cfg = cfg
        self.auditor = auditor
        self.root = project_root
        self._screenshot_dir = screenshot_dir or (
            project_root / "data" / "uploads" / "screens"
        )
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)
        self._pyautogui = None
        self._mss = None

    # lazy import

    def _pg(self):
        if self._pyautogui is None:
            import pyautogui
            pyautogui.FAILSAFE = True  # 鼠标划到屏幕角触发中断
            pyautogui.PAUSE = max(0.0, self.cfg.action_delay_ms / 1000)
            self._pyautogui = pyautogui
        return self._pyautogui

    def _get_mss(self):
        if self._mss is None:
            import mss as _m
            self._mss = _m
        return self._mss

    # 只读

    def screen_info(self) -> dict:
        """返回屏幕分辨率 + 缩放因子"""
        try:
            with self._get_mss().mss() as sct:
                mons = sct.monitors  # [0] 是全屏组合，[1]=主屏
                primary = mons[1] if len(mons) > 1 else mons[0]
                return {
                    "width": int(primary["width"]),
                    "height": int(primary["height"]),
                    "displays": max(0, len(mons) - 1),
                }
        except Exception as e:
            logger.warning(f"screen_info 失败（mss），回退 pyautogui: {e}")
            try:
                w, h = self._pg().size()
                return {"width": int(w), "height": int(h), "displays": 1}
            except Exception as e2:
                return {"error": f"{e2}"}

    def screen_capture(self, region: Optional[list] = None) -> dict:
        """截图并返回路径与尺寸。"""
        try:
            from PIL import Image
        except ImportError:
            return {"error": "Pillow 未安装"}

        with self._get_mss().mss() as sct:
            if region and len(region) == 4:
                x, y, w, h = region
                mon = {"left": int(x), "top": int(y), "width": int(w), "height": int(h)}
            else:
                mons = sct.monitors
                mon = mons[1] if len(mons) > 1 else mons[0]
            shot = sct.grab(mon)
            img = Image.frombytes("RGB", shot.size, shot.rgb)

        # 按配置缩放宽度上限
        max_w = int(self.cfg.screenshot_max_width)
        if max_w > 0 and img.width > max_w:
            ratio = max_w / img.width
            img = img.resize((max_w, int(img.height * ratio)))

        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        sha = hashlib.sha256(img.tobytes()).hexdigest()[:8]
        fname = f"{stamp}_{sha}.png"
        dest = self._screenshot_dir / _dt.date.today().isoformat() / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        img.save(dest, format="PNG", optimize=True)

        try:
            rel = str(dest.relative_to(self.root))
        except ValueError:
            rel = str(dest)
        # /uploads 已挂成静态目录
        url_rel = "/" + rel.replace("data/uploads", "uploads", 1).replace("\\", "/")
        # 审计缩略图
        thumb = self.auditor.save_thumbnail(img, tag="screen")
        self.auditor.write(
            "screen_capture",
            {"region": region, "width": img.width, "height": img.height},
            result={"path": rel, "thumb": thumb},
        )
        return {
            "url": url_rel,
            "path": rel,
            "width": img.width,
            "height": img.height,
            "size_bytes": dest.stat().st_size,
        }

    # 鼠标

    def mouse_move(self, x: int, y: int, duration_ms: int = 300) -> dict:
        check_app(self.cfg, action="mouse_move")
        self._pg().moveTo(int(x), int(y), duration=max(0.0, duration_ms / 1000))
        self.auditor.write("mouse_move", {"x": x, "y": y})
        return {"ok": True, "x": x, "y": y}

    def mouse_click(
        self, x: int, y: int, button: str = "left", clicks: int = 1
    ) -> dict:
        check_app(self.cfg, action=f"mouse_click({button}x{clicks})")
        btn = button.lower()
        if btn not in ("left", "right", "middle"):
            raise ValueError(f"button 必须是 left/right/middle，got {button}")
        self._pg().click(
            x=int(x), y=int(y), clicks=int(clicks), button=btn
        )
        self.auditor.write("mouse_click", {"x": x, "y": y, "button": btn, "clicks": clicks})
        return {"ok": True, "x": x, "y": y, "button": btn, "clicks": clicks}

    def mouse_drag(
        self,
        from_x: int,
        from_y: int,
        to_x: int,
        to_y: int,
        duration_ms: int = 500,
    ) -> dict:
        check_app(self.cfg, action="mouse_drag")
        pg = self._pg()
        pg.moveTo(int(from_x), int(from_y))
        pg.dragTo(
            int(to_x), int(to_y),
            duration=max(0.0, duration_ms / 1000),
            button="left",
        )
        self.auditor.write("mouse_drag", {
            "from": (from_x, from_y), "to": (to_x, to_y),
        })
        return {"ok": True, "from": [from_x, from_y], "to": [to_x, to_y]}

    def mouse_scroll(
        self, dy: int, x: Optional[int] = None, y: Optional[int] = None
    ) -> dict:
        check_app(self.cfg, action="mouse_scroll")
        if x is not None and y is not None:
            self._pg().moveTo(int(x), int(y))
        self._pg().scroll(int(dy))
        self.auditor.write("mouse_scroll", {"dy": dy, "x": x, "y": y})
        return {"ok": True, "dy": dy}

    # 键盘

    def keyboard_type(self, text: str, interval_ms: int = 20) -> dict:
        check_app(self.cfg, action="keyboard_type")
        self._pg().typewrite(
            str(text), interval=max(0.0, interval_ms / 1000)
        )
        self.auditor.write("keyboard_type", {"text_len": len(str(text))})
        return {"ok": True, "length": len(str(text))}

    def keyboard_press(self, keys: str) -> dict:
        check_dangerous_keypress(self.cfg, keys)
        check_app(self.cfg, action=f"keyboard_press({keys})")
        # "cmd+c" -> ["cmd", "c"]
        parts = [p.strip().lower() for p in _KEY_SPLIT.split(keys) if p.strip()]
        if len(parts) == 1:
            self._pg().press(parts[0])
        else:
            self._pg().hotkey(*parts)
        self.auditor.write("keyboard_press", {"keys": keys})
        return {"ok": True, "keys": keys}


# async wrapper

async def run_sync(fn, *args, **kwargs):
    """把同步调用丢到 thread pool。"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: fn(*args, **kwargs))
