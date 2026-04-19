"""Computer Use 浏览器控制。"""
from __future__ import annotations

import asyncio
import datetime as _dt
import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from server.core.computer_use.safety import check_url

if TYPE_CHECKING:
    from server.config.settings import ComputerUseConfig
    from server.core.computer_use.audit import Auditor

logger = logging.getLogger(__name__)


class BrowserController:
    """Playwright 浏览器控制器。首次调用时延迟启动。"""

    def __init__(
        self,
        cfg: "ComputerUseConfig",
        auditor: "Auditor",
        project_root: Path,
    ):
        self.cfg = cfg
        self.auditor = auditor
        self.root = project_root
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._screenshot_dir = project_root / "data" / "uploads" / "screens"
        self._screenshot_dir.mkdir(parents=True, exist_ok=True)

    # 生命周期

    async def _ensure_page(self):
        """确保 playwright / browser / page 已就绪。"""
        if self._page is not None:
            return self._page

        try:
            from playwright.async_api import async_playwright
        except ImportError:
            raise RuntimeError(
                "playwright 未安装。请先: pip install playwright && playwright install chromium"
            )

        if self._playwright is None:
            self._playwright = await async_playwright().start()

        if self._browser is None:
            mode = (self.cfg.browser_mode or "launch").lower()
            if mode == "cdp":
                try:
                    self._browser = await self._playwright.chromium.connect_over_cdp(
                        self.cfg.browser_cdp_url
                    )
                except Exception as e:
                    raise RuntimeError(
                        f"CDP 接管失败（{self.cfg.browser_cdp_url}）: {e}\n"
                        f"请手动启动 Chrome: "
                        f"open -a 'Google Chrome' --args --remote-debugging-port=9222"
                    )
            else:
                launch_kwargs = {"headless": bool(self.cfg.browser_headless)}
                if self.cfg.browser_user_data_dir:
                    launch_kwargs["user_data_dir"] = self.cfg.browser_user_data_dir
                self._browser = await self._playwright.chromium.launch(**launch_kwargs)

        if self._context is None:
            existing = getattr(self._browser, "contexts", [])
            if existing:
                self._context = existing[0]
            else:
                downloads = Path(self.cfg.browser_downloads_dir)
                if not downloads.is_absolute():
                    downloads = self.root / downloads
                downloads.mkdir(parents=True, exist_ok=True)
                self._context = await self._browser.new_context(
                    accept_downloads=True,
                )

        existing_pages = self._context.pages
        if existing_pages:
            self._page = existing_pages[0]
        else:
            self._page = await self._context.new_page()

        return self._page

    async def close(self) -> None:
        """Gateway shutdown 时调。幂等"""
        try:
            if self._context is not None:
                try:
                    await self._context.close()
                except Exception:
                    pass
            if self._browser is not None:
                try:
                    await self._browser.close()
                except Exception:
                    pass
            if self._playwright is not None:
                try:
                    await self._playwright.stop()
                except Exception:
                    pass
        finally:
            self._context = None
            self._browser = None
            self._page = None
            self._playwright = None

    # 9 个工具

    async def browser_open(self, url: str) -> dict:
        check_url(self.cfg, url, action="browser_open")
        page = await self._ensure_page()
        await page.goto(url, wait_until="domcontentloaded", timeout=15000)
        self.auditor.write("browser_open", {"url": url})
        return {"ok": True, "url": page.url, "title": await page.title()}

    async def browser_click(
        self,
        selector: Optional[str] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
    ) -> dict:
        if selector is None and (x is None or y is None):
            raise ValueError("必须提供 selector 或 (x, y)")
        page = await self._ensure_page()
        check_url(self.cfg, page.url, action="browser_click")
        if selector:
            await page.click(selector, timeout=5000)
            self.auditor.write("browser_click", {"selector": selector})
        else:
            await page.mouse.click(int(x), int(y))
            self.auditor.write("browser_click", {"x": x, "y": y})
        return {"ok": True, "selector": selector, "x": x, "y": y}

    async def browser_type(self, selector: str, text: str) -> dict:
        page = await self._ensure_page()
        check_url(self.cfg, page.url, action="browser_type")
        await page.fill(selector, str(text), timeout=5000)
        self.auditor.write("browser_type", {"selector": selector, "len": len(str(text))})
        return {"ok": True}

    async def browser_scroll(self, dy: int) -> dict:
        page = await self._ensure_page()
        check_url(self.cfg, page.url, action="browser_scroll")
        await page.evaluate(f"window.scrollBy(0, {int(dy)})")
        self.auditor.write("browser_scroll", {"dy": dy})
        return {"ok": True, "dy": dy}

    async def browser_screenshot(self, full_page: bool = False) -> dict:
        page = await self._ensure_page()
        check_url(self.cfg, page.url, action="browser_screenshot")
        buf = await page.screenshot(full_page=bool(full_page), type="png")
        sha = hashlib.sha256(buf).hexdigest()[:8]
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        fname = f"{stamp}_browser_{sha}.png"
        dest = self._screenshot_dir / _dt.date.today().isoformat() / fname
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(buf)
        try:
            rel = str(dest.relative_to(self.root))
        except ValueError:
            rel = str(dest)
        url_rel = "/" + rel.replace("data/uploads", "uploads", 1).replace("\\", "/")

        # 生成缩略图
        thumb = None
        try:
            from PIL import Image
            import io
            thumb = self.auditor.save_thumbnail(Image.open(io.BytesIO(buf)), tag="browser")
        except Exception:
            pass

        self.auditor.write("browser_screenshot", {"full_page": full_page}, result={"path": rel, "thumb": thumb})
        return {"url": url_rel, "path": rel, "size_bytes": len(buf)}

    async def browser_extract(
        self, selector: Optional[str] = None, html: bool = False
    ) -> dict:
        page = await self._ensure_page()
        check_url(self.cfg, page.url, action="browser_extract")
        try:
            if selector:
                if html:
                    content = await page.inner_html(selector, timeout=5000)
                else:
                    content = await page.inner_text(selector, timeout=5000)
            else:
                if html:
                    content = await page.content()
                else:
                    content = await page.inner_text("body", timeout=5000)
        except Exception as e:
            return {"error": str(e)}
        # 截断过长内容
        max_len = 8000
        if len(content) > max_len:
            content = content[:max_len] + "…[truncated]"
        self.auditor.write("browser_extract", {"selector": selector, "html": html, "len": len(content)})
        return {"text": content, "selector": selector, "is_html": html}

    async def browser_wait_for(self, selector: str, timeout_ms: int = 5000) -> dict:
        page = await self._ensure_page()
        try:
            await page.wait_for_selector(selector, timeout=int(timeout_ms))
            self.auditor.write("browser_wait_for", {"selector": selector})
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def browser_press(self, keys: str) -> dict:
        page = await self._ensure_page()
        check_url(self.cfg, page.url, action="browser_press")
        await page.keyboard.press(str(keys))
        self.auditor.write("browser_press", {"keys": keys})
        return {"ok": True, "keys": keys}

    async def browser_close_page(self) -> dict:
        """关闭当前 page，保留 browser。"""
        if self._page is not None:
            try:
                await self._page.close()
            except Exception as e:
                return {"ok": False, "error": str(e)}
        self._page = None
        self.auditor.write("browser_close_page", {})
        return {"ok": True}
