"""Computer Use 工具装配。"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from langchain_core.tools import StructuredTool

from server.config.settings import PROJECT_ROOT
from server.core.computer_use.audit import Auditor, timer
from server.core.computer_use.browser import BrowserController
from server.core.computer_use.desktop import DesktopController
from server.core.computer_use.loop_agent import ComputerTaskRunner
from server.core.computer_use.safety import (
    ComputerUsePermissionError,
    is_enabled,
)

if TYPE_CHECKING:
    from server.core.agent import JimiAgent

logger = logging.getLogger(__name__)


# Agent 级单例容器
_CACHE: dict[int, dict] = {}


def _get_controllers(agent: "JimiAgent") -> dict:
    key = id(agent)
    if key in _CACHE:
        return _CACHE[key]

    cfg = agent.settings.computer_use
    auditor = Auditor(cfg, PROJECT_ROOT)
    desktop = DesktopController(cfg, auditor, PROJECT_ROOT) if cfg.desktop_enabled else None
    browser = BrowserController(cfg, auditor, PROJECT_ROOT) if cfg.browser_enabled else None
    loop_runner = None
    if cfg.loop_agent_enabled:
        vlm = getattr(agent, "vlm", None) or getattr(agent, "llm", None)
        loop_runner = ComputerTaskRunner(cfg, auditor, desktop, browser, vlm)

    state = {
        "cfg": cfg,
        "auditor": auditor,
        "desktop": desktop,
        "browser": browser,
        "loop": loop_runner,
    }
    _CACHE[key] = state
    return state


# 通用封装

def _wrap_result(result) -> str:
    try:
        return json.dumps(result, ensure_ascii=False)
    except Exception:
        return str(result)


def _safe_call(agent: "JimiAgent", tool_name: str, fn, **kwargs) -> str:
    """同步工具封装。"""
    state = _get_controllers(agent)
    auditor: Auditor = state["auditor"]
    try:
        with timer() as t:
            result = fn(**kwargs)
        # desktop 自己写审计，这里只兜底异常
        return _wrap_result(result)
    except ComputerUsePermissionError as e:
        auditor.write(tool_name, kwargs, error=str(e))
        return _wrap_result({"error": "permission_denied", "detail": str(e)})
    except Exception as e:
        auditor.write(tool_name, kwargs, error=str(e))
        return _wrap_result({"error": str(e)})


async def _safe_acall(agent: "JimiAgent", tool_name: str, coro_fn, **kwargs) -> str:
    state = _get_controllers(agent)
    auditor: Auditor = state["auditor"]
    try:
        result = await coro_fn(**kwargs)
        return _wrap_result(result)
    except ComputerUsePermissionError as e:
        auditor.write(tool_name, kwargs, error=str(e))
        return _wrap_result({"error": "permission_denied", "detail": str(e)})
    except Exception as e:
        auditor.write(tool_name, kwargs, error=str(e))
        return _wrap_result({"error": str(e)})


# Desktop tools

def _desktop_tools(agent: "JimiAgent") -> list[StructuredTool]:
    state = _get_controllers(agent)
    d: DesktopController = state["desktop"]
    if d is None:
        return []

    def screen_capture(region: Optional[list] = None) -> str:
        """截取当前屏幕。"""
        return _safe_call(agent, "screen_capture", d.screen_capture, region=region)

    def screen_info() -> str:
        """返回屏幕分辨率 + 显示器数量"""
        return _safe_call(agent, "screen_info", d.screen_info)

    def mouse_move(x: int, y: int, duration_ms: int = 300) -> str:
        """把鼠标移到屏幕坐标 (x, y)"""
        return _safe_call(agent, "mouse_move", d.mouse_move, x=x, y=y, duration_ms=duration_ms)

    def mouse_click(x: int, y: int, button: str = "left", clicks: int = 1) -> str:
        """在 (x, y) 点击。"""
        return _safe_call(agent, "mouse_click", d.mouse_click, x=x, y=y, button=button, clicks=clicks)

    def mouse_drag(from_x: int, from_y: int, to_x: int, to_y: int, duration_ms: int = 500) -> str:
        """从起点拖拽到终点。"""
        return _safe_call(agent, "mouse_drag", d.mouse_drag,
                          from_x=from_x, from_y=from_y, to_x=to_x, to_y=to_y, duration_ms=duration_ms)

    def mouse_scroll(dy: int, x: Optional[int] = None, y: Optional[int] = None) -> str:
        """滚动鼠标。"""
        return _safe_call(agent, "mouse_scroll", d.mouse_scroll, dy=dy, x=x, y=y)

    def keyboard_type(text: str, interval_ms: int = 20) -> str:
        """模拟键盘输入文本"""
        return _safe_call(agent, "keyboard_type", d.keyboard_type, text=text, interval_ms=interval_ms)

    def keyboard_press(keys: str) -> str:
        """按下快捷键。"""
        return _safe_call(agent, "keyboard_press", d.keyboard_press, keys=keys)

    return [
        StructuredTool.from_function(func=screen_capture, name="screen_capture",
                                      description="截取屏幕（可选 region）返回 url+path+尺寸"),
        StructuredTool.from_function(func=screen_info, name="screen_info",
                                      description="查询屏幕分辨率 + 显示器数量"),
        StructuredTool.from_function(func=mouse_move, name="mouse_move",
                                      description="移动鼠标到 (x, y)"),
        StructuredTool.from_function(func=mouse_click, name="mouse_click",
                                      description="鼠标点击 (x, y)；button=left/right/middle"),
        StructuredTool.from_function(func=mouse_drag, name="mouse_drag",
                                      description="鼠标拖拽 (from_x,from_y) → (to_x,to_y)"),
        StructuredTool.from_function(func=mouse_scroll, name="mouse_scroll",
                                      description="滚动：dy 正数上滚"),
        StructuredTool.from_function(func=keyboard_type, name="keyboard_type",
                                      description="输入文本"),
        StructuredTool.from_function(func=keyboard_press, name="keyboard_press",
                                      description="按快捷键：'cmd+c' / 'enter' / 'shift+tab'"),
    ]


# Browser tools

def _browser_tools(agent: "JimiAgent") -> list[StructuredTool]:
    state = _get_controllers(agent)
    b: BrowserController = state["browser"]
    if b is None:
        return []

    async def browser_open(url: str) -> str:
        """打开 URL（复用已有 page）"""
        return await _safe_acall(agent, "browser_open", b.browser_open, url=url)

    async def browser_click(
        selector: Optional[str] = None, x: Optional[int] = None, y: Optional[int] = None
    ) -> str:
        """在 selector 或 (x, y) 点击。"""
        return await _safe_acall(agent, "browser_click", b.browser_click,
                                  selector=selector, x=x, y=y)

    async def browser_type(selector: str, text: str) -> str:
        """在 selector 指定的输入框填文本"""
        return await _safe_acall(agent, "browser_type", b.browser_type,
                                  selector=selector, text=text)

    async def browser_scroll(dy: int) -> str:
        """滚动浏览器页面 dy 像素（正数下滚）"""
        return await _safe_acall(agent, "browser_scroll", b.browser_scroll, dy=dy)

    async def browser_screenshot(full_page: bool = False) -> str:
        """浏览器页面截图。"""
        return await _safe_acall(agent, "browser_screenshot", b.browser_screenshot,
                                  full_page=full_page)

    async def browser_extract(selector: Optional[str] = None, html: bool = False) -> str:
        """抽取元素文本或 HTML。"""
        return await _safe_acall(agent, "browser_extract", b.browser_extract,
                                  selector=selector, html=html)

    async def browser_wait_for(selector: str, timeout_ms: int = 5000) -> str:
        """等待 selector 出现。"""
        return await _safe_acall(agent, "browser_wait_for", b.browser_wait_for,
                                  selector=selector, timeout_ms=timeout_ms)

    async def browser_press(keys: str) -> str:
        """向浏览器页面发按键；如 'Enter' / 'Tab' / 'Control+A'"""
        return await _safe_acall(agent, "browser_press", b.browser_press, keys=keys)

    async def browser_close() -> str:
        """关当前 page（保留 browser 实例）"""
        return await _safe_acall(agent, "browser_close", b.browser_close_page)

    return [
        StructuredTool.from_function(coroutine=browser_open, name="browser_open",
                                      description="打开 URL"),
        StructuredTool.from_function(coroutine=browser_click, name="browser_click",
                                      description="点击 selector 或 (x,y)"),
        StructuredTool.from_function(coroutine=browser_type, name="browser_type",
                                      description="填写输入框"),
        StructuredTool.from_function(coroutine=browser_scroll, name="browser_scroll",
                                      description="滚动页面"),
        StructuredTool.from_function(coroutine=browser_screenshot, name="browser_screenshot",
                                      description="页面截图返回 url"),
        StructuredTool.from_function(coroutine=browser_extract, name="browser_extract",
                                      description="抽取元素文本/HTML"),
        StructuredTool.from_function(coroutine=browser_wait_for, name="browser_wait_for",
                                      description="等 selector 出现"),
        StructuredTool.from_function(coroutine=browser_press, name="browser_press",
                                      description="发按键"),
        StructuredTool.from_function(coroutine=browser_close, name="browser_close",
                                      description="关当前 page"),
    ]


# Loop Agent tool

def _loop_tools(agent: "JimiAgent") -> list[StructuredTool]:
    state = _get_controllers(agent)
    runner: ComputerTaskRunner = state["loop"]
    if runner is None:
        return []

    async def computer_task(
        goal: str, max_steps: int = 20, domain_hint: Optional[str] = None
    ) -> str:
        """任务级 AI 操控电脑。"""
        try:
            result = await runner.run(goal=goal, max_steps=max_steps, domain_hint=domain_hint)
            return _wrap_result(result)
        except Exception as e:
            state["auditor"].write("computer_task", {"goal": goal}, error=str(e))
            return _wrap_result({"error": str(e)})

    return [
        StructuredTool.from_function(
            coroutine=computer_task,
            name="computer_task",
            description=(
                "【慎用】任务级 AI 操控电脑：传目标文字，VLM 自动 观察-决策-执行 闭环。"
                "用 max_steps 控制最大步数（默认 20）。"
            ),
        ),
    ]


# 对外入口

def build_computer_use_tools(agent: "JimiAgent") -> list[StructuredTool]:
    """按配置生成 Computer Use 工具列表。"""
    cfg = agent.settings.computer_use
    if not is_enabled(cfg):
        return []
    tools: List[StructuredTool] = []
    if cfg.desktop_enabled:
        tools.extend(_desktop_tools(agent))
    if cfg.browser_enabled:
        tools.extend(_browser_tools(agent))
    if cfg.loop_agent_enabled:
        tools.extend(_loop_tools(agent))
    return tools


async def shutdown(agent: "JimiAgent") -> None:
    """Gateway shutdown 时关闭 browser。"""
    state = _CACHE.pop(id(agent), None)
    if state and state.get("browser"):
        try:
            await state["browser"].close()
        except Exception as e:
            logger.debug(f"computer_use shutdown: {e}")
