"""Computer Use 循环执行器。"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from server.config.settings import ComputerUseConfig
    from server.core.computer_use.audit import Auditor
    from server.core.computer_use.browser import BrowserController
    from server.core.computer_use.desktop import DesktopController

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """\
你是一个电脑操控 Agent。我会提供用户目标 + 当前屏幕截图（或浏览器页面截图），你需要决定下一步动作。

规则：
1. 每次**只返回一个 JSON 对象**（不要 markdown ```json``` 包裹）
2. 合法的动作格式：
   - {"action": "mouse_click", "x": 100, "y": 200}
   - {"action": "mouse_move", "x": 100, "y": 200}
   - {"action": "mouse_scroll", "dy": -300}
   - {"action": "keyboard_type", "text": "hello"}
   - {"action": "keyboard_press", "keys": "cmd+a"}
   - {"action": "browser_open", "url": "https://example.com"}
   - {"action": "browser_click", "selector": "button.submit"}
   - {"action": "browser_type", "selector": "input[name=q]", "text": "..."}
   - {"action": "browser_screenshot"}
   - {"action": "browser_extract", "selector": "h1"}
   - {"action": "wait", "ms": 500}
   - {"action": "done", "result": "简要说明结果"}
3. 坐标基于当前截图（宽度已标注）。不确定时用 browser_* 工具（selector 更稳）
4. 不要重复执行上一步相同的动作。history 已列出你做过什么
5. 最多 __MAX_STEPS__ 步；到达目标立刻 `done`，不要冗余
"""


_ACTION_JSON = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _parse_action(text: str) -> Optional[dict]:
    """从 VLM 输出里抓 JSON。"""
    # 先尝试整段 parse
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```\s*$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    # 再匹配第一个 JSON 对象
    m = _ACTION_JSON.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


class ComputerTaskRunner:
    """任务级循环执行器。"""

    def __init__(
        self,
        cfg: "ComputerUseConfig",
        auditor: "Auditor",
        desktop: Optional["DesktopController"],
        browser: Optional["BrowserController"],
        vlm,                                    # self.agent.vlm（ChatModel）
    ):
        self.cfg = cfg
        self.auditor = auditor
        self.desktop = desktop
        self.browser = browser
        self.vlm = vlm

    async def run(
        self, goal: str, max_steps: int = 0, domain_hint: Optional[str] = None
    ) -> dict:
        """跑一次任务循环。"""
        max_steps = max_steps or self.cfg.max_actions_per_task
        history: list[dict] = []
        final_shot = None

        sys_prompt = _SYSTEM_PROMPT.replace("__MAX_STEPS__", str(max_steps))
        if domain_hint:
            sys_prompt += f"\n\n当前任务领域提示：{domain_hint}"

        for step in range(1, max_steps + 1):
            # 1) 截屏
            try:
                if self.browser is not None and self.browser._page is not None:
                    shot = await self.browser.browser_screenshot(full_page=False)
                elif self.desktop is not None:
                    loop = asyncio.get_event_loop()
                    shot = await loop.run_in_executor(
                        None, self.desktop.screen_capture, None
                    )
                else:
                    return {"status": "no_controller", "steps": step, "history": history}
            except Exception as e:
                logger.warning(f"loop_agent 截屏失败: {e}")
                shot = {"error": str(e)}
            final_shot = shot

            # 2) 问 VLM 下一步
            try:
                action = await asyncio.wait_for(
                    self._decide(goal, sys_prompt, shot, history, step, max_steps),
                    timeout=self.cfg.loop_step_timeout_seconds,
                )
            except asyncio.TimeoutError:
                return {
                    "status": "vlm_timeout",
                    "steps": step,
                    "history": history,
                    "final_screenshot": final_shot.get("url") if isinstance(final_shot, dict) else None,
                }

            if not action or not isinstance(action, dict):
                history.append({"step": step, "action": None, "raw_error": "无法解析 VLM 输出"})
                continue

            # 3) done 直接返回
            if action.get("action") == "done":
                return {
                    "status": "done",
                    "steps": step,
                    "history": history,
                    "final_screenshot": final_shot.get("url") if isinstance(final_shot, dict) else None,
                    "result_text": action.get("result", ""),
                }

            # 4) dispatch
            try:
                result = await self._dispatch(action)
            except Exception as e:
                logger.warning(f"loop_agent 动作失败: {e}")
                result = {"error": str(e)}
            history.append({"step": step, "action": action, "result": result})

            # 5) 每步 sleep
            await asyncio.sleep(max(0.0, self.cfg.action_delay_ms / 1000))

        return {
            "status": "max_steps_exceeded",
            "steps": max_steps,
            "history": history,
            "final_screenshot": final_shot.get("url") if isinstance(final_shot, dict) else None,
        }

    async def _decide(
        self,
        goal: str,
        sys_prompt: str,
        shot: dict,
        history: list[dict],
        step: int,
        max_steps: int,
    ) -> Optional[dict]:
        """调 VLM 生成下一步 action。"""
        from langchain_core.messages import SystemMessage, HumanMessage

        history_brief = [
            {"step": h["step"], "action": h.get("action"), "result_ok": "error" not in (h.get("result") or {})}
            for h in history[-6:]                       # 最近 6 步
        ]
        screenshot_url = shot.get("url") if isinstance(shot, dict) else None

        user_content = [
            {
                "type": "text",
                "text": (
                    f"用户目标：{goal}\n\n"
                    f"当前步数：{step}/{max_steps}\n"
                    f"历史：{json.dumps(history_brief, ensure_ascii=False)}\n"
                    f"截图尺寸：{shot.get('width')}×{shot.get('height')}\n\n"
                    "请返回下一步 JSON action（若已完成目标返回 done）。"
                ),
            }
        ]
        if screenshot_url:
            user_content.append({"type": "image_url", "image_url": screenshot_url})

        messages = [SystemMessage(content=sys_prompt), HumanMessage(content=user_content)]

        try:
            resp = await self.vlm.ainvoke(messages)
        except Exception as e:
            logger.warning(f"VLM 调用失败: {e}")
            return None

        text = resp.content if hasattr(resp, "content") else str(resp)
        if isinstance(text, list):
            text = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p) for p in text
            )
        return _parse_action(str(text))

    async def _dispatch(self, action: dict) -> dict:
        """分发动作到 desktop / browser 控制器。"""
        kind = action.get("action", "")

        # wait 只是 sleep
        if kind == "wait":
            await asyncio.sleep(max(0, int(action.get("ms", 0))) / 1000)
            return {"ok": True}

        # desktop
        if kind.startswith("mouse_") or kind.startswith("keyboard_") or kind == "screen_capture":
            if self.desktop is None or not self.cfg.desktop_enabled:
                return {"error": "desktop 未启用"}
            loop = asyncio.get_event_loop()
            fn = getattr(self.desktop, kind, None)
            if fn is None:
                return {"error": f"未知动作 {kind}"}
            kwargs = {k: v for k, v in action.items() if k != "action"}
            return await loop.run_in_executor(None, lambda: fn(**kwargs))

        # browser
        if kind.startswith("browser_"):
            if self.browser is None or not self.cfg.browser_enabled:
                return {"error": "browser 未启用"}
            # 用 browser_close_page 避免命名冲突
            fn_name = "browser_close_page" if kind == "browser_close" else kind
            fn = getattr(self.browser, fn_name, None)
            if fn is None:
                return {"error": f"未知动作 {kind}"}
            kwargs = {k: v for k, v in action.items() if k != "action"}
            return await fn(**kwargs)

        return {"error": f"未知动作类型 {kind}"}
