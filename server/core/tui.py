'''
Author: JimZhang
Date: 2026-04-19 14:00:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-21 02:00:00
FilePath: /JimiAgent/server/core/tui.py
Description: 【已废弃 DEPRECATED】旧版基于 rich.live + prompt_toolkit 的终端 UI。
             仅作为 `jimi chat --classic` 的 fallback；将在下一版本移除。
             新 UI 请见 tui/（React + Ink）与 server/core/tui_worker.py。

'''
from __future__ import annotations

import asyncio
import logging
import warnings
from pathlib import Path
from typing import Optional

warnings.warn(
    "server.core.tui 已废弃：新 TUI 见 tui/（React + Ink）与 server/core/tui_worker.py；"
    "该模块仅作为 `jimi chat --classic` 的临时 fallback，将在下一版本移除。",
    DeprecationWarning,
    stacklevel=2,
)

from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style as PTKStyle
from prompt_toolkit.formatted_text import FormattedText

logger = logging.getLogger(__name__)


# 可补全的斜杠命令
# 顺序即弹窗顺序
SLASH_COMMANDS: dict[str, str] = {
    "/help":         "显示所有可用命令",
    "/status":       "系统与当前会话状态",
    "/new":          "新建会话 [/new <title>]",
    "/sessions":     "列出历史会话",
    "/skills":       "列出技能",
    "/reset":        "清空当前会话历史",
    "/compact":      "手动压缩上下文",
    "/think":        "推理深度 [low|medium|high]",
    "/verbose":      "详细输出 [on|off]",
    "/trace":        "LangGraph 事件 [on|off]",
    "/usage":        "token 统计 [off|tokens|full]",
    "/activation":   "激活策略 [mention|always]",
    "/bash_confirm": "bash 写命令二次确认 [on|off]",
    "/config":       "读/写 yaml 配置 [/config <key> [value]]",
    "/model":        "切换主模型 [/model <id>]",
    "/reload":       "重读 yaml 并清缓存",
    "/edit":         "在 $EDITOR 中编辑 yaml",
    "/restart":      "重建 Skills 索引",
    "/memories":     "列出跨会话长期记忆",
    "/recall":       "检索记忆 [/recall <query>]",
    "/recall_at":    "时间三元组查询 [/recall_at <ts>]",
    "/forget":       "删除记忆 [/forget <id>]",
    "/plugin":       "OpenClaw 插件管理 (list|show|enable|disable|install|uninstall|doctor)",
    "/computer_use": "AI 操控电脑开关 (on|off|status / desktop|browser|loop on|off)",
    "/clear":        "清屏",
    "/quit":         "退出",
    "/exit":         "退出",
    "/q":            "退出",
}


class SlashCommandCompleter(Completer):
    """只在行首 `/` 时弹出命令补全。"""

    def __init__(self, commands: dict[str, str]):
        self._commands = commands

    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        # 只在行首提示命令
        if not text.startswith("/"):
            return
        if " " in text or "\n" in text:
            return
        prefix = text.lower()
        for cmd, desc in self._commands.items():
            if cmd.lower().startswith(prefix):
                yield Completion(
                    cmd,
                    start_position=-len(text),
                    display=cmd,
                    display_meta=desc,
                )

# 提示语
HINT_LINE = (
    "Enter 发送 | Alt+Enter 换行 | Ctrl+C 中断 | Ctrl+D 退出 | /help 命令"
)


def _build_prompt_session(history_path: Path) -> PromptSession:
    """构造带补全和历史的 PromptSession。"""
    history_path.parent.mkdir(parents=True, exist_ok=True)

    completer = SlashCommandCompleter(SLASH_COMMANDS)

    kb = KeyBindings()

    # Alt+Enter / Esc-Enter 插入换行
    @kb.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    style = PTKStyle.from_dict(
        {
            "prompt": "#00d787 bold",
            "prompt.session": "#888888",
            "continuation": "#444444",
            "completion-menu": "bg:#1a1a27 #e8e8f0",
            "completion-menu.completion": "bg:#1a1a27 #e8e8f0",
            "completion-menu.completion.current": "bg:#6c63ff #ffffff bold",
            "completion-menu.meta.completion": "bg:#1a1a27 #9898b0",
            "completion-menu.meta.completion.current": "bg:#6c63ff #ffffff",
        }
    )

    return PromptSession(
        history=FileHistory(str(history_path)),
        completer=completer,
        complete_while_typing=True,
        multiline=False,
        enable_history_search=True,
        key_bindings=kb,
        style=style,
        # 预留补全菜单空间
        reserve_space_for_menu=8,
        # 补全放到线程
        complete_in_thread=True,
    )


def _status_line(agent, session_id: str) -> Text:
    """一行状态栏。"""
    session = agent.session_mgr.get_session(session_id)
    title = session.title if session else session_id
    msgs = session.message_count if session else 0

    model = f"{agent.settings.model.provider}/{agent.settings.model.model_id}"

    # 状态优先取 session.metadata
    meta = session.metadata if session else {}
    think = meta.get("think_level", agent.settings.session.default_think_level)
    verbose = meta.get("verbose", "off")
    usage = meta.get("usage", "off")

    text = Text()
    text.append("会话 ", style="dim")
    text.append(f"{title}", style="bold cyan")
    text.append(f" [{session_id[:8]}]", style="dim")
    text.append("  ·  ", style="dim")
    text.append(f"{model}", style="green")
    text.append("  ·  ", style="dim")
    text.append(f"think:{think}", style="yellow")
    text.append("  ·  ", style="dim")
    text.append(f"msgs:{msgs}", style="magenta")
    if verbose != "off":
        text.append("  ·  ", style="dim")
        text.append(f"verbose:{verbose}", style="bold yellow")
    if usage != "off":
        text.append("  ·  ", style="dim")
        text.append(f"usage:{usage}", style="bold yellow")

    return text


def _welcome_banner(agent, console: Console) -> None:
    """启动时的欢迎横幅。"""
    model = f"{agent.settings.model.provider}/{agent.settings.model.model_id}"
    skills_count = len(agent.workspace.list_skills())
    body = Text()
    body.append("JimiAgent 终端交互模式\n\n", style="bold cyan")
    body.append("模型:    ", style="dim")
    body.append(f"{model}\n", style="green")
    body.append("Skills:  ", style="dim")
    body.append(f"{skills_count} 个\n", style="green")
    body.append("Gateway: ", style="dim")
    body.append(
        f"{agent.settings.gateway.host}:{agent.settings.gateway.port}\n\n",
        style="green",
    )
    body.append(HINT_LINE, style="dim")

    console.print(Panel(body, title="[bold]JimiAgent[/bold]", border_style="cyan"))


class StreamRenderer:
    """把 chat_stream 事件渲染成 rich.live 面板。"""

    def __init__(self, console: Console):
        self.console = console
        self.buffer: str = ""
        self.new_session_id: Optional[str] = None
        self.had_output: bool = False
        # Stage 2: 遇到 confirm_required 时，把 payload 记下让外层 run_chat 处理
        self.pending_confirms: list[dict] = []

    def _render_panel(self) -> Panel:
        body = Markdown(self.buffer) if self.buffer else Text("...", style="dim")
        return Panel(
            body,
            title="[bold blue]Agent[/bold blue]",
            border_style="blue",
            padding=(0, 1),
        )

    async def consume(self, stream) -> None:
        """消费 agent.chat_stream 产生的事件流。"""
        live: Optional[Live] = None

        def _ensure_live_started():
            nonlocal live
            if live is None:
                live = Live(
                    self._render_panel(),
                    console=self.console,
                    refresh_per_second=12,
                    transient=False,
                    vertical_overflow="visible",
                )
                live.start()

        try:
            async for event in stream:
                etype = event.get("type")

                if etype == "text":
                    chunk = event.get("content", "")
                    if not chunk:
                        continue
                    self.buffer += chunk
                    self.had_output = True
                    _ensure_live_started()
                    live.update(self._render_panel())

                elif etype == "tool":
                    # 工具事件单独打印
                    name = event.get("name", "unknown")
                    if live is not None:
                        live.stop()
                        live = None
                    self.console.print(
                        Panel(
                            Text(f"调用工具: {name}", style="cyan"),
                            border_style="cyan",
                            padding=(0, 1),
                        )
                    )
                    # 下一帧 text 时重启 live

                elif etype == "session":
                    self.new_session_id = event.get("session_id")

                elif etype == "error":
                    if live is not None:
                        live.stop()
                        live = None
                    self.console.print(
                        Panel(
                            Text(event.get("content", ""), style="red"),
                            title="[red]错误[/red]",
                            border_style="red",
                        )
                    )

                elif etype == "confirm_required":
                    # 收到 dangerous 操作的 confirm 请求；暂停 live 并记录，
                    # 让 run_chat 外层主循环弹 y/n 后调 chat_stream_resume
                    if live is not None:
                        live.stop()
                        live = None
                    self.pending_confirms.append({
                        "interrupt_id": event.get("interrupt_id", ""),
                        "payload": event.get("payload", {}),
                        "resumable": event.get("resumable", True),
                    })
        finally:
            if live is not None:
                # 结束前再刷新一次
                try:
                    live.update(self._render_panel())
                finally:
                    live.stop()


async def _run_stream_with_confirm_loop(
    agent,
    first_stream,
    *,
    session_id: str,
    console: Console,
    prompt_session,
) -> "StreamRenderer":
    """跑一轮 chat_stream；若遇到 confirm_required 就弹 y/n 后调 chat_stream_resume。

    直到流自然结束（无新的 pending_confirms）才返回最后一个 renderer。
    """
    current_stream = first_stream
    renderer: Optional[StreamRenderer] = None

    while True:
        renderer = StreamRenderer(console)
        try:
            await renderer.consume(current_stream)
        except asyncio.CancelledError:
            # 关闭当前 stream
            try:
                await current_stream.aclose()
            except Exception:
                pass
            raise

        # 主动关闭流，释放 graph 资源
        try:
            await current_stream.aclose()
        except Exception:
            pass

        if not renderer.pending_confirms:
            return renderer

        # 处理全部 pending 中的第 1 个（LangGraph 一次只能 resume 一个 interrupt）
        confirm = renderer.pending_confirms[0]
        payload = confirm.get("payload") or {}
        summary = payload.get("summary") or str(payload)
        detail_lines = []
        detail = payload.get("detail")
        if isinstance(detail, dict):
            for k, v in detail.items():
                val = str(v)
                if len(val) > 240:
                    val = val[:240] + "…"
                detail_lines.append(f"  [dim]{k}[/dim]: {val}")

        console.print(
            Panel(
                (
                    f"[bold yellow]{summary}[/bold yellow]\n"
                    + ("\n".join(detail_lines) if detail_lines else "")
                ),
                title="[yellow]⚠ 需要确认[/yellow]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

        # 弹 y/n；Ctrl+C 视为拒绝
        approve = False
        try:
            resp = await prompt_session.prompt_async(
                FormattedText([("class:prompt", "确认执行? [y/N] ")])
            )
            approve = resp.strip().lower() in ("y", "yes", "approve", "ok", "1")
        except (KeyboardInterrupt, EOFError):
            approve = False
            console.print("[dim](已取消)[/dim]")

        console.print(
            "[green]→ 已批准[/green]" if approve else "[yellow]→ 已拒绝[/yellow]"
        )

        # 以 Command(resume=approve) 继续流
        current_stream = agent.chat_stream_resume(session_id, approve)
        # 下一轮 while 再 consume


async def run_chat(
    session_id: Optional[str] = None,
    history_path: Optional[Path] = None,
) -> None:
    """启动终端聊天主循环。"""
    # 延迟 import，避免启动时加载重依赖
    from server.core.agent import JimiAgent

    console = Console()
    agent = JimiAgent()
    await agent.ainitialize()

    history = history_path or (Path.home() / ".jimiagent" / "chat_history")
    prompt_session = _build_prompt_session(history)

    try:
        _welcome_banner(agent, console)

        if not session_id:
            session = agent.session_mgr.ensure_default_session()
            session_id = session.id

        while True:
            # 每轮先打印状态
            console.print(_status_line(agent, session_id))

            # 读取用户输入
            try:
                user_input = await prompt_session.prompt_async(
                    FormattedText([("class:prompt", "你 > ")])
                )
            except KeyboardInterrupt:
                # 输入阶段 Ctrl+C 只清空当前行
                console.print("[dim](已取消，输入 Ctrl+D 退出)[/dim]")
                continue
            except EOFError:
                # Ctrl+D 退出
                console.print("[dim]再见。[/dim]")
                break

            user_input = user_input.strip()
            if not user_input:
                continue

            # 本地命令不走 Agent
            lower = user_input.lower()
            if lower in ("/quit", "/exit", "/q"):
                console.print("[dim]再见。[/dim]")
                break
            if lower == "/clear":
                console.clear()
                continue
            if lower == "/sessions":
                _render_sessions(agent, console)
                continue
            if lower == "/skills":
                _render_skills(agent, console)
                continue

            # 其余输入统一走 Agent.chat_stream
            try:
                renderer = await _run_stream_with_confirm_loop(
                    agent, agent.chat_stream(user_input, session_id),
                    session_id=session_id,
                    console=console,
                    prompt_session=prompt_session,
                )
            except asyncio.CancelledError:
                console.print("[yellow](已中断)[/yellow]")
                renderer = None
            except KeyboardInterrupt:
                console.print("[yellow](已中断)[/yellow]")
                renderer = None
            except Exception as e:
                logger.exception("chat_stream 异常")
                console.print(f"[red]错误: {type(e).__name__}: {e}[/red]")
                renderer = None

            if renderer and renderer.new_session_id:
                session_id = renderer.new_session_id

            console.print()  # 每轮留一行空白
    finally:
        await agent.aclose()


def _render_sessions(agent, console: Console) -> None:
    """表格展示会话列表。"""
    sessions = agent.session_mgr.list_sessions()
    table = Table(title="会话列表", border_style="cyan")
    table.add_column("ID", style="bold")
    table.add_column("标题")
    table.add_column("消息数", justify="right")
    for s in sessions:
        table.add_row(s.id, s.title, str(s.message_count))
    console.print(table)
    console.print()


def _render_skills(agent, console: Console) -> None:
    """Skills 列表（CLI 本地版，表格呈现）。"""
    from server.core.skill_loader import parse_skill_md

    table = Table(title="Skills 列表", border_style="cyan")
    table.add_column("名称", style="bold")
    table.add_column("描述")
    table.add_column("脚本")
    for name in agent.workspace.list_skills():
        path = agent.workspace.get_skills_dir() / name
        meta = parse_skill_md(path)
        if meta:
            has_script = "yes" if meta.script_path else "no"
            table.add_row(meta.name, meta.description, has_script)
    console.print(table)
    console.print()
