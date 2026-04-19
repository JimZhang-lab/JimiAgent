'''
Author: JimZhang
Date: 2026-04-19 01:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:10:00
FilePath: /JimiAgent/server/core/commands.py
Description: 斜杠命令分发器。

'''
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from server.core.agent import JimiAgent

logger = logging.getLogger(__name__)


# Think levels

THINK_LEVELS = ("low", "medium", "high")

THINK_PROMPTS: dict[str, str] = {
    "low": (
        "## 推理深度: low\n"
        "直接给出答案；除非用户明确要求，否则不展开详细步骤。"
    ),
    "medium": (
        "## 推理深度: medium\n"
        "在回答前进行合理推理，必要时列出关键步骤或选项，但避免冗长分析。"
    ),
    "high": (
        "## 推理深度: high\n"
        "在回答复杂问题前，先充分思考：拆解问题、列出所有相关因素、"
        "检查边界条件与风险，再给出结论。适合严谨的技术/决策类问题。"
    ),
}


def think_prompt_for(level: str) -> str:
    """根据 think level 返回要追加到 system prompt 的说明段"""
    return THINK_PROMPTS.get(level.lower(), THINK_PROMPTS["medium"])


# 状态常量

VERBOSE_VALUES = ("on", "off")
TRACE_VALUES = ("on", "off")
USAGE_VALUES = ("off", "tokens", "full")
ACTIVATION_VALUES = ("mention", "always")
BASH_CONFIRM_VALUES = ("on", "off")


# 命令分发

@dataclass
class CommandResult:
    """命令处理结果"""
    handled: bool
    response: str = ""
    # 命令可能切换 session
    new_session_id: Optional[str] = None


class CommandDispatcher:
    """斜杠命令调度器。"""

    def __init__(self, agent: "JimiAgent"):
        self.agent = agent

    async def try_handle(self, message: str, session_id: str) -> CommandResult:
        """若消息是斜杠命令则处理，否则返回 handled=False"""
        msg = message.strip()
        if not msg.startswith("/"):
            return CommandResult(handled=False)

        parts = msg.split(None, 1)
        cmd = parts[0].lower().lstrip("/")
        arg = parts[1].strip() if len(parts) > 1 else ""

        handler: Optional[Callable[[str, str], Awaitable[CommandResult]]] = {
            "status": self._cmd_status,
            "new": self._cmd_new,
            "reset": self._cmd_reset,
            "compact": self._cmd_compact,
            "think": self._cmd_think,
            "verbose": self._cmd_verbose,
            "usage": self._cmd_usage,
            "trace": self._cmd_trace,
            "activation": self._cmd_activation,
            "bash_confirm": self._cmd_bash_confirm,
            "restart": self._cmd_restart,
            "help": self._cmd_help,
            "h": self._cmd_help,
            "config": self._cmd_config,
            "model": self._cmd_model,
            "reload": self._cmd_reload,
            "edit": self._cmd_edit,
            "memories": self._cmd_memories,
            "memory": self._cmd_memories,
            "recall": self._cmd_recall,
            "recall_at": self._cmd_recall_at,
            "forget": self._cmd_forget,
            "plugin": self._cmd_plugin,
            "plugins": self._cmd_plugin,
            "computer_use": self._cmd_computer_use,
        }.get(cmd)

        if handler is None:
            return CommandResult(
                handled=True,
                response=(
                    f"未知命令: `/{cmd}`\n"
                    f"输入 `/help` 查看所有命令。"
                ),
            )

        try:
            return await handler(arg, session_id)
        except Exception as e:
            logger.error(f"命令 /{cmd} 执行失败: {e}", exc_info=True)
            return CommandResult(
                handled=True,
                response=f"命令执行失败: {e}",
            )

    # 各命令实现

    async def _cmd_status(self, arg: str, session_id: str) -> CommandResult:
        a = self.agent
        s = a.settings
        sess = a.session_mgr.get_session(session_id)
        skills = a.workspace.list_skills()
        lines = [
            "## 系统状态",
            "",
            f"- **模型**: `{s.model.provider}/{s.model.model_id}`",
            f"- **Embedding**: `{s.embedding.model}`",
            f"- **Skills**: {len(skills)} 个 ({', '.join(skills) if skills else '无'})",
            f"- **会话总数**: {len(a.session_mgr.list_sessions())}",
            f"- **记忆**: {'SQLite 持久化' if a.memory.is_persistent else '内存（未持久化）'}",
            f"- **Gateway**: `{s.gateway.host}:{s.gateway.port}`",
            f"- **日志**: `{s.logging.level}` · `{s.logging.output}`",
        ]

        # Scheduler 状态
        sched = getattr(a, "scheduler", None)
        if sched is not None:
            jobs = sched.list_jobs()
            active = sum(1 for j in jobs if j["enabled"])
            state = "running" if s.scheduler.enabled else "disabled"
            line = f"- **Scheduler**: `{state}` · {active}/{len(jobs)} 个任务启用"
            # 最近一次执行时间
            upcoming = [
                j["next_run_at"] for j in jobs
                if j["enabled"] and j["next_run_at"]
            ]
            if upcoming:
                next_at = min(upcoming)
                line += f"（下次: `{next_at}`）"
            lines.append(line)

        if sess:
            think = sess.metadata.get("think_level", s.session.default_think_level)
            verbose = sess.metadata.get("verbose", "off")
            usage = sess.metadata.get("usage", "off")
            lines.extend([
                "",
                f"### 当前会话 `{sess.id}` - {sess.title}",
                f"- 消息数: {sess.message_count}",
                f"- 推理深度: `{think}`",
                f"- Verbose: `{verbose}` | Usage: `{usage}`",
            ])
        return CommandResult(handled=True, response="\n".join(lines))

    async def _cmd_new(self, arg: str, session_id: str) -> CommandResult:
        title = arg or "新对话"
        sess = self.agent.session_mgr.create_session(title)
        return CommandResult(
            handled=True,
            response=f"已创建新会话: `{sess.id}` - {sess.title}",
            new_session_id=sess.id,
        )

    async def _cmd_reset(self, arg: str, session_id: str) -> CommandResult:
        """清空当前会话的消息历史（但保留 session 元信息）"""
        ok = await self.agent.clear_session_history(session_id)
        if ok:
            self.agent.session_mgr.touch(session_id, message_count=0)
            return CommandResult(
                handled=True,
                response=f"会话 `{session_id}` 的历史已清空。",
            )
        return CommandResult(
            handled=True,
            response=f"无法清空会话 `{session_id}`（可能是内存模式或底层不支持）。",
        )

    async def _cmd_compact(self, arg: str, session_id: str) -> CommandResult:
        """手动触发上下文压缩"""
        try:
            summary = await self.agent.compact_session(session_id)
            return CommandResult(
                handled=True,
                response=f"已压缩上下文，摘要如下:\n\n{summary}",
            )
        except Exception as e:
            return CommandResult(handled=True, response=f"Compact 失败: {e}")

    async def _cmd_think(self, arg: str, session_id: str) -> CommandResult:
        if not arg:
            sess = self.agent.session_mgr.get_session(session_id)
            current = (
                sess.metadata.get("think_level", self.agent.settings.session.default_think_level)
                if sess else "medium"
            )
            return CommandResult(
                handled=True,
                response=(
                    f"当前推理深度: `{current}`\n"
                    f"可选值: `{', '.join(THINK_LEVELS)}`\n"
                    f"用法: `/think <level>`"
                ),
            )
        level = arg.lower()
        if level not in THINK_LEVELS:
            return CommandResult(
                handled=True,
                response=f"无效值 `{arg}`。可选: {', '.join(THINK_LEVELS)}",
            )
        self._set_session_meta(session_id, "think_level", level)
        # 切换后清 graph 缓存
        self.agent.invalidate_agent_cache()
        return CommandResult(handled=True, response=f"推理深度已设为 `{level}`。")

    async def _cmd_verbose(self, arg: str, session_id: str) -> CommandResult:
        return self._handle_enum_toggle(
            arg, session_id, "verbose", VERBOSE_VALUES, "Verbose"
        )

    async def _cmd_usage(self, arg: str, session_id: str) -> CommandResult:
        return self._handle_enum_toggle(
            arg, session_id, "usage", USAGE_VALUES, "Usage"
        )

    async def _cmd_trace(self, arg: str, session_id: str) -> CommandResult:
        return self._handle_enum_toggle(
            arg, session_id, "trace", TRACE_VALUES, "Trace"
        )

    async def _cmd_activation(self, arg: str, session_id: str) -> CommandResult:
        return self._handle_enum_toggle(
            arg, session_id, "activation", ACTIVATION_VALUES, "激活策略"
        )

    async def _cmd_bash_confirm(self, arg: str, session_id: str) -> CommandResult:
        return self._handle_enum_toggle(
            arg, session_id, "bash_confirm", BASH_CONFIRM_VALUES,
            "bash 二次确认",
        )

    async def _cmd_restart(self, arg: str, session_id: str) -> CommandResult:
        """重建 Skills 索引并清缓存。"""
        try:
            self.agent.skill_retriever.rebuild_index()
            self.agent.invalidate_agent_cache()
            self.agent.workspace.reload()
            return CommandResult(
                handled=True,
                response="已重启: Skills 索引重建、Workspace 重载、Agent 缓存清空。",
            )
        except Exception as e:
            return CommandResult(handled=True, response=f"重启失败: {e}")

    async def _cmd_help(self, arg: str, session_id: str) -> CommandResult:
        return CommandResult(
            handled=True,
            response=(
                "## 可用命令\n\n"
                "| 命令 | 说明 |\n"
                "| --- | --- |\n"
                "| `/status` | 查看系统与当前会话状态 |\n"
                "| `/new [title]` | 新建会话 |\n"
                "| `/reset` | 清空当前会话历史 |\n"
                "| `/compact` | 手动压缩上下文 |\n"
                "| `/think <low\\|medium\\|high>` | 切换推理深度 |\n"
                "| `/verbose <on\\|off>` | 是否展示详细输出（工具名） |\n"
                "| `/trace <on\\|off>` | 是否透出 LangGraph 内部事件 |\n"
                "| `/usage <off\\|tokens\\|full>` | 切换 usage 粒度 |\n"
                "| `/activation <mention\\|always>` | 激活策略（mention 下需 @agent-name 才响应） |\n"
                "| `/bash_confirm <on\\|off>` | bash 工具对写类命令是否要求二次确认 |\n"
                "| `/config [key] [value]` | 查看/修改 yaml 配置 |\n"
                "| `/model [id]` | 查看/切换主模型 model_id（持久化） |\n"
                "| `/reload` | 重读 yaml 并清缓存 |\n"
                "| `/edit` | 在 TUI 内用 $EDITOR 打开 yaml |\n"
                "| `/restart` | 重建 Skills 索引、清缓存 |\n"
                "| `/memories [kind=semantic\\|clear yes]` | 列出/清空跨会话长期记忆 |\n"
                "| `/recall <query>` | 检索长期记忆 |\n"
                "| `/recall_at <ts> [subject=... predicate=...]` | 时间三元组查询 |\n"
                "| `/forget <id>` | 软删某条记忆 |\n"
                "| `/plugin list\\|show\\|enable\\|disable\\|install\\|uninstall\\|doctor` | OpenClaw 插件管理 |\n"
                "| `/computer_use [on\\|off\\|status]` 或 `/computer_use <desktop\\|browser\\|loop> on\\|off` | AI 操控电脑开关（M 组） |\n"
                "| `/help` | 显示本帮助 |"
            ),
        )

    async def _cmd_config(self, arg: str, session_id: str) -> CommandResult:
        """查看 / 修改 yaml 配置"""
        from server.config.config_io import get_value, list_keys, set_value

        parts = arg.split(None, 1)

        # 无参数时列出配置项
        if not parts:
            lines = ["## 可修改配置项\n", "| key | 当前值 |", "| --- | --- |"]
            for key, val in list_keys():
                lines.append(f"| `{key}` | `{val}` |")
            lines.append(
                "\n用法: `/config <key>` 读取；`/config <key> <value>` 修改"
            )
            return CommandResult(handled=True, response="\n".join(lines))

        key = parts[0].strip()
        # 单参数读取
        if len(parts) == 1:
            try:
                val = get_value(key)
                return CommandResult(
                    handled=True, response=f"`{key}` = `{val!r}`"
                )
            except KeyError:
                return CommandResult(
                    handled=True, response=f"未找到 key: `{key}`"
                )
            except FileNotFoundError as e:
                return CommandResult(handled=True, response=f"{e}")

        # 双参数写入
        value = parts[1].strip()
        try:
            new_val = set_value(key, value)
        except PermissionError as e:
            return CommandResult(handled=True, response=f"{e}")
        except (ValueError, KeyError) as e:
            return CommandResult(handled=True, response=f"设置失败: {e}")

        # 清 Agent graph 缓存
        self.agent.invalidate_agent_cache()
        return CommandResult(
            handled=True,
            response=(
                f"已写入 `{key}` = `{new_val!r}`\n\n"
                "已清 Agent graph 缓存。注意：\n"
                "- 模型/Embedding/Gateway 端口等核心字段需**重启 Gateway** 才完全生效\n"
                "- think / session / skills / logging 改动会在下一轮对话立即生效"
            ),
        )

    async def _cmd_model(self, arg: str, session_id: str) -> CommandResult:
        """查看或切换主模型 model_id（持久化）"""
        from server.config.config_io import set_value

        a = self.agent
        current = f"{a.settings.model.provider}/{a.settings.model.model_id}"
        if not arg:
            return CommandResult(
                handled=True,
                response=(
                    f"当前模型: `{current}`\n"
                    "用法: `/model <model_id>` 切换并持久化；"
                    "`/config agent.model.provider <name>` 切换 provider。"
                ),
            )

        new_model = arg.strip()
        try:
            set_value("agent.model.model_id", new_model)
        except Exception as e:
            return CommandResult(handled=True, response=f"切换失败: {e}")

        # 同步内存中的 settings 和 LLM
        a.settings.model.model_id = new_model
        try:
            a.llm = a._create_llm()
            a.invalidate_agent_cache()
            note = "已重建 LLM 客户端并清 Agent graph 缓存，下一轮对话生效。"
        except Exception as e:
            note = (
                f"yaml 已写入但运行时重建 LLM 失败: {e}\n"
                "建议重启 Gateway。"
            )

        return CommandResult(
            handled=True,
            response=f"模型已切换为 `{new_model}`。\n\n{note}",
        )

    async def _cmd_reload(self, arg: str, session_id: str) -> CommandResult:
        """重读 yaml 配置 + 清 Agent graph 缓存"""
        from server.config.settings import reload_settings

        try:
            new_settings = reload_settings()
        except Exception as e:
            return CommandResult(handled=True, response=f"重载失败: {e}")

        self.agent.settings = new_settings
        extra_notes: list[str] = []
        try:
            self.agent.llm = self.agent._create_llm()
        except Exception as e:
            return CommandResult(
                handled=True,
                response=(
                    f"配置已重载但重建 LLM 失败: {e}\n"
                    "建议重启 Gateway。"
                ),
            )

        # 清 Agent graph 缓存
        self.agent.invalidate_agent_cache()

        # 热重载 Scheduler
        sched = getattr(self.agent, "scheduler", None)
        if sched is not None:
            try:
                sched.reload(new_settings.scheduler)
                extra_notes.append(
                    f"Scheduler 已热重载（{len(sched.list_jobs())} 个任务）。"
                )
            except Exception as e:
                extra_notes.append(f"Scheduler 重载失败: {e}")

        tail = ("\n" + "\n".join(extra_notes)) if extra_notes else ""
        return CommandResult(
            handled=True,
            response=(
                "已重载 yaml 配置并重建 LLM 客户端。" + tail + "\n\n"
                "Gateway 端口、checkpointer 路径等初始化时敏感的字段若已改变，"
                "仍需**重启 Gateway** 才能生效。"
            ),
        )

    async def _cmd_edit(self, arg: str, session_id: str) -> CommandResult:
        """在终端内用 $EDITOR 打开 yaml，编辑完成后自动 /reload"""
        import asyncio
        import os
        import subprocess
        import sys
        from server.config.config_io import get_config_path

        # 非终端环境禁用交互编辑器
        if not (sys.stdin.isatty() and sys.stdout.isatty()):
            return CommandResult(
                handled=True,
                response=(
                    "`/edit` 仅在本地终端 TUI 中可用。"
                    "在远程调用（WebSocket / Webhook）中请改用 "
                    "`/config <key> <value>` 或 CLI 的 `config edit`。"
                ),
            )

        path = get_config_path()
        editor = os.getenv("EDITOR") or os.getenv("VISUAL") or "vi"

        def _run_editor() -> int:
            try:
                return subprocess.call([editor, str(path)])
            except FileNotFoundError:
                return -1

        # 编辑器阻塞运行，放到线程池
        rc = await asyncio.get_running_loop().run_in_executor(None, _run_editor)
        if rc == -1:
            return CommandResult(
                handled=True,
                response=f"未找到编辑器 {editor!r}，请设置 $EDITOR。",
            )

        # 编辑后自动 reload
        return await self._cmd_reload("", session_id)

    # helpers

    def _handle_enum_toggle(
        self,
        arg: str,
        session_id: str,
        key: str,
        allowed: tuple[str, ...],
        label: str,
    ) -> CommandResult:
        if not arg:
            sess = self.agent.session_mgr.get_session(session_id)
            current = sess.metadata.get(key, allowed[0]) if sess else allowed[0]
            return CommandResult(
                handled=True,
                response=(
                    f"{label} 当前: `{current}`\n"
                    f"可选值: `{', '.join(allowed)}`"
                ),
            )
        val = arg.lower()
        if val not in allowed:
            return CommandResult(
                handled=True,
                response=f"无效值 `{arg}`。可选: {', '.join(allowed)}",
            )
        self._set_session_meta(session_id, key, val)
        return CommandResult(handled=True, response=f"{label} 已设为 `{val}`。")

    def _set_session_meta(self, session_id: str, key: str, value) -> None:
        self.agent.session_mgr.touch(session_id, **{key: value})

    # 长期记忆

    async def _cmd_memories(self, arg: str, session_id: str) -> CommandResult:
        """列出或清空长期记忆"""
        store = getattr(self.agent, "memory_store", None)
        if store is None:
            return CommandResult(
                handled=True,
                response="（记忆系统未启用。yaml agent.memory.enabled: true 开启）",
            )
        arg = arg.strip()
        if arg.startswith("clear"):
            tail = arg[len("clear"):].strip()
            if tail != "yes":
                return CommandResult(
                    handled=True,
                    response="⚠️ 将软删所有长期记忆。确认请输入 `/memories clear yes`",
                )
            # 软删全部未删项
            try:
                mems = store.list_recent(limit=10000)
                cnt = 0
                for m in mems:
                    if store.delete(m.id):
                        cnt += 1
                return CommandResult(
                    handled=True,
                    response=f"已软删 {cnt} 条记忆（可在 DB 里恢复）。",
                )
            except Exception as e:
                return CommandResult(handled=True, response=f"清空失败: {e}")

        kinds = None
        if arg.startswith("kind="):
            kinds = [arg[5:].strip()]

        mems = store.list_recent(limit=20, kinds=kinds)
        if not mems:
            return CommandResult(handled=True, response="暂无长期记忆。")
        lines = [f"近 {len(mems)} 条长期记忆："]
        for m in mems:
            day = m.updated_at[:10]
            lines.append(f"  [{m.id}] [{m.kind}] {m.text}  ({day}, hits={m.hits})")
        return CommandResult(handled=True, response="\n".join(lines))

    async def _cmd_recall(self, arg: str, session_id: str) -> CommandResult:
        """检索长期记忆"""
        store = getattr(self.agent, "memory_store", None)
        if store is None:
            return CommandResult(
                handled=True, response="（记忆系统未启用）",
            )
        q = arg.strip()
        if not q:
            return CommandResult(
                handled=True, response="用法: `/recall <查询词>`",
            )
        mems = store.search(q, k=5)
        if not mems:
            return CommandResult(handled=True, response="（无匹配）")
        lines = [f"检索到 {len(mems)} 条："]
        for m in mems:
            day = m.updated_at[:10]
            lines.append(
                f"  [{m.id}] [{m.kind}] {m.text}  "
                f"({day}, score={m.score:.3f})"
            )
        return CommandResult(handled=True, response="\n".join(lines))

    async def _cmd_recall_at(self, arg: str, session_id: str) -> CommandResult:
        """时间感知三元组查询"""
        store = getattr(self.agent, "memory_store", None)
        if store is None:
            return CommandResult(
                handled=True, response="（记忆系统未启用）",
            )
        parts = arg.strip().split()
        ts = ""
        subject = None
        predicate = None
        for tok in parts:
            if tok.startswith("subject="):
                subject = tok[8:].strip() or None
            elif tok.startswith("predicate="):
                predicate = tok[10:].strip() or None
            elif not ts:
                ts = tok
        # 补全短日期
        at = ts or None
        if at and len(at) == 10 and at.count("-") == 2:
            at = at + "T00:00:00Z"

        mems = store.triples_at(
            ts=at, subject=subject, predicate=predicate, limit=50,
        )
        if not mems:
            return CommandResult(
                handled=True,
                response=f"（{ts or '当前'} 无有效三元组）",
            )
        lines = [f"在 {ts or '当前'} 有效的 {len(mems)} 条事实："]
        for m in mems:
            expired = f" 过期于 {m.valid_until[:10]}" if m.valid_until else ""
            lines.append(
                f"  [{m.id}] {m.subject} -{m.predicate}-> {m.object}"
                f"  (自 {m.valid_from[:10]}{expired})"
            )
        return CommandResult(handled=True, response="\n".join(lines))

    async def _cmd_forget(self, arg: str, session_id: str) -> CommandResult:
        """软删某条长期记忆"""
        store = getattr(self.agent, "memory_store", None)
        if store is None:
            return CommandResult(
                handled=True, response="（记忆系统未启用）",
            )
        try:
            mid = int(arg.strip())
        except ValueError:
            return CommandResult(
                handled=True, response="用法: `/forget <数字 id>`",
            )
        ok = store.delete(mid)
        return CommandResult(
            handled=True,
            response=f"已删除 id={mid}" if ok else f"未找到 id={mid} 或已删除",
        )

    # 插件管理

    async def _cmd_plugin(self, arg: str, session_id: str) -> CommandResult:
        """OpenClaw 插件管理"""
        registry = getattr(self.agent, "plugin_registry", None)
        if registry is None:
            return CommandResult(
                handled=True, response="（插件系统未启用）"
            )

        parts = arg.split(None, 2)
        sub = (parts[0].lower() if parts else "list").strip()
        rest = parts[1] if len(parts) >= 2 else ""

        if sub in ("list", "ls", ""):
            plugins = registry.list()
            if not plugins:
                return CommandResult(
                    handled=True,
                    response=(
                        "当前未发现插件。放到 `<workspace>/.openclaw/plugins/<id>/` "
                        "或 `~/.openclaw/plugins/<id>/` 下，再 `/plugin install` 即可。"
                    ),
                )
            lines = ["## 插件列表", "", "| id | 状态 | 类型 | 版本 | 来源 |",
                     "| --- | --- | --- | --- | --- |"]
            for s in plugins:
                lines.append(
                    f"| `{s.id}` | {s.status} | {s.manifest.manifest_kind} | "
                    f"{s.manifest.version or '-'} | `{s.manifest.source}` |"
                )
            return CommandResult(handled=True, response="\n".join(lines))

        if sub in ("show", "inspect", "info"):
            if not rest:
                return CommandResult(handled=True, response="用法: `/plugin show <id>`")
            state = registry.get(rest.strip())
            if state is None:
                return CommandResult(handled=True, response=f"未找到插件: `{rest}`")
            m = state.manifest
            lines = [
                f"## 插件 `{state.id}`",
                "",
                f"- **name**: {m.name}",
                f"- **version**: {m.version or '-'}",
                f"- **description**: {m.description or '-'}",
                f"- **manifest**: `{m.manifest_file}` ({m.manifest_kind})",
                f"- **kind**: {m.kind or '-'}",
                f"- **status**: {state.status}"
                + (f" ({state.reason_disabled})" if state.reason_disabled else ""),
                f"- **providers**: {m.providers or '-'}",
                f"- **skills**: {m.skills or '-'}",
                f"- **channels**: {m.channels or '-'}",
                f"- **command aliases**: {[c.get('name') for c in m.command_aliases] or '-'}",
            ]
            if state.warnings:
                lines.append("")
                lines.append("### 诊断告警")
                for w in state.warnings:
                    lines.append(f"- ⚠ {w}")
            if m.error:
                lines.append("")
                lines.append(f"**错误**: {m.error}")
            return CommandResult(handled=True, response="\n".join(lines))

        if sub == "enable":
            if not rest:
                return CommandResult(handled=True, response="用法: `/plugin enable <id>`")
            ok = registry.enable(rest.strip())
            return CommandResult(
                handled=True,
                response=f"已启用 `{rest}`" if ok else f"未找到插件 `{rest}`"
            )

        if sub == "disable":
            if not rest:
                return CommandResult(handled=True, response="用法: `/plugin disable <id>`")
            ok = registry.disable(rest.strip())
            return CommandResult(
                handled=True,
                response=f"已禁用 `{rest}`" if ok else f"未找到插件 `{rest}`"
            )

        if sub == "install":
            if not rest:
                return CommandResult(
                    handled=True,
                    response=(
                        "用法: `/plugin install <spec>`\n"
                        "支持：本地路径 (`./path` / 绝对路径)、git URL / `<owner>/<repo>`、"
                        "npm 包 (`@openclaw/voice-call`, `clawhub:<pkg>`)"
                    ),
                )
            from server.core.plugin_installer import PluginInstaller
            inst = PluginInstaller(self.agent.settings)
            res = inst.install(rest.strip())
            if res.ok:
                # 重扫 registry
                registry.scan()
                return CommandResult(
                    handled=True,
                    response=f"✅ 安装 `{res.plugin_id}` 成功：{res.message}\n目标: `{res.target_dir}`",
                )
            return CommandResult(
                handled=True, response=f"❌ 安装失败：{res.message}"
            )

        if sub == "uninstall":
            if not rest:
                return CommandResult(handled=True, response="用法: `/plugin uninstall <id>`")
            from server.core.plugin_installer import PluginInstaller
            inst = PluginInstaller(self.agent.settings)
            res = inst.uninstall(rest.strip())
            if res.ok:
                registry.scan()
                return CommandResult(
                    handled=True,
                    response=f"✅ 卸载 `{res.plugin_id or rest}`：{res.message}"
                )
            return CommandResult(
                handled=True, response=f"❌ 卸载失败：{res.message}"
            )

        if sub in ("doctor", "diagnose"):
            plugins = registry.list()
            lines = ["## 插件诊断"]
            if not plugins:
                lines.append("\n未发现插件。")
            any_warn = False
            for s in plugins:
                header = f"\n### `{s.id}` ({s.status})"
                if s.manifest.error:
                    any_warn = True
                    lines.append(header)
                    lines.append(f"- ❌ {s.manifest.error}")
                    continue
                if s.warnings:
                    any_warn = True
                    lines.append(header)
                    for w in s.warnings:
                        lines.append(f"- ⚠ {w}")
            if not any_warn and plugins:
                lines.append("\n全部插件无告警 ✅")
            return CommandResult(handled=True, response="\n".join(lines))

        return CommandResult(
            handled=True,
            response=(
                f"未知子命令 `{sub}`。\n\n"
                "用法: `/plugin list|show <id>|enable <id>|disable <id>|install <spec>"
                "|uninstall <id>|doctor`"
            ),
        )

    # Computer Use

    async def _cmd_computer_use(self, arg: str, session_id: str) -> CommandResult:
        """/computer_use [on|off|status] 或 /computer_use <layer> [on|off]

        layer ∈ {desktop, browser, loop}
        """
        from server.config.config_io import set_value
        cfg = self.agent.settings.computer_use

        parts = arg.strip().split()

        # 无参数或 status 时打印状态
        if not parts or parts[0].lower() == "status":
            lines = [
                "## Computer Use 状态",
                "",
                f"- **总开关**: `{cfg.enabled}`" + (" ✅" if cfg.enabled else " ❌"),
                f"- **desktop**: `{cfg.desktop_enabled}` (鼠标/键盘/截图)",
                f"- **browser**: `{cfg.browser_enabled}` (Playwright `{cfg.browser_mode}`)",
                f"- **loop_agent**: `{cfg.loop_agent_enabled}` (VLM 循环)",
                "",
                f"- **app_allowlist**: `{cfg.app_allowlist or '[]（全开）'}`",
                f"- **app_denylist**: {len(cfg.app_denylist)} 条",
                f"- **domain_denylist**: {len(cfg.domain_denylist)} 条",
                f"- **审计日志**: `{cfg.audit_log_path}` "
                f"({'缩略图开' if cfg.audit_thumbnails else '仅文本'})",
                "",
                "用法:",
                "- `/computer_use on|off` — 总开关",
                "- `/computer_use desktop on|off` — 分层切换",
                "- `/computer_use browser on|off`",
                "- `/computer_use loop on|off`",
            ]
            return CommandResult(handled=True, response="\n".join(lines))

        # 单参数切总开关
        if len(parts) == 1 and parts[0].lower() in ("on", "off"):
            val = parts[0].lower() == "on"
            try:
                set_value("agent.computer_use.enabled", val)
            except Exception as e:
                return CommandResult(handled=True, response=f"写入 yaml 失败: {e}")
            cfg.enabled = val
            self.agent.invalidate_agent_cache()
            return CommandResult(
                handled=True,
                response=(
                    f"Computer Use 总开关 → `{val}`\n"
                    f"下一轮对话生效。**{'⚠️ 请注意 AI 现在可以操作你的电脑' if val else '已禁用'}**"
                ),
            )

        # 双参数切分层
        if len(parts) == 2 and parts[1].lower() in ("on", "off"):
            layer = parts[0].lower()
            val = parts[1].lower() == "on"
            field_map = {
                "desktop": ("agent.computer_use.desktop_enabled", "desktop_enabled"),
                "browser": ("agent.computer_use.browser_enabled", "browser_enabled"),
                "loop": ("agent.computer_use.loop_agent_enabled", "loop_agent_enabled"),
            }
            if layer not in field_map:
                return CommandResult(
                    handled=True,
                    response=f"未知 layer `{layer}`。可选: desktop / browser / loop",
                )
            yaml_key, attr = field_map[layer]
            try:
                set_value(yaml_key, val)
            except Exception as e:
                return CommandResult(handled=True, response=f"写入 yaml 失败: {e}")
            setattr(cfg, attr, val)
            self.agent.invalidate_agent_cache()
            return CommandResult(
                handled=True,
                response=f"Computer Use `{layer}` → `{val}`，下一轮对话生效。",
            )

        return CommandResult(
            handled=True,
            response=(
                f"用法错误。\n"
                "- `/computer_use` — 查看状态\n"
                "- `/computer_use on|off` — 总开关\n"
                "- `/computer_use <desktop|browser|loop> on|off` — 分层"
            ),
        )
