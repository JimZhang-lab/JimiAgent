'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/server/core/agent.py
Description: Agent 引擎。

'''
import logging
import datetime
from typing import AsyncGenerator, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import StructuredTool

from server.config.settings import get_settings, Settings
from server.core.workspace import Workspace
from server.core.skill_retriever import SkillRetriever
from server.core.memory import MemoryManager
from server.core.session import SessionManager
from server.core.commands import CommandDispatcher, CommandResult, think_prompt_for

logger = logging.getLogger(__name__)


def _stringify_tool_output(out: object) -> str:
    """把工具返回值压成前端可展示的字符串。"""
    if out is None:
        return ""
    # ToolMessage / AIMessage 等对象优先取 content
    content = getattr(out, "content", None)
    if content is not None:
        if isinstance(content, str):
            return content
        if isinstance(content, (list, tuple)):
            # ContentBlock 只做文本拼接
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict):
                    parts.append(part.get("text") or str(part))
                else:
                    parts.append(str(part))
            return "".join(parts)
        return str(content)
    if isinstance(out, str):
        return out
    if isinstance(out, bytes):
        try:
            return out.decode("utf-8", errors="replace")
        except Exception:
            return str(out)
    try:
        import json as _json
        return _json.dumps(out, ensure_ascii=False, indent=2)
    except Exception:
        return str(out)


def _build_datetime_tool() -> StructuredTool:
    """构建内置日期时间工具。"""

    def get_current_datetime() -> str:
        """获取当前日期、时间和星期信息"""
        now = datetime.datetime.now().astimezone()
        return now.strftime("当前时间: %Y-%m-%d %H:%M:%S %Z (%A)")

    return StructuredTool.from_function(
        func=get_current_datetime,
        name="datetime_info",
        description="获取当前日期、时间和星期信息（无参数）",
    )


def _build_session_tools(agent: "JimiAgent") -> list[StructuredTool]:
    """构建会话管理工具。"""

    def list_recent_sessions(limit: int = 10) -> str:
        """列出最近的会话（按更新时间倒序）。"""
        limit = max(1, min(int(limit or 10), 50))
        sessions = agent.session_mgr.list_sessions()[:limit]
        if not sessions:
            return "(暂无会话)"
        lines = ["最近会话："]
        for s in sessions:
            lines.append(
                f"- `{s.id}`  {s.title}  (消息数: {s.message_count})"
            )
        return "\n".join(lines)

    def get_session_info(session_id: str) -> str:
        """获取指定会话的元信息：标题、消息数、创建/更新时间、think/verbose 配置。"""
        sess = agent.session_mgr.get_session(session_id)
        if sess is None:
            # 支持 8 位前缀
            for s in agent.session_mgr.list_sessions():
                if s.id.startswith(session_id):
                    sess = s
                    break
        if sess is None:
            return f"未找到会话: {session_id}"
        created = datetime.datetime.fromtimestamp(sess.created_at).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        updated = datetime.datetime.fromtimestamp(sess.updated_at).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        think = sess.metadata.get(
            "think_level", agent.settings.session.default_think_level
        )
        verbose = sess.metadata.get("verbose", "off")
        usage = sess.metadata.get("usage", "off")
        return (
            f"会话 `{sess.id}`\n"
            f"- 标题: {sess.title}\n"
            f"- 消息数: {sess.message_count}\n"
            f"- 创建: {created}\n"
            f"- 更新: {updated}\n"
            f"- think: `{think}`  verbose: `{verbose}`  usage: `{usage}`"
        )

    def create_new_session(title: str = "新对话") -> str:
        """创建一个新会话但不切换（切换须由 Gateway/CLI 的 /new 命令完成）。"""
        sess = agent.session_mgr.create_session(title.strip() or "新对话")
        return (
            f"已创建新会话 `{sess.id}`（标题: {sess.title}）。\n"
            "注意: 当前对话仍在原会话中，如需切换请用 /new 斜杠命令。"
        )

    def _run_coro(coro):
        """在同步工具里安全跑 coroutine。"""
        import asyncio as _aio
        try:
            _aio.get_running_loop()
            # 当前线程已有 loop 时改用新 loop
            new_loop = _aio.new_event_loop()
            try:
                return new_loop.run_until_complete(coro)
            finally:
                new_loop.close()
        except RuntimeError:
            # 当前线程没有运行中的 loop
            return _aio.run(coro)

    def sessions_history(session_id: str, limit: int = 10) -> str:
        """读取指定会话的最近消息（HumanMessage / AIMessage），便于跨会话引用。"""
        # 前缀匹配
        target_id = session_id
        for s in agent.session_mgr.list_sessions():
            if s.id.startswith(session_id):
                target_id = s.id
                break

        async def _read():
            config = {"configurable": {"thread_id": target_id}}
            try:
                state_graph = agent._get_state_graph()
                snapshot = await state_graph.aget_state(config)
                messages = (snapshot.values or {}).get("messages", [])
            except Exception:
                return f"无法读取会话 {target_id} 的历史"
            if not messages:
                return f"会话 {target_id} 暂无消息"
            recent = messages[-int(limit):]
            lines = [f"会话 `{target_id}` 最近 {len(recent)} 条消息:"]
            for msg in recent:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                text = msg.content[:300] if msg.content else "(empty)"
                lines.append(f"  [{role}] {text}")
            return "\n".join(lines)

        return _run_coro(_read())

    def sessions_send(session_id: str, message: str) -> str:
        """向指定会话发送一条消息并获取 Agent 回复（跨会话主动推送）。"""
        async def _send():
            resp, sid = await agent.chat(message, session_id)
            return f"已向会话 {sid} 发送消息，Agent 回复:\n{resp[:500]}"

        return _run_coro(_send())

    def sessions_spawn(title: str, initial_message: str) -> str:
        """创建一个新会话并立即执行首轮对话（派生子任务）。"""
        sess = agent.session_mgr.create_session(title.strip() or "子任务")

        async def _run():
            resp, sid = await agent.chat(initial_message, sess.id)
            return (
                f"已创建并执行新会话 `{sid}`（标题: {sess.title}）\n"
                f"Agent 回复:\n{resp[:500]}"
            )

        return _run_coro(_run())

    return [
        StructuredTool.from_function(
            func=list_recent_sessions,
            name="list_recent_sessions",
            description=(
                "列出最近的会话（按更新时间倒序）。当用户问起之前聊过的内容、"
                "历史主题时使用。可选参数 limit 控制返回条数。"
            ),
        ),
        StructuredTool.from_function(
            func=list_recent_sessions,
            name="sessions_list",
            description=(
                "列出最近的会话（OpenClaw 兼容别名，等价于 list_recent_sessions）。"
                "参数 limit 控制返回条数。"
            ),
        ),
        StructuredTool.from_function(
            func=get_session_info,
            name="get_session_info",
            description=(
                "获取指定会话的元信息（标题、消息数、时间、think/verbose/usage）。"
                "参数 session_id 支持 UUID 前 8 位匹配。"
            ),
        ),
        StructuredTool.from_function(
            func=create_new_session,
            name="create_new_session",
            description=(
                "创建一个新会话（仅创建，不切换当前对话）。"
                "当用户明确说'再开一个会话记录' / '把这个话题分流出去'时使用。"
                "参数 title 是新会话标题。"
            ),
        ),
        StructuredTool.from_function(
            func=sessions_history,
            name="sessions_history",
            description=(
                "读取指定会话的最近消息历史。当用户说'看看上次聊了什么'或"
                "需要跨会话引用时使用。参数 session_id 支持前缀匹配。"
            ),
        ),
        StructuredTool.from_function(
            func=sessions_send,
            name="sessions_send",
            description=(
                "向指定会话发送消息并获取 Agent 回复（跨会话主动推送）。"
                "参数 session_id 和 message。"
            ),
        ),
        StructuredTool.from_function(
            func=sessions_spawn,
            name="sessions_spawn",
            description=(
                "创建一个新会话并立即执行首轮对话（派生子任务）。"
                "当用户说'帮我开一个新任务去处理这个'时使用。"
                "参数 title 和 initial_message。"
            ),
        ),
    ]


def _build_memory_tools(agent: "JimiAgent") -> list[StructuredTool]:
    """跨会话长期记忆工具：remember / recall / forget。"""

    def remember(text: str, kind: str = "semantic") -> str:
        """把一条重要事实 / 偏好 / 规则写入长期记忆（跨会话可检索）。"""
        if agent.memory_store is None:
            return "（记忆系统未启用）"
        try:
            mid = agent.memory_store.add(text=text, kind=kind, source_session="")
        except Exception as e:
            return f"记忆写入失败: {e}"
        if mid == 0:
            return "（文本为空，未写入）"
        return f"已记住 (id={mid})"

    def recall(query: str, k: int = 5) -> str:
        """跨会话检索长期记忆。当用户问起"你记不记得"、"我之前说过"、
        "我叫什么名字"等问题时使用。"""
        if agent.memory_store is None:
            return "（记忆系统未启用）"
        mems = agent.memory_store.search(query, k=int(k))
        if not mems:
            return "（无匹配记忆）"
        lines = [f"检索到 {len(mems)} 条:"]
        for m in mems:
            day = m.updated_at[:10]
            lines.append(f"- [{m.kind}] {m.text}  (id={m.id}, {day})")
        return "\n".join(lines)

    def forget(memory_id: int) -> str:
        """软删除一条长期记忆。删前通常先用 recall 找到对应 id。"""
        if agent.memory_store is None:
            return "（记忆系统未启用）"
        ok = agent.memory_store.delete(int(memory_id))
        return f"已删除 id={memory_id}" if ok else f"未找到 id={memory_id} 或已删除"

    def recall_at(
        timestamp: str,
        subject: str = "",
        predicate: str = "",
    ) -> str:
        """时间感知查询：返回给定时间戳下有效的三元组事实。"""
        if agent.memory_store is None:
            return "（记忆系统未启用）"
        ts = timestamp.strip() or None
        # YYYY-MM-DD 补全为 UTC 零点
        if ts and len(ts) == 10 and ts.count("-") == 2:
            ts = ts + "T00:00:00Z"
        try:
            mems = agent.memory_store.triples_at(
                ts=ts,
                subject=subject.strip() or None,
                predicate=predicate.strip() or None,
            )
        except Exception as e:
            return f"recall_at 失败: {e}"
        if not mems:
            return "（该时点无有效三元组）"
        lines = [f"在 {ts or 'now'} 有效的 {len(mems)} 条事实:"]
        for m in mems:
            expired = f" 过期于 {m.valid_until[:10]}" if m.valid_until else ""
            lines.append(
                f"  [{m.id}] {m.subject} -{m.predicate}-> {m.object}"
                f"  (自 {m.valid_from[:10]}{expired})"
            )
        return "\n".join(lines)

    return [
        StructuredTool.from_function(
            func=remember,
            name="remember",
            description=(
                "把一条重要事实/偏好/规则写入跨会话长期记忆。"
                "用户说'记住 X' / '以后都 Y' / 告诉你长期有效的信息时使用。"
                "参数 text（事实文本）、kind（semantic/episodic/procedural）。"
            ),
        ),
        StructuredTool.from_function(
            func=recall,
            name="recall",
            description=(
                "跨会话检索长期记忆。用户问'你记不记得' / '我上次说过什么' / "
                "'我叫什么' 时使用。参数 query 是检索词，k 是返回条数。"
            ),
        ),
        StructuredTool.from_function(
            func=forget,
            name="forget",
            description=(
                "软删除一条长期记忆。参数 memory_id（int）；通常先用 recall 拿 id。"
            ),
        ),
        StructuredTool.from_function(
            func=recall_at,
            name="recall_at",
            description=(
                "时间感知检索：给定时间戳下有效的三元组事实（subject -predicate-> object）。"
                "当用户问'我上个月的 CEO 是谁' / '2026-01 时我负责什么' 这类"
                "带时间点的问题时使用。参数 timestamp（空代表现在；支持 YYYY-MM-DD 或 ISO 全时间），"
                "可选 subject / predicate 过滤。"
            ),
        ),
    ]


class JimiAgent:
    """整合 prompt、skills、memory、sessions 与 LangGraph 的主引擎。"""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        # 基础组件
        self.workspace = Workspace()
        self.memory = MemoryManager(self.settings.memory_abs_path)
        self.session_mgr = SessionManager(
            self.settings.memory_abs_path.parent
        )

        # 插件注册表
        from server.core.plugin_registry import PluginRegistry
        self.plugin_registry = PluginRegistry(self.settings)
        try:
            self.plugin_registry.scan()
        except Exception as e:
            logger.warning(f"PluginRegistry 扫描失败（忽略）: {e}")

        # Skills 召回
        self.skill_retriever = SkillRetriever(
            skills_dir=self.workspace.get_skills_dir(),
            persist_dir=self.settings.skill_index_abs_path,
            top_k=self.settings.skills.top_k,
            similarity_threshold=self.settings.skills.similarity_threshold,
            extra_skill_dirs=self.plugin_registry.active_skill_dirs(),
        )

        # 主 LLM
        self.llm = self._create_llm()
        # VLM 未配置时复用主 LLM
        self.vlm = self._create_vlm()

        # 长期记忆，可被插件槽位替换
        from server.core.memory_store import MemoryStore
        from server.core.plugin_slots import resolve_memory_slot
        self.memory_store = None
        if self.settings.memory.enabled:
            strategy, inst = resolve_memory_slot(
                self.plugin_registry, self.settings,
            )
            try:
                if strategy == "none":
                    self.memory_store = None
                elif strategy.startswith("plugin:") and inst is not None:
                    # 插件实例在这里补做 initialize
                    if hasattr(inst, "initialize"):
                        try:
                            inst.initialize()
                        except Exception as e:
                            logger.warning(
                                f"插件 memory slot initialize 失败，回退 builtin: {e}"
                            )
                            inst = None
                    if inst is not None:
                        self.memory_store = inst
                        logger.info(f"memory slot = {strategy}")
                if self.memory_store is None and strategy != "none":
                    # 插件失败时回退 builtin
                    self.memory_store = MemoryStore(self.settings)
                    self.memory_store.initialize()
            except Exception as e:
                logger.warning(f"MemoryStore 初始化失败（禁用）: {e}")
                self.memory_store = None

        # 内置工具
        from server.core.file_tools import build_file_tools, build_bash_tools
        from server.core.computer_use import build_computer_use_tools
        self._builtin_tools = [
            _build_datetime_tool(),
            *_build_session_tools(self),
            *build_file_tools(self),
            *build_bash_tools(self),  # 默认关闭
            *build_computer_use_tools(self),  # 默认关闭
        ]
        if self.memory_store is not None:
            self._builtin_tools.extend(_build_memory_tools(self))

        # Agent graph 缓存
        self._agent_graph_cache: dict[tuple, object] = {}

        # 读写 state 的专用 graph
        self._state_graph = None

        # 斜杠命令分发器
        self.commands = CommandDispatcher(self)

        # 由 Gateway 注入，可选
        self.scheduler = None

        # Evolver
        from server.core.evolver import EvolutionEngine
        gep_dir = self.settings.memory_abs_path.parent / "gep"
        self._tool_log_path = self.settings.memory_abs_path.parent / "tool_log.jsonl"
        self.evolver = EvolutionEngine(
            gep_dir=gep_dir,
            tool_log_path=self._tool_log_path,
            memory_store=self.memory_store,  # 让 evolver 可扫 procedural 记忆
        )

        # Pairing / Allowlist
        from server.plugin.channels.pairing import PairingStore
        allowlist_path = self.settings.memory_abs_path.parent / "allowlist.json"
        # 从 yaml 读取静态白名单
        channel_defaults: dict[str, list[str]] = {}
        for ch_name, ch_cfg in (self.settings.channels.raw or {}).items():
            if isinstance(ch_cfg, dict):
                allow = ch_cfg.get("allowFrom") or []
                if allow:
                    channel_defaults[ch_name] = list(allow)
        self.pairing = PairingStore(allowlist_path, channel_defaults=channel_defaults)

        self._initialized = False
        logger.info("JimiAgent 引擎初始化完成")

    def _get_state_graph(self):
        """懒构建一个无工具的 graph，仅用于访问/更新 session state"""
        if self._state_graph is None:
            from langgraph.prebuilt import create_react_agent
            self._state_graph = create_react_agent(
                model=self.llm,
                tools=[],
                checkpointer=self.memory.get_checkpointer(),
                prompt="",
            )
        return self._state_graph

    def _create_llm(self):
        """创建 LLM 实例。

        若 settings.model.fallbacks 非空，主模型会被 LangChain 的
        Runnable.with_fallbacks 包装，主模型抛异常（API error / rate-limit 等）
        时自动切换到下一个备选，按配置顺序逐个尝试。
        """
        primary = self._build_single_llm(
            provider=self.settings.model.provider,
            model_id=self.settings.model.model_id,
            temperature=self.settings.model.temperature,
            max_tokens=self.settings.model.max_tokens,
            streaming=self.settings.model.streaming,
            api_key=self.settings.api_key or self.settings.model.api_key,
            base_url=self.settings.base_url or self.settings.model.base_url,
        )

        fallback_specs = self.settings.model.fallbacks or []
        if not fallback_specs:
            return primary

        fallbacks = []
        for idx, spec in enumerate(fallback_specs):
            if not isinstance(spec, dict):
                logger.warning(f"忽略非法 fallback[{idx}]: {spec!r}")
                continue
            try:
                fb = self._build_single_llm(
                    provider=spec.get("provider", self.settings.model.provider),
                    model_id=spec.get("model_id"),
                    temperature=spec.get("temperature", self.settings.model.temperature),
                    max_tokens=spec.get("max_tokens", self.settings.model.max_tokens),
                    streaming=spec.get("streaming", self.settings.model.streaming),
                    api_key=spec.get("api_key", ""),
                    base_url=spec.get("base_url", ""),
                )
                fallbacks.append(fb)
                logger.info(
                    f"已注册 fallback 模型[{idx}]: "
                    f"{spec.get('provider')}/{spec.get('model_id')}"
                )
            except Exception as e:
                logger.warning(f"构造 fallback[{idx}] 失败，跳过: {e}")

        if not fallbacks:
            return primary

        # 默认对所有异常做 fallback
        return primary.with_fallbacks(fallbacks)

    def _create_vlm(self):
        """创建 VLM 实例。"""
        vcfg = self.settings.vlm
        if not (vcfg.model_id or vcfg.provider):
            return self.llm  # 回退到主 LLM

        eff = self.settings.effective_vlm
        try:
            return self._build_single_llm(
                provider=eff.provider,
                model_id=eff.model_id,
                temperature=eff.temperature,
                max_tokens=eff.max_tokens,
                streaming=eff.streaming,
                api_key=eff.api_key,
                base_url=eff.base_url,
            )
        except Exception as e:
            logger.warning(f"VLM 初始化失败（回退到 LLM）: {e}")
            return self.llm

    @staticmethod
    def _build_single_llm(
        provider: str,
        model_id: str,
        temperature: float,
        max_tokens: int,
        streaming: bool,
        api_key: str,
        base_url: str,
    ):
        """根据 provider 构造单个 Chat 模型实例（不含 fallback 包装）。"""
        provider = (provider or "openai").lower()
        kwargs: dict = dict(
            model=model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
        )
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        if provider == "openai":
            return ChatOpenAI(**kwargs)
        if provider == "anthropic":
            # 延迟 import，避免未安装时报错
            from langchain_anthropic import ChatAnthropic
            # ChatAnthropic 不接受 base_url
            akwargs = {
                k: v for k, v in kwargs.items()
                if k not in ("base_url",)
            }
            return ChatAnthropic(**akwargs)
        raise ValueError(f"不支持的 provider: {provider}")

    def _log_tool_call(
        self, tool_name: str, status: str,
        error: str = "", session_id: str = "",
    ) -> None:
        """追加一条 tool_log 记录。"""
        import datetime
        import json as _json
        try:
            self._tool_log_path.parent.mkdir(parents=True, exist_ok=True)
            rec = {
                "ts": datetime.datetime.utcnow().isoformat() + "Z",
                "tool": tool_name,
                "status": status,
                "session_id": session_id,
            }
            if error:
                rec["error"] = error[:500]  # 截断过长错误
            with self._tool_log_path.open("a", encoding="utf-8") as f:
                f.write(_json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug(f"tool_log 写入失败（不阻塞）: {e}")

    def _check_activation(self, message: str, session_id: str) -> bool:
        """检查 activation 策略。"""
        sess = self.session_mgr.get_session(session_id)
        mode = (sess.metadata.get("activation", "always") if sess else "always")
        if mode == "always":
            return True
        # mention 模式下要求出现 @agent_name
        agent_name = self.settings.agent_name.lower()
        return f"@{agent_name}" in message.lower()

    def _get_think_level(self, session_id: str) -> str:
        """读取当前会话的 think level（未设则取全局默认）"""
        sess = self.session_mgr.get_session(session_id)
        if sess:
            return sess.metadata.get(
                "think_level", self.settings.session.default_think_level
            )
        return self.settings.session.default_think_level

    def _build_system_prompt(self, think_level: str) -> str:
        """构造基础 prompt，并补上运行时环境与 think level。"""
        from server.core.file_tools import user_cwd  # 延迟 import 避免循环

        base = self.workspace.build_system_prompt()
        env_section = (
            "## 运行时环境\n"
            f"- 用户当前目录（CWD，即用户说「当前文件夹」「这里」时所指）："
            f"`{user_cwd()}`\n"
            f"- 项目 workspace（agent 的配置与 SKILL.md 所在）："
            f"`{self.settings.workspace_abs_path}`\n"
            "- 两者通常**不同**。处理用户请求时：\n"
            "  - 不确定路径，就先调用 `get_cwd` 工具确认\n"
            "  - `list_dir()` 无参数默认指向 CWD 而非 workspace"
        )
        return (
            f"{base}\n\n---\n\n{env_section}\n\n---\n\n"
            f"{think_prompt_for(think_level)}"
        )

    def _build_input_messages(
        self,
        user_message: str,
        images: list[str] | None = None,
    ) -> dict:
        """组装 LangGraph 输入消息。"""
        from langchain_core.messages import SystemMessage
        msgs: list = []
        if self.memory_store is not None:
            try:
                rule_limit = int(
                    self.settings.memory.procedural_inject_max or 10
                )
                rules = self.memory_store.build_rules_block(limit=rule_limit)
                if rules:
                    msgs.append(SystemMessage(content=rules))

                k = max(1, int(self.settings.memory.recall_max_inject or 5))
                block = self.memory_store.build_recall_block(user_message, k=k)
                if block:
                    msgs.append(SystemMessage(content=block))
            except Exception as e:
                logger.debug(f"memory 注入跳过: {e}")

        # 有图时改用 multi-part content
        if images:
            parts: list[dict] = []
            if user_message:
                parts.append({"type": "text", "text": user_message})
            for url in images:
                if not url:
                    continue
                parts.append({"type": "image_url", "image_url": {"url": url}})
            msgs.append(HumanMessage(content=parts))
        else:
            msgs.append(HumanMessage(content=user_message))
        return {"messages": msgs}

    # per-turn 抽取

    def _get_extract_lock(self, session_id: str):
        """懒加载每个 session 的抽取锁。"""
        import asyncio as _aio
        import weakref
        locks = getattr(self, "_extract_locks", None)
        if locks is None:
            locks = weakref.WeakValueDictionary()
            self._extract_locks = locks
        lock = locks.get(session_id)
        if lock is None:
            lock = _aio.Lock()
            locks[session_id] = lock
        return lock

    async def _hot_extract(self, session_id: str, user_msg: str, ai_msg: str) -> None:
        """per_turn 抽取：后台跑，失败/超时都静默。"""
        import asyncio as _aio
        from langchain_core.messages import HumanMessage as _Hm, AIMessage as _Am

        if self.memory_store is None:
            return
        lock = self._get_extract_lock(session_id)
        if lock.locked():
            # 同一会话已有抽取任务时直接跳过
            return
        timeout = max(1, int(self.settings.memory.extract_timeout_seconds or 10))
        messages = [_Hm(content=user_msg), _Am(content=ai_msg)]
        try:
            async with lock:
                await _aio.wait_for(
                    self.memory_store.extract_and_store(
                        llm=self.llm, messages=messages, session_id=session_id,
                    ),
                    timeout=timeout,
                )
        except _aio.TimeoutError:
            logger.debug("hot_extract 超时，跳过")
        except Exception as e:
            logger.debug(f"hot_extract 异常（忽略）: {e}")

    def _schedule_hot_extract(
        self, session_id: str, user_msg: str, ai_msg: str,
    ) -> None:
        """异步派发一次 hot_extract。"""
        import asyncio as _aio
        if self.memory_store is None:
            return
        if (self.settings.memory.extract_mode or "").lower() != "per_turn":
            return
        if not user_msg or not ai_msg:
            return
        try:
            loop = _aio.get_running_loop()
            # 懒加载 task 集合，便于 aclose 时 cancel
            tasks = getattr(self, "_hot_tasks", None)
            if tasks is None:
                tasks = set()
                self._hot_tasks = tasks
            task = loop.create_task(
                self._hot_extract(session_id, user_msg, ai_msg)
            )
            tasks.add(task)
            task.add_done_callback(tasks.discard)
        except RuntimeError:
            # 没有运行中的 loop 时直接跳过
            pass
        except Exception as e:
            logger.debug(f"schedule_hot_extract 失败（忽略）: {e}")

    def _build_agent(
        self,
        tools: list[StructuredTool],
        think_level: str,
        *,
        use_vlm: bool = False,
    ):
        """构建或复用 LangGraph ReAct Agent。"""
        from langgraph.prebuilt import create_react_agent

        cache_key = (
            tuple(sorted(t.name for t in tools)),
            think_level,
            "vlm" if use_vlm else "llm",
        )
        if cache_key in self._agent_graph_cache:
            return self._agent_graph_cache[cache_key]

        system_prompt = self._build_system_prompt(think_level)
        checkpointer = self.memory.get_checkpointer()

        model = self.vlm if use_vlm else self.llm
        graph = create_react_agent(
            model=model,
            tools=tools,
            checkpointer=checkpointer,
            prompt=system_prompt,
        )
        self._agent_graph_cache[cache_key] = graph
        logger.debug(
            f"构建 Agent graph, 工具: {list(cache_key[0])}, "
            f"think={think_level}, model={'vlm' if use_vlm else 'llm'}"
        )
        return graph

    def _get_tools_for_query(self, query: str) -> list[StructuredTool]:
        """获取当前查询的工具集：内置工具 + LlamaIndex 召回的 Skills"""
        try:
            retrieved_tools = self.skill_retriever.get_tools_for_query(query)
        except Exception as e:
            logger.warning(f"Skills 召回失败: {e}，回退为全量加载")
            retrieved_tools = self.skill_retriever.get_all_tools()

        all_tools = self._builtin_tools + retrieved_tools

        # 按 tool name 去重
        seen = set()
        unique_tools = []
        for t in all_tools:
            if t.name not in seen:
                seen.add(t.name)
                unique_tools.append(t)

        return unique_tools

    async def chat(
        self,
        message: str,
        session_id: str,
        *,
        images: list[str] | None = None,
    ) -> tuple[str, str]:
        """处理用户消息并返回完整响应。"""
        await self._ensure_initialized()

        # 1. 斜杠命令
        cmd_res = await self.commands.try_handle(message, session_id)
        if cmd_res.handled:
            return cmd_res.response, (cmd_res.new_session_id or session_id)

        # 2. activation
        if not self._check_activation(message, session_id):
            return "", session_id

        # 3. 自动 compact
        await self._maybe_auto_compact(session_id)

        # 4. 常规 Agent 调用
        think_level = self._get_think_level(session_id)
        tools = self._get_tools_for_query(message)
        use_vlm = bool(images)
        agent = self._build_agent(tools, think_level, use_vlm=use_vlm)

        config = {"configurable": {"thread_id": session_id}}
        input_messages = self._build_input_messages(message, images=images)

        result = await agent.ainvoke(input_messages, config=config)

        # 非流式接口无法继续确认时，直接返回提示
        paused = await self._check_interrupts(agent, config)
        if paused:
            first = paused[0]
            summary = (
                first.get("payload", {}).get("summary") if isinstance(
                    first.get("payload"), dict
                ) else str(first.get("payload", ""))
            ) or "未知操作"
            return (
                f"[等待确认] 存在 {len(paused)} 项危险操作需要批准：{summary}\n"
                "非流式 /api/chat 不支持实时 confirm，请改用流式接口 "
                "（/api/events 或 WebSocket /ws/chat），或在 yaml 里把 "
                "`safety.confirm_mode` 改为 `llm` 让 LLM 中转。",
                session_id,
            )

        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                self.session_mgr.increment_message_count(session_id)
                self._schedule_hot_extract(session_id, message, str(msg.content))
                return msg.content, session_id

        return "抱歉，我无法生成回复。", session_id

    async def chat_stream(
        self,
        message: str,
        session_id: str,
        *,
        images: list[str] | None = None,
    ) -> AsyncGenerator[dict, None]:
        """流式处理用户消息。"""
        await self._ensure_initialized()

        # 1. 斜杠命令
        cmd_res = await self.commands.try_handle(message, session_id)
        if cmd_res.handled:
            if cmd_res.new_session_id:
                yield {"type": "session", "session_id": cmd_res.new_session_id}
            yield {"type": "text", "content": cmd_res.response}
            return

        # 2. activation
        if not self._check_activation(message, session_id):
            return

        # 3. 自动 compact
        await self._maybe_auto_compact(session_id)

        # 4. 常规流式调用
        think_level = self._get_think_level(session_id)
        tools = self._get_tools_for_query(message)
        use_vlm = bool(images)
        agent = self._build_agent(tools, think_level, use_vlm=use_vlm)

        # trace 模式透出 LangGraph 事件
        sess = self.session_mgr.get_session(session_id)
        trace_on = (sess.metadata.get("trace", "off") == "on") if sess else False

        config = {"configurable": {"thread_id": session_id}}
        input_messages = self._build_input_messages(message, images=images)

        has_content = False
        ai_chunks: list[str] = []
        try:
            async for ev in self._iter_stream_events(
                agent, input_messages, config=config, trace_on=trace_on,
                session_id=session_id, ai_chunks=ai_chunks,
            ):
                if ev.get("type") == "text":
                    has_content = True
                yield ev
        except Exception as e:
            logger.error(f"Agent 流式执行失败: {e}", exc_info=True)
            yield {"type": "error", "content": f"{type(e).__name__}: {e}"}
            return

        # 若卡在 interrupt，则把确认需求继续透给前端
        paused = await self._check_interrupts(agent, config)
        if paused:
            for pay in paused:
                yield {"type": "confirm_required", **pay}
            # 流暂停，等外部调 chat_stream_resume
            return

        if not has_content:
            yield {"type": "text", "content": "抱歉，我无法生成回复。"}

        self.session_mgr.increment_message_count(session_id)

        # 流结束后后台抽取
        if ai_chunks:
            self._schedule_hot_extract(session_id, message, "".join(ai_chunks))

    async def _iter_stream_events(
        self,
        agent,
        input_or_command,
        *,
        config: dict,
        trace_on: bool,
        session_id: str,
        ai_chunks: list[str],
    ):
        """astream_events 统一解析（chat_stream 与 chat_stream_resume 共用）。"""
        async for event in agent.astream_events(
            input_or_command, config=config, version="v2",
        ):
            kind = event.get("event", "")

            if trace_on:
                yield {
                    "type": "trace",
                    "event": kind,
                    "name": event.get("name", ""),
                    "run_id": event.get("run_id", ""),
                }

            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and getattr(chunk, "content", None):
                    content = chunk.content
                    if isinstance(content, list):
                        content = "".join(
                            part.get("text", "") if isinstance(part, dict) else str(part)
                            for part in content
                        )
                    if content:
                        ai_chunks.append(content)
                        yield {"type": "text", "content": content}

            elif kind == "on_tool_start":
                tool_name = event.get("name", "unknown")
                self._log_tool_call(tool_name, "start", session_id=session_id)
                yield {"type": "tool", "name": tool_name}

            elif kind == "on_tool_end":
                tool_name = event.get("name", "unknown")
                self._log_tool_call(
                    tool_name, "ok",
                    session_id=session_id,
                )
                # 把工具结果统一压成字符串再透给前端
                out = event.get("data", {}).get("output")
                out_str = _stringify_tool_output(out)
                if out_str:
                    yield {
                        "type": "tool_result",
                        "name": tool_name,
                        "output": out_str,
                    }

            elif kind == "on_tool_error":
                tool_name = event.get("name", "unknown")
                err = event.get("data", {}).get("error")
                self._log_tool_call(
                    tool_name, "error",
                    error=str(err) if err else "",
                    session_id=session_id,
                )
                # 错误也通过 tool_result 透给前端
                yield {
                    "type": "tool_result",
                    "name": tool_name,
                    "output": f"[error] {err}" if err else "[error] 工具执行失败",
                }

    async def _check_interrupts(self, agent, config: dict) -> list[dict]:
        """读取 graph 当前 state 的 interrupts。"""
        try:
            state = await agent.aget_state(config)
        except Exception as e:
            logger.debug(f"aget_state 失败: {e}")
            return []
        intrs = getattr(state, "interrupts", None) or ()
        results: list[dict] = []
        for intr in intrs:
            try:
                results.append({
                    "interrupt_id": getattr(intr, "id", ""),
                    "payload": getattr(intr, "value", {}),
                    "resumable": bool(getattr(intr, "resumable", True)),
                })
            except Exception:
                continue
        return results

    async def chat_stream_resume(
        self,
        session_id: str,
        approve: bool,
    ):
        """恢复被 interrupt 暂停的会话流。

        前提：`chat_stream` 已 yield 过 `confirm_required`。
        `approve=True` → tool 执行；`approve=False` → tool 返回 [CANCELLED]。
        """
        from langgraph.types import Command
        await self._ensure_initialized()

        # resume 时没有原始 query，只能用全量 tools 保证 graph 结构完整
        think_level = self._get_think_level(session_id)
        tools = list(self._builtin_tools)
        try:
            tools.extend(self.skill_retriever.get_all_tools())
        except Exception as e:
            logger.debug(f"resume 时拉取 skills 失败: {e}")
        # 去重
        seen: set[str] = set()
        uniq = []
        for t in tools:
            if t.name not in seen:
                seen.add(t.name)
                uniq.append(t)

        agent = self._build_agent(uniq, think_level, use_vlm=False)
        config = {"configurable": {"thread_id": session_id}}

        ai_chunks: list[str] = []
        sess = self.session_mgr.get_session(session_id)
        trace_on = (sess.metadata.get("trace", "off") == "on") if sess else False

        try:
            async for ev in self._iter_stream_events(
                agent, Command(resume=approve), config=config,
                trace_on=trace_on, session_id=session_id, ai_chunks=ai_chunks,
            ):
                yield ev
        except Exception as e:
            logger.error(f"resume 流式执行失败: {e}", exc_info=True)
            yield {"type": "error", "content": f"{type(e).__name__}: {e}"}
            return

        # resume 后可能再次遇到 interrupt
        paused = await self._check_interrupts(agent, config)
        if paused:
            for pay in paused:
                yield {"type": "confirm_required", **pay}
            return

        if ai_chunks:
            self._schedule_hot_extract(session_id, "", "".join(ai_chunks))

    # 会话清空 / 压缩

    async def clear_session_history(self, session_id: str) -> bool:
        """清空某个 session 在 checkpointer 里的对话历史。"""
        checkpointer = self.memory.get_checkpointer()
        try:
            # 新版 LangGraph 支持 delete_thread
            if hasattr(checkpointer, "adelete_thread"):
                await checkpointer.adelete_thread(session_id)
                return True
            if hasattr(checkpointer, "delete_thread"):
                checkpointer.delete_thread(session_id)
                return True
        except Exception as e:
            logger.warning(f"清空会话历史失败: {e}")
        return False

    async def compact_session(
        self,
        session_id: str,
        keep_recent: int = 6,
    ) -> str:
        """把旧消息浓缩成摘要，只保留最近 `keep_recent` 条原文。"""
        from langchain_core.messages import SystemMessage
        from langchain_core.messages.modifier import RemoveMessage

        graph = self._get_state_graph()
        config = {"configurable": {"thread_id": session_id}}

        snapshot = await graph.aget_state(config)
        if snapshot is None or not snapshot.values:
            return "(会话无历史，无需压缩)"

        messages = snapshot.values.get("messages", []) or []
        if len(messages) <= keep_recent:
            return f"(当前 {len(messages)} 条消息，不足以压缩)"

        to_summarize = messages[:-keep_recent]

        # 拼接历史做摘要
        def _msg_line(m) -> str:
            role = getattr(m, "type", "msg")
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            return f"[{role}] {content}"

        history_text = "\n".join(_msg_line(m) for m in to_summarize)
        prompt = (
            "请把下面的对话历史浓缩为一段简洁的摘要（200 字以内），"
            "保留关键事实、决定和未解决的问题。仅返回摘要文本。\n\n"
            f"{history_text}"
        )
        try:
            summary_msg = await self.llm.ainvoke(prompt)
            summary_text = getattr(summary_msg, "content", str(summary_msg))
            if isinstance(summary_text, list):
                summary_text = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in summary_text
                )
        except Exception as e:
            logger.warning(f"LLM 摘要失败: {e}，放弃 compact")
            return f"(摘要失败: {e})"

        # 先删旧消息，再按 [summary, *kept_recent_copies] 重写，保证摘要在前。
        kept_recent = messages[-keep_recent:] if keep_recent > 0 else []
        all_removes = [
            RemoveMessage(id=m.id) for m in messages if getattr(m, "id", None)
        ]
        summary_message = SystemMessage(
            content=f"[先前对话摘要] {summary_text.strip()}"
        )

        # kept_recent 要复制成无 id 的新消息，避免被 RemoveMessage 一起删掉。
        def _clone_without_id(m):
            # 清掉 id，让 LangGraph 把它当新消息重新追加
            clone = m.model_copy()
            try:
                clone.id = None
            except Exception:
                pass
            return clone

        new_kept = [_clone_without_id(m) for m in kept_recent]
        replacement = all_removes + [summary_message] + new_kept

        # 多节点 graph 下优先指定 as_node，避免 Ambiguous
        try:
            await graph.aupdate_state(
                config,
                {"messages": replacement},
                as_node="__start__",
            )
        except Exception as e:
            logger.debug(f"aupdate_state with as_node=__start__ 失败 ({e})，重试不带 as_node")
            try:
                await graph.aupdate_state(config, {"messages": replacement})
            except Exception as e2:
                logger.warning(f"写回 compacted state 失败: {e2}")
                return f"(写回失败: {e2})"

        logger.info(
            f"Session {session_id} 已压缩：{len(to_summarize)} 条旧消息 → 1 条摘要"
        )

        # compact 成功后顺手抽一批长期记忆；失败不影响 compact 结果。
        if (
            self.memory_store is not None
            and self.settings.memory.extract_on_compact
            and (self.settings.memory.extract_mode or "").lower() != "off"
        ):
            try:
                added = await self.memory_store.extract_and_store(
                    llm=self.llm,
                    messages=to_summarize,
                    session_id=session_id,
                )
                if added:
                    logger.info(f"compact 后抽入 {added} 条长期记忆")
            except Exception as e:
                logger.debug(f"compact 后记忆抽取失败（忽略）: {e}")

        return summary_text.strip()

    async def _maybe_auto_compact(self, session_id: str) -> None:
        """若会话消息数超阈值，自动触发 compact（失败不抛）"""
        threshold = self.settings.session.compact_threshold
        try:
            graph = self._get_state_graph()
            config = {"configurable": {"thread_id": session_id}}
            snapshot = await graph.aget_state(config)
            if snapshot is None or not snapshot.values:
                return
            messages = snapshot.values.get("messages", []) or []
            if len(messages) >= threshold:
                logger.info(
                    f"Session {session_id} 已累积 {len(messages)} 条消息 ≥ "
                    f"阈值 {threshold}，自动 compact"
                )
                await self.compact_session(session_id)
        except Exception as e:
            logger.debug(f"auto-compact 检查失败（忽略）: {e}")

    async def _ensure_initialized(self):
        """确保 Agent 已完成异步初始化（幂等）"""
        if not self._initialized:
            await self.ainitialize()

    async def ainitialize(self):
        """
        异步初始化（必须在 event loop 内调用）

        - 初始化 SQLite Checkpointer
        - 初始化 LlamaIndex Skills 索引
        - 确保默认会话存在
        """
        if self._initialized:
            return

        logger.info("初始化 JimiAgent（异步）...")

        # 1. Memory
        await self.memory.initialize()

        # 2. Skills
        try:
            self.skill_retriever.initialize()
        except Exception as e:
            logger.warning(f"Skills 召回引擎初始化失败: {e}，将使用全量加载模式")

        # 3. 默认会话
        self.session_mgr.ensure_default_session()

        self._initialized = True
        self._log_status()

    # 同步 initialize 仅供 CLI doctor/status 等轻量场景使用。
    def initialize(self):
        """仅初始化同步资源（Skills + 默认会话）；异步依赖请用 ainitialize。"""
        if self._initialized:
            return
        logger.info("初始化 JimiAgent（同步子集）...")
        try:
            self.skill_retriever.initialize()
        except Exception as e:
            logger.warning(f"Skills 召回引擎初始化失败: {e}")
        self.session_mgr.ensure_default_session()
        self._log_status()

    async def aclose(self):
        """关闭资源：停 hot_extract、关 Memory/MemoryStore、清 cache。"""
        # 1) cancel 挂起的 hot_extract tasks，避免 pending task 警告
        tasks = getattr(self, "_hot_tasks", None)
        if tasks:
            import asyncio as _aio
            # 先 snapshot 再 cancel，避免等待时集合被回调修改
            snapshot = [t for t in tasks if not t.done()]
            for t in snapshot:
                t.cancel()
            # 最多等 2s；剩余任务交给 loop 清理
            if snapshot:
                try:
                    await _aio.wait(snapshot, timeout=2.0)
                except Exception as e:
                    logger.debug(f"等待 hot_tasks 完成失败（忽略）: {e}")
            tasks.clear()

        # 2) 关 Memory
        try:
            await self.memory.close()
        except Exception as e:
            logger.warning(f"关闭 Memory 失败: {e}")

        # 3) 关 MemoryStore
        if self.memory_store is not None:
            try:
                self.memory_store.close()
            except Exception as e:
                logger.warning(f"关闭 MemoryStore 失败: {e}")

        # 4) 清 graph 缓存
        self._agent_graph_cache.clear()
        self._state_graph = None
        logger.info("JimiAgent 已关闭")

    def invalidate_agent_cache(self):
        """清空 Agent graph 缓存（Skills / workspace / think level 变化后调用）"""
        self._agent_graph_cache.clear()
        # state_graph 不依赖工具和 prompt，保留即可

    def _log_status(self):
        """输出当前状态"""
        skills = self.workspace.list_skills()
        sessions = self.session_mgr.list_sessions()
        logger.info(f"  模型: {self.settings.model.provider}/{self.settings.model.model_id}")
        logger.info(f"  Skills: {len(skills)} 个 ({', '.join(skills) if skills else '无'})")
        logger.info(f"  会话: {len(sessions)} 个")
        logger.info(
            f"  Memory: {'SQLite' if self.memory.is_persistent else 'Memory(降级)'}"
        )
        logger.info(
            f"  Gateway: {self.settings.gateway.host}:{self.settings.gateway.port}"
        )
