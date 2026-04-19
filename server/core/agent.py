'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/server/core/agent.py
Description: Agent 核心引擎。

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
            # 支持 8 位前缀匹配
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
            # 当前线程已有 loop，改用新 loop
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
    """
    JimiAgent 核心引擎

    整合所有组件:
    - Workspace: system prompt 组装
    - SkillRetriever: LlamaIndex 语义召回 Skills → LangChain Tools
    - MemoryManager: SQLite 持久化记忆
    - SessionManager: 多会话管理
    - LangGraph ReAct Agent: 推理+行动循环

    生命周期：
        agent = JimiAgent(settings)
        await agent.ainitialize()   # 异步初始化（Memory + Skills）
        ... 调用 chat / chat_stream ...
        await agent.aclose()        # 关闭资源
    """

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or get_settings()

        # 初始化组件
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

        # 长期记忆，支持插件槽位替换
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
        """Workspace 基础 prompt + think level 追加段"""
        base = self.workspace.build_system_prompt()
        return f"{base}\n\n---\n\n{think_prompt_for(think_level)}"

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

        # 有图时切换到 multi-part content
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

    # J3 per-turn 抽取

    def _get_extract_lock(self, session_id: str):
        """懒加载每个 session 的抽取锁。"""
        import asyncio as _aio
        locks = getattr(self, "_extract_locks", None)
        if locks is None:
            locks = {}
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
            # 已有任务在跑时直接跳过
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
        """
        构建（或复用）LangGraph ReAct Agent

        缓存 key = (sorted tool names, think_level, "vlm" or "llm")
        相同 key 复用已编译的 graph。use_vlm=True 时走 self.vlm，否则 self.llm。
        """
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

        # 去重（按 tool name）
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
        """
        处理用户消息并返回完整响应。

        Args:
            message: 用户消息
            session_id: 当前会话 ID
            images: 可选图片 URL / data:URI 列表；非空时走 VLM

        Returns:
            (response_text, effective_session_id)
            — effective_session_id 可能因 /new 命令发生变化
        """
        await self._ensure_initialized()

        # 1. 前置：斜杠命令拦截
        cmd_res = await self.commands.try_handle(message, session_id)
        if cmd_res.handled:
            return cmd_res.response, (cmd_res.new_session_id or session_id)

        # 2. activation 策略检查
        if not self._check_activation(message, session_id):
            return "", session_id

        # 3. 自动 compact（阈值检查，失败不抛）
        await self._maybe_auto_compact(session_id)

        # 4. 常规 Agent 调用
        think_level = self._get_think_level(session_id)
        tools = self._get_tools_for_query(message)
        use_vlm = bool(images)
        agent = self._build_agent(tools, think_level, use_vlm=use_vlm)

        config = {"configurable": {"thread_id": session_id}}
        input_messages = self._build_input_messages(message, images=images)

        result = await agent.ainvoke(input_messages, config=config)

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
        """
        流式处理用户消息

        Args:
            message: 用户消息
            session_id: 会话 ID

        Yields:
            事件字典:
                {"type": "session", "session_id": "xxx"}  — 若命令切换了会话
                {"type": "text", "content": "..."}        — 文本片段
                {"type": "tool", "name": "..."}           — 工具调用开始
                {"type": "error", "content": "..."}       — 错误
        """
        await self._ensure_initialized()

        # 1. 前置：斜杠命令拦截
        cmd_res = await self.commands.try_handle(message, session_id)
        if cmd_res.handled:
            if cmd_res.new_session_id:
                yield {"type": "session", "session_id": cmd_res.new_session_id}
            yield {"type": "text", "content": cmd_res.response}
            return

        # 2. activation 检查
        if not self._check_activation(message, session_id):
            return

        # 3. 自动 compact（阈值检查）
        await self._maybe_auto_compact(session_id)

        # 4. 常规 Agent 流式调用
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
            async for event in agent.astream_events(
                input_messages, config=config, version="v2"
            ):
                kind = event.get("event", "")

                # trace 透传轻量 meta
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
                            has_content = True
                            ai_chunks.append(content)
                            yield {"type": "text", "content": content}

                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    self._log_tool_call(tool_name, "start", session_id=session_id)
                    yield {"type": "tool", "name": tool_name}

                elif kind == "on_tool_end":
                    # 成功收尾
                    self._log_tool_call(
                        event.get("name", "unknown"), "ok",
                        session_id=session_id,
                    )

                elif kind == "on_tool_error":
                    tool_name = event.get("name", "unknown")
                    err = event.get("data", {}).get("error")
                    self._log_tool_call(
                        tool_name, "error",
                        error=str(err) if err else "",
                        session_id=session_id,
                    )

        except Exception as e:
            logger.error(f"Agent 流式执行失败: {e}", exc_info=True)
            yield {"type": "error", "content": f"{type(e).__name__}: {e}"}
            return

        if not has_content:
            yield {"type": "text", "content": "抱歉，我无法生成回复。"}

        self.session_mgr.increment_message_count(session_id)

        # 流结束后后台抽取
        if ai_chunks:
            self._schedule_hot_extract(session_id, message, "".join(ai_chunks))

    # ===== 会话清空 / 压缩 =====

    async def clear_session_history(self, session_id: str) -> bool:
        """
        清空某 session 的 Agent 对话历史（Checkpointer 中的 thread）

        通过覆写当前 thread 的最新 checkpoint 为空消息列表实现。
        返回是否操作成功。
        """
        checkpointer = self.memory.get_checkpointer()
        try:
            # LangGraph 的 checkpointer 支持 delete_thread (>= 0.2.x)
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
        """
        压缩当前会话的上下文

        工作流程：
        1. 读取 thread 的消息列表
        2. 保留最新 `keep_recent` 条，旧消息通过 LLM 浓缩为摘要
        3. 用 RemoveMessage 删除旧消息 + 注入 SystemMessage 形式的摘要
        4. 用 aupdate_state 把新 state 写回 checkpointer

        Args:
            session_id: 会话 ID
            keep_recent: 保留末尾几条消息

        Returns:
            摘要文本（若 messages 过短则返回占位说明）
        """
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

        # 拼接历史，做摘要
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

        # 构造 state 更新：
        # 1) 删除全部现有消息（包括 kept_recent），避免 summary 被追加到尾部
        # 2) 按 [summary, *kept_recent_copies] 顺序重新写入
        # 这样 system 摘要在前，近期消息在后，与人类阅读顺序一致
        kept_recent = messages[-keep_recent:] if keep_recent > 0 else []
        all_removes = [
            RemoveMessage(id=m.id) for m in messages if getattr(m, "id", None)
        ]
        summary_message = SystemMessage(
            content=f"[先前对话摘要] {summary_text.strip()}"
        )

        # kept_recent 消息需要"复制"为没有 id 的新实例，否则会被 RemoveMessage 同步移除
        # （注意：LangGraph 的 add_messages 按 id 去重/替换）
        def _clone_without_id(m):
            # 使用 model_copy 生成新实例，清除 id 让其被当作"新消息"重新追加
            clone = m.model_copy()
            try:
                clone.id = None
            except Exception:
                pass
            return clone

        new_kept = [_clone_without_id(m) for m in kept_recent]
        replacement = all_removes + [summary_message] + new_kept

        # aupdate_state 在多节点 graph 中需指定 as_node；固定用入口节点避免 Ambiguous
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

        # 背景路径：compact 成功后顺手抽取一次长期记忆（失败不影响 compact 结果）
        # extract_mode="off" 时完全跳过；"on_compact" 和 "per_turn" 都会在 compact 时抽一批
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

    # 保留同步 initialize 作为别名（仅初始化非 async 部分，供 CLI doctor/status 使用）
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
        """关闭资源：cancel 后台 hot_extract → 关 Memory → 关 MemoryStore → 清 cache"""
        # 1) cancel 挂起的 hot_extract tasks（避免"Task was destroyed but pending"）
        tasks = getattr(self, "_hot_tasks", None)
        if tasks:
            import asyncio as _aio
            # 先 snapshot 再 cancel —— add_done_callback(discard) 会在 await 期间
            # 修改 tasks set，不能直接对它 wait
            snapshot = [t for t in tasks if not t.done()]
            for t in snapshot:
                t.cancel()
            # 最多等 2s；未完成的由 loop 自行清理
            if snapshot:
                try:
                    await _aio.wait(snapshot, timeout=2.0)
                except Exception as e:
                    logger.debug(f"等待 hot_tasks 完成失败（忽略）: {e}")
            tasks.clear()

        # 2) 关 Memory（LangGraph checkpointer）
        try:
            await self.memory.close()
        except Exception as e:
            logger.warning(f"关闭 Memory 失败: {e}")

        # 3) 关 MemoryStore（长期记忆 SQLite）
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
        # state_graph 不依赖工具/prompt，保留

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
