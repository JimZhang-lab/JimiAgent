'''
Author: JimZhang
Date: 2026-04-21 02:00:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-21 02:00:00
FilePath: /JimiAgent/server/core/tui_worker.py
Description: React/Ink TUI 的 Python 端 worker。
             通过 stdin/stdout NDJSON 与 Node TUI 通信。
             不含任何渲染代码；所有 UI 呈现在 Node 端完成。

'''
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from typing import Any, AsyncGenerator, Optional

# 关键：logging 必须走 stderr，否则会污染 stdout 的 NDJSON
logging.basicConfig(
    level=os.environ.get("JIMI_TUI_WORKER_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
    force=True,
)
# 压低第三方库噪音
for _noisy in ("httpx", "httpcore", "openai", "urllib3", "websockets"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

logger = logging.getLogger("jimi.tui_worker")


WORKER_VERSION = "0.1.0"


# ============================================================================
# 斜杠命令清单（原 `server/core/tui.py` 的 SLASH_COMMANDS，迁入本模块作为
# 唯一数据源；旧 rich.live TUI 已下线）
# 顺序即补全列表顺序。
# ============================================================================

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


# ============================================================================
# 输出：单例锁 + 行缓冲 flush
# ============================================================================

_stdout_lock = asyncio.Lock()


async def emit(event: dict) -> None:
    """把一个事件作为一行 JSON 写入 stdout（原子）。"""
    line = json.dumps(event, ensure_ascii=False) + "\n"
    async with _stdout_lock:
        # 直接写二进制避免编码相关的 print 副作用
        sys.stdout.buffer.write(line.encode("utf-8"))
        sys.stdout.buffer.flush()


def emit_sync(event: dict) -> None:
    """同步 emit：用于还没进 asyncio 或异常退出路径。"""
    line = json.dumps(event, ensure_ascii=False) + "\n"
    try:
        sys.stdout.buffer.write(line.encode("utf-8"))
        sys.stdout.buffer.flush()
    except Exception:
        pass


# ============================================================================
# Session/状态序列化
# ============================================================================

def session_to_dict(sess) -> dict:
    if sess is None:
        return None  # type: ignore[return-value]
    return {
        "id": sess.id,
        "title": sess.title,
        "created_at": sess.created_at,
        "updated_at": sess.updated_at,
        "message_count": sess.message_count,
        "metadata": dict(sess.metadata or {}),
    }


def agent_info(agent) -> dict:
    s = agent.settings
    try:
        skills = list(agent.workspace.list_skills())
    except Exception:
        skills = []
    # memory mode
    try:
        mem_mode = "sqlite" if agent.memory.is_persistent else "memory"
    except Exception:
        mem_mode = "disabled"
    return {
        "model": f"{s.model.provider}/{s.model.model_id}",
        "embedding": getattr(s.embedding, "model", ""),
        "skills": skills,
        "gateway": {"host": s.gateway.host, "port": s.gateway.port},
        "memory": mem_mode,
    }


def status_snapshot(agent, session_id: str) -> dict:
    sess = agent.session_mgr.get_session(session_id) if session_id else None
    return {
        "session": session_to_dict(sess) if sess else None,
        "agent": agent_info(agent),
        "session_count": len(agent.session_mgr.list_sessions()),
    }


# ============================================================================
# 请求分派
# ============================================================================

class WorkerShutdown(Exception):
    """shutdown 请求触发的退出信号。"""


class WorkerState:
    """保存 worker 运行时状态（当前流任务等）。"""

    def __init__(self, agent):
        self.agent = agent
        # 当前正在进行的 chat/resume 任务；新请求会先 cancel 旧的
        self.current_stream_task: Optional[asyncio.Task] = None
        self.current_stream_session: Optional[str] = None


async def consume_stream(
    state: WorkerState,
    stream: AsyncGenerator[dict, None],
    session_id: str,
    req_id: Optional[str],
) -> None:
    """把 agent.chat_stream / chat_stream_resume 事件透传到 stdout。"""
    try:
        async for ev in stream:
            # 透传原始事件字段；只保证是 dict 且有 type
            if not isinstance(ev, dict) or "type" not in ev:
                logger.warning("丢弃无效事件 %r", ev)
                continue
            await emit(ev)
    except asyncio.CancelledError:
        await emit({
            "type": "error",
            "content": "（已取消）",
        })
        raise
    except Exception as e:
        logger.exception("stream 异常")
        await emit({
            "type": "error",
            "content": f"{type(e).__name__}: {e}",
        })
    finally:
        try:
            await stream.aclose()
        except Exception:
            pass
        state.current_stream_task = None
        state.current_stream_session = None
        await emit({"type": "done", "session_id": session_id, "req_id": req_id})


async def cancel_current_stream(state: WorkerState) -> None:
    task = state.current_stream_task
    if task is None or task.done():
        return
    task.cancel()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


async def handle_chat(state: WorkerState, req: dict) -> None:
    session_id = req.get("session_id", "")
    message = req.get("message", "")
    images = req.get("images") or None
    req_id = req.get("req_id")
    if not session_id:
        await emit({"type": "error", "content": "chat: 缺 session_id"})
        await emit({"type": "done", "session_id": "", "req_id": req_id})
        return

    # 有旧流则先取消
    await cancel_current_stream(state)

    stream = state.agent.chat_stream(
        message,
        session_id,
        images=images,
    )
    state.current_stream_session = session_id
    state.current_stream_task = asyncio.create_task(
        consume_stream(state, stream, session_id, req_id),
    )


async def handle_resume(state: WorkerState, req: dict) -> None:
    session_id = req.get("session_id", "")
    approve = bool(req.get("approve", False))
    req_id = req.get("req_id")
    if not session_id:
        await emit({"type": "error", "content": "resume: 缺 session_id"})
        await emit({"type": "done", "session_id": "", "req_id": req_id})
        return

    await cancel_current_stream(state)

    stream = state.agent.chat_stream_resume(session_id, approve)
    state.current_stream_session = session_id
    state.current_stream_task = asyncio.create_task(
        consume_stream(state, stream, session_id, req_id),
    )


async def handle_cancel(state: WorkerState, req: dict) -> None:
    await cancel_current_stream(state)
    await emit({
        "type": "done",
        "session_id": req.get("session_id", ""),
        "req_id": req.get("req_id"),
    })


async def handle_list_sessions(state: WorkerState, req: dict) -> None:
    items = [session_to_dict(s) for s in state.agent.session_mgr.list_sessions()]
    await emit({
        "type": "sessions",
        "items": items,
        "req_id": req.get("req_id"),
    })


async def handle_list_commands(state: WorkerState, req: dict) -> None:
    items = [{"name": k, "description": v} for k, v in SLASH_COMMANDS.items()]
    await emit({
        "type": "commands",
        "items": items,
        "req_id": req.get("req_id"),
    })


async def _emit_history_for(
    state: WorkerState,
    session_id: str,
    req_id: Optional[str] = None,
) -> None:
    """读取 checkpointer 中会话的消息列表，转成一条 `history` 事件发送。

    用于会话切换 / 首次进入时把历史消息"回放"到前端，配合 Ink <Static>
    print-above 架构，历史对话会直接落到终端 scrollback 里供原生滚动查看。

    消息角色映射（langchain_core.messages → 前端 MessageRole）：
      - HumanMessage  → user
      - AIMessage     → assistant（content 为空且仅有 tool_calls 的会被跳过）
      - ToolMessage   → tool（content 前置"调用工具: <name>"）
      - SystemMessage → system

    读取失败 / 无历史时仍发送空 items 的 history 事件，让前端清晰知道
    "回放完毕但无内容"，避免加载动画卡住。
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        SystemMessage,
        ToolMessage,
    )

    items: list[dict] = []
    try:
        graph = state.agent._get_state_graph()
        config = {"configurable": {"thread_id": session_id}}
        snapshot = await graph.aget_state(config)
        messages = (snapshot.values or {}).get("messages", []) if snapshot else []
    except Exception as e:
        logger.warning("读取会话 %s 历史失败: %s", session_id, e)
        messages = []

    for msg in messages:
        content = getattr(msg, "content", "") or ""
        if not isinstance(content, str):
            # ContentBlocks（多模态）压成纯文本粗略展示；print-above 架构
            # 不处理 inline 图片，后续可扩展
            try:
                content = json.dumps(content, ensure_ascii=False)
            except Exception:
                content = str(content)

        tool_name: Optional[str] = None
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
            # 空 content 的 AIMessage 一般是纯 tool_call 决策，不对用户有价值，跳过
            if not content.strip():
                continue
        elif isinstance(msg, ToolMessage):
            role = "tool"
            tool_name = getattr(msg, "name", None) or None
            if tool_name:
                content = f"调用工具: {tool_name}\n{content}".rstrip()
        elif isinstance(msg, SystemMessage):
            role = "system"
        else:
            # 未知类型，保守当 system
            role = "system"

        item: dict = {"role": role, "content": content}
        if tool_name:
            item["tool_name"] = tool_name
        items.append(item)

    await emit({
        "type": "history",
        "session_id": session_id,
        "items": items,
        "req_id": req_id,
    })


async def handle_switch_session(state: WorkerState, req: dict) -> None:
    # 发 session ACK + 回放历史消息，让前端能把整段对话写入 scrollback
    session_id = req.get("session_id", "")
    req_id = req.get("req_id")
    sess = state.agent.session_mgr.get_session(session_id)
    if sess is None:
        await emit({
            "type": "error",
            "content": f"会话不存在: {session_id}",
        })
    else:
        await emit({"type": "session", "session_id": session_id})
        # 先发 session 事件让前端先 clearMessages / 切 currentSessionId，
        # 再批量追加 history items，顺序与视觉一致
        await _emit_history_for(state, session_id, req_id)
    await emit({
        "type": "done",
        "session_id": session_id,
        "req_id": req_id,
    })


async def handle_new_session(state: WorkerState, req: dict) -> None:
    title = req.get("title") or "新对话"
    sess = state.agent.session_mgr.create_session(title)
    await emit({"type": "session", "session_id": sess.id})
    # 新会话必定无历史，但仍发一条空 history 事件，保证 UI 侧状态机对齐
    await _emit_history_for(state, sess.id, req.get("req_id"))
    # 同步把最新列表也发一次
    await handle_list_sessions(state, {"req_id": req.get("req_id")})


async def handle_delete_session(state: WorkerState, req: dict) -> None:
    session_id = req.get("session_id", "")
    ok = state.agent.session_mgr.delete_session(session_id)
    if ok:
        try:
            await state.agent.clear_session_history(session_id)
        except Exception as e:
            logger.warning("清理 checkpointer 失败: %s", e)
    await emit({
        "type": "session_deleted",
        "session_id": session_id,
        "ok": ok,
        "req_id": req.get("req_id"),
    })
    await handle_list_sessions(state, {"req_id": req.get("req_id")})


async def handle_get_status(state: WorkerState, req: dict) -> None:
    session_id = req.get("session_id", "") or ""
    await emit({
        "type": "status",
        "data": status_snapshot(state.agent, session_id),
        "req_id": req.get("req_id"),
    })


async def handle_ping(state: WorkerState, req: dict) -> None:
    await emit({"type": "pong", "req_id": req.get("req_id")})


# 记忆操作 ==================================================================

def _memory_to_dict(m) -> dict:
    return {
        "id": int(m.id),
        "kind": m.kind,
        "subject": m.subject,
        "text": m.text,
        "created_at": m.created_at,
        "updated_at": m.updated_at,
        "source_session": m.source_session,
        "hits": int(m.hits),
        "score": float(getattr(m, "score", 0.0) or 0.0),
        "predicate": getattr(m, "predicate", "") or "",
        "object": getattr(m, "object", "") or "",
        "valid_from": getattr(m, "valid_from", "") or "",
        "valid_until": getattr(m, "valid_until", "") or "",
        "namespace": getattr(m, "namespace", "default") or "default",
    }


async def handle_list_memories(state: WorkerState, req: dict) -> None:
    store = getattr(state.agent, "memory_store", None)
    if store is None:
        await emit({
            "type": "memories",
            "items": [],
            "req_id": req.get("req_id"),
        })
        return
    limit = int(req.get("limit") or 100)
    kinds = req.get("kinds") or None
    try:
        mems = store.list_recent(limit=limit, kinds=kinds)
    except Exception as e:
        logger.warning("list_memories 失败: %s", e)
        mems = []
    await emit({
        "type": "memories",
        "items": [_memory_to_dict(m) for m in mems],
        "req_id": req.get("req_id"),
    })


async def handle_search_memories(state: WorkerState, req: dict) -> None:
    store = getattr(state.agent, "memory_store", None)
    if store is None:
        await emit({
            "type": "memories",
            "items": [],
            "query": req.get("query", ""),
            "req_id": req.get("req_id"),
        })
        return
    query = str(req.get("query") or "").strip()
    k = int(req.get("k") or 10)
    kinds = req.get("kinds") or None
    if not query:
        # 空查询直接当 list
        try:
            mems = store.list_recent(limit=k, kinds=kinds)
        except Exception:
            mems = []
    else:
        try:
            mems = store.search(query, k=k, kinds=kinds)
        except Exception as e:
            logger.warning("search_memories 失败: %s", e)
            mems = []
    await emit({
        "type": "memories",
        "items": [_memory_to_dict(m) for m in mems],
        "query": query,
        "req_id": req.get("req_id"),
    })


async def handle_delete_memory(state: WorkerState, req: dict) -> None:
    store = getattr(state.agent, "memory_store", None)
    if store is None:
        await emit({
            "type": "memory_deleted",
            "memory_id": int(req.get("memory_id") or 0),
            "ok": False,
            "req_id": req.get("req_id"),
        })
        return
    mid = int(req.get("memory_id") or 0)
    ok = False
    try:
        ok = bool(store.delete(mid))
    except Exception as e:
        logger.warning("delete_memory(%s) 失败: %s", mid, e)
    await emit({
        "type": "memory_deleted",
        "memory_id": mid,
        "ok": ok,
        "req_id": req.get("req_id"),
    })


DISPATCH = {
    "chat": handle_chat,
    "resume": handle_resume,
    "cancel": handle_cancel,
    "list_sessions": handle_list_sessions,
    "list_commands": handle_list_commands,
    "switch_session": handle_switch_session,
    "new_session": handle_new_session,
    "delete_session": handle_delete_session,
    "get_status": handle_get_status,
    "list_memories": handle_list_memories,
    "search_memories": handle_search_memories,
    "delete_memory": handle_delete_memory,
    "ping": handle_ping,
}


async def dispatch(state: WorkerState, req: dict) -> None:
    kind = req.get("kind", "")
    if kind == "shutdown":
        raise WorkerShutdown()
    handler = DISPATCH.get(kind)
    if handler is None:
        await emit({
            "type": "error",
            "content": f"未知请求: kind={kind!r}",
        })
        return
    try:
        await handler(state, req)
    except Exception as e:
        logger.exception("handler 执行失败: %s", kind)
        await emit({
            "type": "error",
            "content": f"{type(e).__name__}: {e}",
        })


# ============================================================================
# stdin 读取
# ============================================================================

async def stdin_lines() -> AsyncGenerator[str, None]:
    """异步行读取 stdin；EOF 时结束。"""
    loop = asyncio.get_running_loop()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await loop.connect_read_pipe(lambda: protocol, sys.stdin)
    while True:
        line = await reader.readline()
        if not line:
            return
        try:
            yield line.decode("utf-8", errors="replace").rstrip("\n")
        except Exception as e:
            logger.warning("stdin 解码失败: %s", e)


# ============================================================================
# 主入口
# ============================================================================

async def main_async() -> int:
    # 延迟 import 避免启动慢
    from server.core.agent import JimiAgent

    t0 = time.time()
    try:
        agent = JimiAgent()
        await agent.ainitialize()
    except Exception as e:
        logger.exception("JimiAgent 初始化失败")
        emit_sync({"type": "error", "content": f"init 失败: {e}"})
        return 1

    # 确保至少有一个默认会话
    default_sess = agent.session_mgr.ensure_default_session()

    await emit({
        "type": "ready",
        "worker_version": WORKER_VERSION,
        "agent_info": agent_info(agent),
        "default_session_id": default_sess.id,
    })
    logger.info(
        "tui_worker ready in %.2fs (session=%s)",
        time.time() - t0,
        default_sess.id,
    )

    state = WorkerState(agent)
    try:
        async for line in stdin_lines():
            text = line.strip()
            if not text:
                continue
            try:
                req = json.loads(text)
            except Exception as e:
                logger.warning("stdin 非 JSON: %s (line=%r)", e, text[:200])
                await emit({
                    "type": "error",
                    "content": f"请求解析失败: {e}",
                })
                continue
            if not isinstance(req, dict):
                await emit({
                    "type": "error",
                    "content": f"请求必须是对象: {type(req).__name__}",
                })
                continue
            try:
                await dispatch(state, req)
            except WorkerShutdown:
                logger.info("收到 shutdown 请求")
                break
    except asyncio.CancelledError:
        pass
    finally:
        # 优雅收尾：取消当前流，关闭 agent
        await cancel_current_stream(state)
        try:
            await agent.aclose()
        except Exception as e:
            logger.warning("agent.aclose 失败: %s", e)

    return 0


def main() -> int:
    try:
        return asyncio.run(main_async())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
