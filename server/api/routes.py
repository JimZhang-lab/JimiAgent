'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/server/api/routes.py
Description: REST 与 WebSocket 路由。

'''
import asyncio
import hmac
import json
import logging
from typing import Optional

from fastapi import (
    APIRouter,
    File,
    Form,
    Header,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import StreamingResponse

from server.api.models import (
    ChatRequest,
    ChatResponse,
    SessionInfo,
    SessionCreateRequest,
    SkillInfo,
    StatusResponse,
    WebhookRequest,
    WebhookResponse,
)
from server.core.agent import JimiAgent
from server.core.skill_loader import parse_skill_md
from server.plugin.channels.webhook import sender_to_thread_id

logger = logging.getLogger(__name__)

router = APIRouter()

# 由 gateway lifespan 注入
_agent: Optional[JimiAgent] = None
_scheduler = None  # 避免循环引用，运行时才引入类型
_registry = None  # AgentRegistry（多 agent 路由）


def set_agent(agent: JimiAgent):
    """注入 Agent 实例"""
    global _agent
    _agent = agent


def get_agent() -> JimiAgent:
    """获取 Agent 实例"""
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent 未初始化")
    return _agent


def set_registry(registry) -> None:
    """注入 AgentRegistry 实例"""
    global _registry
    _registry = registry


def get_registry():
    return _registry


def set_scheduler(scheduler) -> None:
    """注入 Scheduler 实例"""
    global _scheduler
    _scheduler = scheduler


def get_scheduler():
    """获取 Scheduler 实例"""
    if _scheduler is None:
        raise HTTPException(status_code=503, detail="Scheduler 未初始化")
    return _scheduler


# REST 端点

@router.get("/api/status", response_model=StatusResponse)
async def get_status():
    """系统状态"""
    agent = get_agent()
    return StatusResponse(
        status="running",
        model=f"{agent.settings.model.provider}/{agent.settings.model.model_id}",
        skills_count=len(agent.workspace.list_skills()),
        sessions_count=len(agent.session_mgr.list_sessions()),
        gateway_port=agent.settings.gateway.port,
    )


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """发送消息"""
    agent = get_agent()

    # 获取或创建会话
    session_id = request.session_id
    if not session_id:
        session = agent.session_mgr.ensure_default_session()
        session_id = session.id

    # 调用 Agent，避免裸 500
    try:
        response, effective_sid = await agent.chat(
            request.message, session_id, images=request.images,
        )
    except Exception as e:
        logger.error(f"Agent chat 失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Agent 处理失败: {type(e).__name__}: {e}",
        )

    return ChatResponse(response=response, session_id=effective_sid)


@router.get("/api/sessions", response_model=list[SessionInfo])
async def list_sessions():
    """会话列表"""
    agent = get_agent()
    sessions = agent.session_mgr.list_sessions()
    return [
        SessionInfo(
            id=s.id,
            title=s.title,
            created_at=s.created_at,
            updated_at=s.updated_at,
            message_count=s.message_count,
        )
        for s in sessions
    ]


@router.post("/api/sessions/new", response_model=SessionInfo)
async def create_session(request: SessionCreateRequest):
    """创建新会话"""
    agent = get_agent()
    session = agent.session_mgr.create_session(request.title)
    return SessionInfo(
        id=session.id,
        title=session.title,
        created_at=session.created_at,
        updated_at=session.updated_at,
        message_count=session.message_count,
    )


@router.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除会话"""
    agent = get_agent()
    if agent.session_mgr.delete_session(session_id):
        return {"status": "deleted", "session_id": session_id}
    raise HTTPException(status_code=404, detail="会话不存在")


@router.get("/api/skills", response_model=list[SkillInfo])
async def list_skills():
    """技能列表"""
    agent = get_agent()
    skills_dir = agent.workspace.get_skills_dir()
    skill_infos = []
    for name in agent.workspace.list_skills():
        skill_path = skills_dir / name
        meta = parse_skill_md(skill_path)
        if meta:
            skill_infos.append(SkillInfo(
                name=meta.name,
                description=meta.description,
                version=meta.version,
                has_script=meta.script_path is not None,
            ))
    return skill_infos


@router.post("/api/skills/rebuild")
async def rebuild_skills():
    """强制重建 Skills 语义索引（workspace/skills 发生变更后调用）"""
    agent = get_agent()
    try:
        agent.skill_retriever.rebuild_index()
        agent.invalidate_agent_cache()
        agent.workspace.reload()
        return {
            "status": "rebuilt",
            "skills_count": len(agent.workspace.list_skills()),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重建失败: {e}")


# Webhook 渠道

@router.post("/api/webhook/{channel_name}", response_model=WebhookResponse)
async def webhook_inbound(
    channel_name: str,
    request: WebhookRequest,
    x_jimi_webhook_secret: Optional[str] = Header(None),
):
    """通用 Webhook 入站端点"""
    # 路由到目标 agent
    registry = get_registry()
    if registry:
        agent = registry.resolve(
            channel="webhook", channel_name=channel_name,
            sender_id=request.sender_id,
        )
    else:
        agent = get_agent()
    webhook_cfg = agent.settings.channels.get("webhook") or {}

    if not webhook_cfg.get("enabled", False):
        raise HTTPException(status_code=404, detail="Webhook 渠道未启用")

    # 恒定时间密钥校验
    expected_secret = webhook_cfg.get("secret", "") or ""
    if expected_secret:
        got = x_jimi_webhook_secret or ""
        if not hmac.compare_digest(got, expected_secret):
            raise HTTPException(status_code=401, detail="Webhook secret 不匹配")

    # Pairing / Allowlist 校验
    dm_policy = webhook_cfg.get("dmPolicy", "open")
    if dm_policy != "open" and not agent.pairing.is_allowed(channel_name, request.sender_id):
        if dm_policy == "closed":
            raise HTTPException(status_code=403, detail="sender 未在白名单中")
        # pairing 模式返回配对码
        code = agent.pairing.create_pair_code(channel_name, request.sender_id)
        return WebhookResponse(
            sender_id=request.sender_id,
            session_id="",
            response=(
                f"请让管理员执行: python cli.py pairing approve {channel_name} {code}\n"
                f"配对码 {code} 有效期 10 分钟。"
            ),
        )

    # 计算 session_id
    session_id = request.session_id
    if not session_id:
        prefix = webhook_cfg.get("session_prefix", "wh") or "wh"
        session_id = sender_to_thread_id(request.sender_id, prefix=prefix)
        # 幂等写入元信息
        agent.session_mgr.create_session_with_id(
            session_id=session_id,
            title=f"[{channel_name}] {request.sender_id}",
            metadata={
                "channel": channel_name,
                "sender_id": request.sender_id,
                **(request.metadata or {}),
            },
        )

    logger.info(
        f"Webhook inbound: channel={channel_name} sender={request.sender_id} "
        f"-> session={session_id}"
    )

    try:
        response_text, effective_sid = await agent.chat(
            request.message, session_id
        )
    except Exception as e:
        logger.error(f"Webhook Agent 处理失败: {e}", exc_info=True)
        raise HTTPException(
            status_code=502,
            detail=f"Agent 处理失败: {type(e).__name__}: {e}",
        )

    return WebhookResponse(
        sender_id=request.sender_id,
        session_id=effective_sid,
        response=response_text,
    )


# Scheduler

@router.get("/api/scheduler/jobs")
async def scheduler_list_jobs():
    """列出所有定时任务及下次执行时间"""
    return {"jobs": get_scheduler().list_jobs()}


@router.post("/api/scheduler/jobs/{job_id}/run")
async def scheduler_run_job(job_id: str):
    """手动触发一次。"""
    sched = get_scheduler()
    try:
        return await sched.trigger_now(job_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"未找到 job: {job_id}")


# Evolver

@router.post("/api/evolver/run")
async def evolver_run(body: dict = None):
    """触发一轮进化，生成事件"""
    from dataclasses import asdict
    body = body or {}
    strategy = body.get("strategy", "balanced")
    agent = get_agent()
    try:
        events = agent.evolver.run_cycle(strategy=strategy, apply=False)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"events": [asdict(e) for e in events]}


@router.post("/api/evolver/apply")
async def evolver_apply(body: dict):
    """应用特征进化事件"""
    from dataclasses import asdict
    event_ts = (body or {}).get("event_ts")
    gene_id = (body or {}).get("gene_id")
    if not event_ts and not gene_id:
        raise HTTPException(status_code=400, detail="需提供 event_ts 或 gene_id")

    agent = get_agent()
    events = agent.evolver.load_recent_events(limit=100)
    target = None
    for ev in events:
        if event_ts and ev.ts == event_ts:
            target = ev
            break
        if gene_id and ev.gene_id == gene_id and not ev.applied:
            target = ev
            break
    if target is None:
        raise HTTPException(status_code=404, detail="未找到匹配 event")

    result = await agent.evolver.apply_event(target, agent)
    return result


@router.get("/api/evolver/status")
async def evolver_status():
    """返回 PersonalityState 和最近事件。"""
    from dataclasses import asdict
    agent = get_agent()
    ps = agent.evolver.load_personality()
    events = agent.evolver.load_recent_events(limit=20)
    return {
        "personality": asdict(ps),
        "events": [asdict(e) for e in events],
    }


@router.get("/api/evolver/genes")
async def evolver_list_genes():
    """列出当前 gene 库。"""
    from dataclasses import asdict
    agent = get_agent()
    genes = agent.evolver.load_genes()
    return {"genes": [asdict(g) for g in genes]}


@router.get("/api/evolver/signals")
async def evolver_list_signals():
    """列出当前信号。"""
    from dataclasses import asdict
    agent = get_agent()
    sigs = agent.evolver.scan_signals()
    return {"signals": [asdict(s) for s in sigs]}


# Memory

@router.get("/api/memories")
async def memories_search(q: str = "", k: int = 5, kind: str = ""):
    """搜索长期记忆。"""
    agent = get_agent()
    store = getattr(agent, "memory_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="MemoryStore 未启用")
    kinds = [kind] if kind else None
    if q:
        mems = store.search(q, k=k, kinds=kinds)
    else:
        mems = store.list_recent(limit=k, kinds=kinds)
    return {"memories": [m.to_dict() for m in mems]}


@router.post("/api/memories")
async def memories_add(body: dict):
    """手动写入一条长期记忆"""
    text = str((body or {}).get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="text 不能为空")
    kind = str((body or {}).get("kind") or "semantic")
    agent = get_agent()
    store = getattr(agent, "memory_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="MemoryStore 未启用")
    mid = store.add(text=text, kind=kind)
    return {"id": mid}


@router.delete("/api/memories/{memory_id}")
async def memories_delete(memory_id: int):
    """软删除一条长期记忆。"""
    agent = get_agent()
    store = getattr(agent, "memory_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="MemoryStore 未启用")
    ok = store.delete(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail="未找到或已删除")
    return {"ok": True, "id": memory_id}


@router.get("/api/memories/triples")
async def memories_triples(
    at: str = "", subject: str = "", predicate: str = "", limit: int = 50,
):
    """基于时间戳查询知识图谱三元组"""
    agent = get_agent()
    store = getattr(agent, "memory_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="MemoryStore 未启用")
    ts = at.strip() or None
    if ts and len(ts) == 10 and ts.count("-") == 2:
        ts = ts + "T00:00:00Z"
    mems = store.triples_at(
        ts=ts,
        subject=subject.strip() or None,
        predicate=predicate.strip() or None,
        limit=int(limit),
    )
    return {"at": ts, "triples": [m.to_dict() for m in mems]}


@router.post("/api/memories/triples")
async def memories_triple_add(body: dict):
    """追加三元组。"""
    agent = get_agent()
    store = getattr(agent, "memory_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="MemoryStore 未启用")
    body = body or {}
    s = str(body.get("subject", "user"))
    p = str(body.get("predicate", ""))
    o = str(body.get("object", ""))
    if not (s and p and o):
        raise HTTPException(
            status_code=400, detail="subject / predicate / object 均为必填",
        )
    vf = body.get("valid_from")
    mid = store.add_triple(
        subject=s, predicate=p, obj=o, valid_from=vf,
    )
    return {"id": mid}


# Uploads

_ALLOWED_IMAGE_MIME = {
    "image/jpeg", "image/png", "image/gif", "image/webp",
    "image/bmp", "image/tiff", "image/svg+xml",
}
_ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".tif", ".tiff", ".svg"}
_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@router.post("/api/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(..., description="图片文件"),
    describe: bool = Form(False, description="上传后是否调 VLM 生成描述并写入记忆"),
):
    """上传文件并返回链接。"""
    import datetime as _dt
    import hashlib as _hashlib
    import mimetypes
    from pathlib import Path as _Path
    from server.config.settings import PROJECT_ROOT

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="空文件")
    if len(data) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"文件超过上限 {_MAX_UPLOAD_BYTES // 1024 // 1024} MB",
        )

    filename = (file.filename or "upload").strip()
    ext = _Path(filename).suffix.lower()
    mime = (file.content_type or "").lower() or (
        mimetypes.guess_type(filename)[0] or ""
    )
    if ext not in _ALLOWED_IMAGE_EXT and mime not in _ALLOWED_IMAGE_MIME:
        raise HTTPException(
            status_code=415,
            detail=f"不支持的文件类型: ext={ext} mime={mime}",
        )

    digest = _hashlib.sha256(data).hexdigest()[:16]
    today = _dt.date.today().strftime("%Y%m%d")
    uploads_root = (PROJECT_ROOT / "data" / "uploads" / today)
    uploads_root.mkdir(parents=True, exist_ok=True)
    safe_name = _Path(filename).name.replace("..", "_")
    out_path = uploads_root / f"{digest}-{safe_name}"
    if not out_path.exists():
        out_path.write_bytes(data)

    rel = f"/uploads/{today}/{out_path.name}"
    base = str(request.base_url).rstrip("/")
    public_url = f"{base}{rel}"

    result = {
        "ok": True,
        "url": public_url,
        "rel_url": rel,
        "path": str(out_path),
        "size": len(data),
        "mime": mime or "application/octet-stream",
    }

    if describe:
        try:
            from server.core.multimodal import describe_and_remember
            agent = get_agent()
            extra = await describe_and_remember(agent, public_url)
            result.update(extra)
        except Exception as e:
            logger.warning(f"describe 失败（忽略）: {e}")
            result["describe_error"] = f"{type(e).__name__}: {e}"

    return result


# Plugins

@router.get("/api/plugins")
async def plugins_list_route(enabled: bool = False):
    """列出已发现插件。"""
    agent = get_agent()
    reg = getattr(agent, "plugin_registry", None)
    if reg is None:
        return {"plugins": []}
    items = reg.enabled_plugins() if enabled else reg.list()
    return {"plugins": [s.to_dict() for s in items]}


@router.get("/api/plugins/{plugin_id}")
async def plugins_get_route(plugin_id: str):
    agent = get_agent()
    reg = getattr(agent, "plugin_registry", None)
    state = reg.get(plugin_id) if reg else None
    if state is None:
        raise HTTPException(status_code=404, detail=f"未找到插件 {plugin_id}")
    return state.to_dict()


@router.post("/api/plugins/{plugin_id}/enable")
async def plugins_enable_route(plugin_id: str):
    agent = get_agent()
    reg = getattr(agent, "plugin_registry", None)
    if reg is None or not reg.enable(plugin_id):
        raise HTTPException(status_code=404, detail=f"未找到插件 {plugin_id}")
    return {"ok": True, "id": plugin_id, "enabled": True}


@router.post("/api/plugins/{plugin_id}/disable")
async def plugins_disable_route(plugin_id: str):
    agent = get_agent()
    reg = getattr(agent, "plugin_registry", None)
    if reg is None or not reg.disable(plugin_id):
        raise HTTPException(status_code=404, detail=f"未找到插件 {plugin_id}")
    return {"ok": True, "id": plugin_id, "enabled": False}


@router.post("/api/plugins/rescan")
async def plugins_rescan_route():
    agent = get_agent()
    reg = getattr(agent, "plugin_registry", None)
    if reg is None:
        raise HTTPException(status_code=503, detail="插件系统未启用")
    reg.scan()
    return {"ok": True, "count": len(reg.list())}


# Server-Sent Events

def _sse_format(event_type: str, data: dict) -> str:
    """序列化 SSE 帧。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n"


@router.post("/api/events")
async def chat_events(request: ChatRequest, http_request: Request):
    """SSE 流式聊天端点"""
    agent = get_agent()

    session_id = request.session_id
    if not session_id:
        session = agent.session_mgr.ensure_default_session()
        session_id = session.id

    async def _generator():
        # 首帧先发 session
        yield _sse_format("session", {"session_id": session_id})

        stream = agent.chat_stream(
            request.message, session_id, images=request.images,
        )
        try:
            async for event in stream:
                # 断连后停止推送
                if await http_request.is_disconnected():
                    logger.info("SSE 客户端已断开，提前终止流")
                    break

                etype = event.get("type")
                if etype == "text":
                    yield _sse_format("stream", {"content": event.get("content", "")})
                elif etype == "tool":
                    yield _sse_format("tool", {"name": event.get("name", "unknown")})
                elif etype == "session":
                    # /new 会切换会话
                    yield _sse_format(
                        "session", {"session_id": event.get("session_id")}
                    )
                elif etype == "error":
                    yield _sse_format("error", {"content": event.get("content", "")})

            yield _sse_format("end", {})
        except asyncio.CancelledError:
            logger.info("SSE 流被取消（客户端断开或服务关闭）")
            raise
        except Exception as e:
            logger.error(f"SSE 流异常: {e}", exc_info=True)
            yield _sse_format(
                "error", {"content": f"{type(e).__name__}: {e}"}
            )
        finally:
            # 主动关闭生成器
            try:
                await stream.aclose()
            except Exception:
                pass

    return StreamingResponse(
        _generator(),
        media_type="text/event-stream",
        headers={
            # 禁用代理缓冲
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# WebSocket 端点

@router.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket 聊天端点（流式响应）"""
    await websocket.accept()
    agent = get_agent()

    logger.info("WebSocket 客户端已连接")

    try:
        while True:
            # 接收消息
            data = await websocket.receive_text()
            images: list[str] | None = None
            try:
                payload = json.loads(data)
                message = payload.get("message", "")
                session_id = payload.get("session_id", "")
                imgs = payload.get("images")
                if isinstance(imgs, list) and imgs:
                    images = [str(x) for x in imgs if x]
            except json.JSONDecodeError:
                message = data
                session_id = ""

            if not message:
                await websocket.send_json({"type": "error", "content": "消息为空"})
                continue

            # 获取或创建会话
            if not session_id:
                session = agent.session_mgr.ensure_default_session()
                session_id = session.id

            # 先发会话 ID
            await websocket.send_json({
                "type": "session",
                "session_id": session_id,
            })

            # 转发结构化流事件
            try:
                async for event in agent.chat_stream(
                    message, session_id, images=images,
                ):
                    etype = event.get("type")
                    if etype == "text":
                        await websocket.send_json({
                            "type": "stream",
                            "content": event.get("content", ""),
                        })
                    elif etype == "tool":
                        await websocket.send_json({
                            "type": "tool",
                            "name": event.get("name", "unknown"),
                        })
                    elif etype == "session":
                        # /new 后同步会话
                        session_id = event["session_id"]
                        await websocket.send_json(event)
                    elif etype == "error":
                        await websocket.send_json({
                            "type": "error",
                            "content": event.get("content", ""),
                        })

                # 流结束
                await websocket.send_json({"type": "end"})

            except Exception as e:
                logger.error(f"Agent 处理失败: {e}")
                await websocket.send_json({
                    "type": "error",
                    "content": f"处理失败: {str(e)}",
                })

    except WebSocketDisconnect:
        logger.info("WebSocket 客户端断开连接")
    except Exception as e:
        logger.error(f"WebSocket 异常: {e}")
