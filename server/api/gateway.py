'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/server/api/gateway.py
Description: FastAPI 入口。

'''
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from server.config.settings import get_settings, PROJECT_ROOT
from server.config.logging_setup import setup_from_settings
from server.core.agent import JimiAgent
from server.core.agent_registry import AgentRegistry
from server.core.scheduler import Scheduler
from server.api.routes import router, set_agent, set_scheduler, set_registry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan 管理器"""
    settings = get_settings()
    setup_from_settings(settings)

    logger.info("=" * 50)
    logger.info("  JimiAgent Gateway 启动中...")
    logger.info("=" * 50)

    # 先给 finally 兜底值
    scheduler = None
    registry = AgentRegistry()
    agents_cfg = settings.agents

    if agents_cfg.agent_list:
        # 多 Agent
        for acfg in agents_cfg.agent_list:
            name = acfg.get("name", "main")
            # 为子 agent 复制配置
            from copy import deepcopy
            from server.config.settings import ModelConfig
            child_settings = deepcopy(settings)
            child_settings.workspace_path = acfg.get("workspace", settings.workspace_path)
            child_settings.agent_name = acfg.get("name", settings.agent_name)
            if "model" in acfg:
                child_settings.model = ModelConfig(**acfg["model"])
            child_agent = JimiAgent(child_settings)
            await child_agent.ainitialize()
            registry.register(name, child_agent)
        registry.set_default(agents_cfg.default_agent)
        if agents_cfg.routing:
            registry.set_rules(agents_cfg.routing)
    else:
        # 单 Agent
        agent_single = JimiAgent(settings)
        await agent_single.ainitialize()
        registry.register("main", agent_single)

    agent = registry.default_agent
    set_agent(agent)
    set_registry(registry)

    # 调度器
    scheduler = Scheduler(agent, settings.scheduler)
    scheduler.start()
    set_scheduler(scheduler)
    agent.scheduler = scheduler  # 供斜杠命令访问

    logger.info(f"  Web UI: http://localhost:{settings.gateway.port}")
    logger.info(f"  API:    http://localhost:{settings.gateway.port}/api/status")
    logger.info(f"  WS:     ws://localhost:{settings.gateway.port}/ws/chat")
    logger.info("=" * 50)

    try:
        yield
    finally:
        logger.info("JimiAgent Gateway 关闭中...")
        if scheduler is not None:
            try:
                await scheduler.stop()
            except Exception as e:
                logger.warning(f"关闭 Scheduler 时出错: {e}")
        # 关闭 Computer Use browser
        try:
            from server.core.computer_use.tools import shutdown as cu_shutdown
            for _name, _agent in registry.all_agents().items():
                await cu_shutdown(_agent)
        except Exception as e:
            logger.debug(f"Computer Use shutdown: {e}")
        try:
            for _name, _agent in registry.all_agents().items():
                await _agent.aclose()
        except Exception as e:
            logger.warning(f"关闭 Agent 时出错: {e}")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    settings = get_settings()

    app = FastAPI(
        title="JimiAgent Gateway",
        description="基于 LangChain/LangGraph 的个人 AI 助手 Gateway",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.gateway.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册路由
    app.include_router(router)

    # Web UI 静态文件
    static_dir = PROJECT_ROOT / "server" / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            """提供 Web UI 入口"""
            return FileResponse(str(static_dir / "index.html"))

    # 公开 /uploads
    uploads_dir = PROJECT_ROOT / "data" / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/uploads", StaticFiles(directory=str(uploads_dir)), name="uploads")

    return app


# 全局 app
app = create_app()
