'''
Author: JimZhang
Date: 2026-04-19 15:05:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 15:05:00
FilePath: /JimiAgent/server/core/scheduler.py
Description: Cron 调度器。

'''
from __future__ import annotations

import asyncio
import datetime
import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from croniter import croniter

if TYPE_CHECKING:
    from server.core.agent import JimiAgent
    from server.config.settings import SchedulerConfig

logger = logging.getLogger(__name__)


def _job_session_id(job_id: str, explicit: Optional[str] = None) -> str:
    """派生 job 专属 session_id。"""
    if explicit:
        return explicit
    # cron- 前缀避免与 wh-/UUID 冲突
    return f"cron-{job_id}"


@dataclass
class Job:
    """运行时 Job 状态"""
    id: str
    schedule: str
    prompt: str
    session_id: str
    session_title: str
    kind: str = "chat"
    strategy: str = "balanced"
    enabled: bool = True
    next_run_at: Optional[datetime.datetime] = None
    last_run_at: Optional[datetime.datetime] = None
    last_status: str = "pending"      # pending / ok / error
    last_error: str = ""
    run_count: int = 0

    @classmethod
    def from_raw(cls, raw: dict) -> "Job":
        job_id = str(raw.get("id") or "")
        if not job_id:
            raise ValueError("cron job 缺少 id 字段")
        schedule = str(raw.get("schedule") or "").strip()
        if not schedule:
            raise ValueError(f"cron job {job_id} 缺少 schedule 字段")
        if not croniter.is_valid(schedule):
            raise ValueError(f"cron job {job_id} 的 schedule 非法: {schedule!r}")

        kind = str(raw.get("kind") or "chat").lower()
        if kind not in ("chat", "evolver"):
            raise ValueError(f"cron job {job_id} 的 kind 非法: {kind!r}")

        # prompt 仅对 kind=chat 必填
        prompt = str(raw.get("prompt") or "").strip()
        if kind == "chat" and not prompt:
            raise ValueError(f"cron job {job_id} (kind=chat) 缺少 prompt 字段")

        session_id = _job_session_id(job_id, raw.get("session_id"))
        title = str(raw.get("session_title") or f"[定时] {job_id}")
        strategy = str(raw.get("strategy") or "balanced")
        return cls(
            id=job_id,
            schedule=schedule,
            prompt=prompt,
            session_id=session_id,
            session_title=title,
            kind=kind,
            strategy=strategy,
            enabled=bool(raw.get("enabled", True)),
        )

    def compute_next(self, base: Optional[datetime.datetime] = None) -> None:
        """按当前时间计算下次执行时刻。"""
        base = base or datetime.datetime.now().astimezone()
        itr = croniter(self.schedule, base)
        self.next_run_at = itr.get_next(datetime.datetime)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "schedule": self.schedule,
            "kind": self.kind,
            "strategy": self.strategy,
            "prompt": self.prompt,
            "session_id": self.session_id,
            "session_title": self.session_title,
            "enabled": self.enabled,
            "next_run_at": (
                self.next_run_at.isoformat() if self.next_run_at else None
            ),
            "last_run_at": (
                self.last_run_at.isoformat() if self.last_run_at else None
            ),
            "last_status": self.last_status,
            "last_error": self.last_error,
            "run_count": self.run_count,
        }


class Scheduler:
    """Cron 调度器。"""

    def __init__(self, agent: "JimiAgent", cfg: "SchedulerConfig"):
        self.agent = agent
        self.cfg = cfg
        self._jobs: dict[str, Job] = {}
        self._task: Optional[asyncio.Task] = None
        self._stop_event = asyncio.Event()
        self._reload_jobs(cfg)

    def _reload_jobs(self, cfg: "SchedulerConfig") -> None:
        """根据配置重新解析并覆盖 jobs，保留合法任务的运行时状态"""
        old = dict(self._jobs)
        new: dict[str, Job] = {}
        for idx, raw in enumerate(cfg.jobs or []):
            if not isinstance(raw, dict):
                logger.warning(f"忽略非法 scheduler.jobs[{idx}]: {raw!r}")
                continue
            try:
                job = Job.from_raw(raw)
                # 保留已有运行状态
                prev = old.get(job.id)
                if prev is not None:
                    job.last_run_at = prev.last_run_at
                    job.last_status = prev.last_status
                    job.last_error = prev.last_error
                    job.run_count = prev.run_count
                job.compute_next()
                new[job.id] = job
            except Exception as e:
                logger.warning(f"忽略 scheduler.jobs[{idx}]: {e}")

        removed = set(old) - set(new)
        for rid in removed:
            logger.info(f"Scheduler 移除 job: {rid}")
        added = set(new) - set(old)
        for aid in added:
            logger.info(f"Scheduler 新增 job: {aid}")
        self._jobs = new

    def reload(self, cfg: "SchedulerConfig") -> None:
        """热重载配置与任务"""
        self.cfg = cfg
        self._reload_jobs(cfg)

    # ===== 查询 / 控制 =====

    def list_jobs(self) -> list[dict]:
        return [j.to_dict() for j in self._jobs.values()]

    def get_job(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    async def trigger_now(self, job_id: str) -> dict:
        """手动立即触发一次。返回 Job.to_dict()。"""
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        await self._run_job(job)
        return job.to_dict()

    # ===== 后台循环 =====

    def start(self) -> None:
        if not self.cfg.enabled:
            logger.info("Scheduler 未启用（scheduler.enabled=false）")
            return
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop(), name="jimi-scheduler")
        logger.info(
            f"Scheduler 已启动: {len(self._jobs)} 个任务, tick={self.cfg.tick_seconds}s"
        )

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop_event.set()
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
        self._task = None
        logger.info("Scheduler 已停止")

    async def _loop(self) -> None:
        tick = max(1, int(self.cfg.tick_seconds or 30))
        while not self._stop_event.is_set():
            try:
                await self._tick_once()
            except Exception as e:
                logger.exception(f"Scheduler tick 异常: {e}")

            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=tick)
            except asyncio.TimeoutError:
                continue

    async def _tick_once(self) -> None:
        now = datetime.datetime.now().astimezone()
        for job in list(self._jobs.values()):
            if not job.enabled:
                continue
            if job.next_run_at is None:
                job.compute_next(now)
                continue
            if now >= job.next_run_at:
                # 先推进下次时间，避免重复触发
                job.compute_next(now)
                await self._run_job(job)

    async def _run_job(self, job: Job) -> None:
        """执行单个定时任务"""
        now = datetime.datetime.now().astimezone()
        job.last_run_at = now
        job.run_count += 1

        if job.kind == "evolver":
            logger.info(f"cron job 触发 (evolver): {job.id} strategy={job.strategy}")
            try:
                events = self.agent.evolver.run_cycle(
                    strategy=job.strategy, apply=False,
                )
                job.last_status = "ok"
                job.last_error = ""
                logger.info(
                    f"cron job {job.id} (evolver) 完成: 生成 {len(events)} 条 event"
                )
            except Exception as e:
                job.last_status = "error"
                job.last_error = f"{type(e).__name__}: {e}"
                logger.exception(f"cron job {job.id} (evolver) 失败")
            return

        # kind == "chat" 默认路径
        # 幂等建会话
        try:
            self.agent.session_mgr.create_session_with_id(
                session_id=job.session_id,
                title=job.session_title,
                metadata={"channel": "cron", "job_id": job.id},
            )
        except Exception as e:
            logger.warning(f"cron job {job.id} 创建会话失败: {e}")

        logger.info(
            f"cron job 触发: {job.id} (session={job.session_id}) -> {job.prompt[:40]}"
        )
        try:
            response, _ = await self.agent.chat(job.prompt, job.session_id)
            job.last_status = "ok"
            job.last_error = ""
            logger.info(
                f"cron job {job.id} 完成: {response[:80].replace(chr(10), ' ')}"
            )
        except Exception as e:
            job.last_status = "error"
            job.last_error = f"{type(e).__name__}: {e}"
            logger.exception(f"cron job {job.id} 失败")
