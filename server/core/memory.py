'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/server/core/memory.py
Description: SQLite 记忆管理器。

'''
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """基于 LangGraph SQLite Checkpointer 的短时记忆管理器"""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._checkpointer = None
        self._conn = None
        self._fallback_memory = False
        self._ensure_dir()

    def _ensure_dir(self):
        """确保数据库目录存在"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """初始化持久化 Checkpointer 或回退至内存模式"""
        if self._checkpointer is not None:
            return

        try:
            import aiosqlite
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

            self._conn = await aiosqlite.connect(str(self.db_path))
            saver = AsyncSqliteSaver(self._conn)
            await saver.setup()
            self._checkpointer = saver
            logger.info(f"SQLite Checkpointer 已就绪: {self.db_path}")

        except ImportError:
            logger.warning(
                "langgraph-checkpoint-sqlite 或 aiosqlite 未安装，"
                "降级为内存 Checkpointer（对话不会持久化）"
            )
            self._use_memory_fallback()
        except Exception as e:
            logger.error(f"初始化 SQLite Checkpointer 失败: {e}，降级为内存模式")
            self._use_memory_fallback()

    def _use_memory_fallback(self):
        """降级到内存 Checkpointer"""
        from langgraph.checkpoint.memory import MemorySaver
        self._checkpointer = MemorySaver()
        self._fallback_memory = True

    def get_checkpointer(self):
        """获取已初始化的 Checkpointer 实例"""
        if self._checkpointer is None:
            logger.warning("Checkpointer 未初始化，使用内存降级模式")
            self._use_memory_fallback()
        return self._checkpointer

    @property
    def is_persistent(self) -> bool:
        """是否正在使用持久化存储"""
        return self._checkpointer is not None and not self._fallback_memory

    async def close(self):
        """关闭底层数据库连接"""
        if self._conn is not None:
            try:
                await self._conn.close()
                logger.info("SQLite Checkpointer 连接已关闭")
            except Exception as e:
                logger.warning(f"关闭 SQLite 连接出错: {e}")
            finally:
                self._conn = None
                self._checkpointer = None
