'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/server/core/session.py
Description: 会话元信息管理器。

'''
import os
import uuid
import time
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """对话会话"""
    id: str
    title: str = "新对话"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    message_count: int = 0
    metadata: dict = field(default_factory=dict)


class SessionManager:
    """本地 JSON 会话元信息管理"""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.sessions_file = data_dir / "sessions.json"
        self._sessions: dict[str, Session] = {}
        self._load()

    def _load(self):
        """加载 JSON 数据

        坐文件时备份为 `.corrupt.<timestamp>.bak`，避免下一次 _save 直接覆盖丢失历史。
        """
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if self.sessions_file.exists():
            try:
                data = json.loads(self.sessions_file.read_text(encoding="utf-8"))
                for sid, info in data.items():
                    self._sessions[sid] = Session(**info)
            except Exception as e:
                logger.error(f"加载会话文件失败: {e}")
                # 坐文件 → 搬开避免下次 _save 直接覆盖
                try:
                    backup = self.sessions_file.with_name(
                        f"{self.sessions_file.name}.corrupt.{int(time.time())}.bak"
                    )
                    self.sessions_file.replace(backup)
                    logger.warning(f"损坏的会话文件已备份至 {backup}")
                except Exception as be:
                    logger.error(f"备份损坏会话文件失败: {be}")

    def _save(self):
        """原子写 JSON：先写 .tmp 再 os.replace（SIGKILL 也不会发出半构文件）"""
        data = {sid: asdict(s) for sid, s in self._sessions.items()}
        tmp = self.sessions_file.with_suffix(self.sessions_file.suffix + ".tmp")
        tmp.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(tmp, self.sessions_file)

    def create_session(self, title: str = "新对话") -> Session:
        """创建新会话（随机 UUID 前 8 位为 ID）"""
        return self.create_session_with_id(
            session_id=str(uuid.uuid4())[:8],
            title=title,
        )

    def create_session_with_id(
        self,
        session_id: str,
        title: str = "新对话",
        metadata: Optional[dict] = None,
    ) -> Session:
        """以显式 ID 创建或更新会话（幂等）"""
        existing = self._sessions.get(session_id)
        if existing is not None:
            if title and existing.title != title:
                existing.title = title
            if metadata:
                existing.metadata.update(metadata)
            existing.updated_at = time.time()
            self._save()
            return existing

        session = Session(
            id=session_id,
            title=title,
            metadata=dict(metadata or {}),
        )
        self._sessions[session.id] = session
        self._save()
        logger.info(f"创建会话: {session.id} - {session.title}")
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取指定会话"""
        return self._sessions.get(session_id)

    def list_sessions(self) -> list[Session]:
        """列出所有会话，按更新时间降序"""
        return sorted(
            self._sessions.values(),
            key=lambda s: s.updated_at,
            reverse=True,
        )

    def update_session(self, session_id: str, **kwargs):
        """更新会话信息"""
        session = self._sessions.get(session_id)
        if session:
            for key, value in kwargs.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            session.updated_at = time.time()
            self._save()

    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save()
            logger.info(f"删除会话: {session_id}")
            return True
        return False

    def increment_message_count(self, session_id: str):
        """增加会话消息计数"""
        session = self._sessions.get(session_id)
        if session:
            session.message_count += 1
            session.updated_at = time.time()
            self._save()

    def touch(self, session_id: str, **metadata_updates) -> Optional[Session]:
        """更新元数据并落盘"""
        session = self._sessions.get(session_id)
        if session is None:
            return None

        # 分离 Session 字段 vs metadata 字段
        session_fields = {"title", "message_count"}
        for key, value in metadata_updates.items():
            if key in session_fields:
                setattr(session, key, value)
            else:
                session.metadata[key] = value

        session.updated_at = time.time()
        self._save()
        return session

    def ensure_default_session(self) -> Session:
        """确保至少有一个默认会话"""
        sessions = self.list_sessions()
        if sessions:
            return sessions[0]
        return self.create_session("默认对话")
