'''K5 - Memory 槽位协议。'''
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


@runtime_checkable
class MemoryPort(Protocol):
    """跨会话长期记忆的最小必要接口（见 MemoryStore）。"""

    # --- 生命周期 ---
    def initialize(self) -> None: ...
    def close(self) -> None: ...

    # --- 基础 CRUD ---
    def add(
        self,
        text: str,
        *,
        kind: str = "semantic",
        source: str = "",
        metadata: Optional[dict] = None,
    ) -> int: ...

    def delete(self, memory_id: int) -> bool: ...

    def list_recent(
        self, *, kinds: Optional[list[str]] = None, limit: int = 20,
    ) -> list[Any]: ...

    def search(
        self, query: str, *, kinds: Optional[list[str]] = None, top_k: int = 5,
    ) -> list[Any]: ...

    # --- system prompt 拼装 ---
    def build_recall_block(self, query: str, max_items: int = 5) -> str: ...
    def build_rules_block(self, max_items: int = 10) -> str: ...

    # --- 后台抽取 ---
    def extract_and_store(
        self,
        llm: Any,
        messages: list[Any],
        *,
        source: str = "",
    ) -> list[Any]: ...

    # --- Triple 支持（K5 可选） ---
    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        *,
        valid_from: Optional[str] = None,
    ) -> int: ...

    def triples_at(
        self,
        ts: str,
        *,
        subject: str = "",
        predicate: str = "",
        limit: int = 50,
    ) -> list[Any]: ...
