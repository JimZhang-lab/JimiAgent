'''K5 - Context Engine 槽位协议。'''
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ContextEnginePort(Protocol):
    """上下文引擎最小接口。"""

    def initialize(self) -> None: ...
    def retrieve_skills(self, query: str) -> list[Any]: ...
    def get_tools_for_query(self, query: str) -> list[Any]: ...
