'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/server/plugin/channels/base.py
Description: 渠道抽象。

'''
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class InboundMessage:
    """入站消息"""
    content: str
    sender_id: str
    channel: str
    session_id: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class OutboundMessage:
    """出站消息"""
    content: str
    recipient_id: str
    channel: str
    session_id: Optional[str] = None
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseChannel(ABC):
    """消息渠道抽象基类"""

    @property
    @abstractmethod
    def channel_name(self) -> str:
        """渠道名称"""
        ...

    @abstractmethod
    async def start(self):
        """启动渠道"""
        ...

    @abstractmethod
    async def stop(self):
        """停止渠道"""
        ...

    @abstractmethod
    async def send_message(self, message: OutboundMessage):
        """发送消息"""
        ...

    async def on_message(self, message: InboundMessage):
        """接收消息回调"""
        if self._message_handler:
            await self._message_handler(message)

    def set_message_handler(self, handler):
        """设置消息处理回调"""
        self._message_handler = handler

    def __init__(self):
        self._message_handler = None
