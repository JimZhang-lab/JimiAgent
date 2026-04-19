'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/server/plugin/channels/webchat.py
Description: WebChat 渠道占位实现。

'''
import logging
from server.plugin.channels.base import BaseChannel, OutboundMessage

logger = logging.getLogger(__name__)


class WebChatChannel(BaseChannel):
    """WebChat 渠道占位实现"""

    @property
    def channel_name(self) -> str:
        return "webchat"

    async def start(self):
        logger.info("WebChat 渠道已启动")

    async def stop(self):
        logger.info("WebChat 渠道已停止")

    async def send_message(self, message: OutboundMessage):
        # WebChat 走 WebSocket，不用这里发送
        pass
