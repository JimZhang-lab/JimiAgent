'''
Author: JimZhang
Date: 2026-04-19 14:45:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 14:45:00
FilePath: /JimiAgent/server/plugin/channels/webhook.py
Description: 通用 Webhook 渠道。

'''
import hashlib
import logging

from server.plugin.channels.base import BaseChannel, OutboundMessage

logger = logging.getLogger(__name__)


def sender_to_thread_id(sender_id: str, prefix: str = "wh") -> str:
    """把 sender_id 映射为稳定 thread_id。"""
    if not sender_id:
        return f"{prefix}-anonymous"
    digest = hashlib.md5(sender_id.encode("utf-8")).hexdigest()[:8]
    return f"{prefix}-{digest}"


class WebhookChannel(BaseChannel):
    """Webhook 渠道占位实现"""

    @property
    def channel_name(self) -> str:
        return "webhook"

    async def start(self):
        logger.info("Webhook 渠道已启用（入站端点: POST /api/webhook/{name}）")

    async def stop(self):
        logger.info("Webhook 渠道已停用")

    async def send_message(self, message: OutboundMessage):
        """主动出站暂未实现。"""
        logger.debug(
            f"Webhook 出站占位: to={message.recipient_id} "
            f"len={len(message.content)}"
        )
