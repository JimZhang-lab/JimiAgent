'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/server/api/models.py
Description: API 模型定义。

'''
from pydantic import BaseModel, Field
from typing import Optional


class ChatRequest(BaseModel):
    """聊天请求"""
    message: str = Field(..., description="用户消息")
    session_id: Optional[str] = Field(None, description="会话 ID，为空则使用默认会话")
    images: Optional[list[str]] = Field(
        None,
        description="可选图片列表",
    )


class ChatResponse(BaseModel):
    """聊天响应"""
    response: str = Field(..., description="Agent 回复")
    session_id: str = Field(..., description="会话 ID")


class SessionInfo(BaseModel):
    """会话信息"""
    id: str
    title: str
    created_at: float
    updated_at: float
    message_count: int


class SessionCreateRequest(BaseModel):
    """创建会话请求"""
    title: str = Field("新对话", description="会话标题")


class SkillInfo(BaseModel):
    """技能信息"""
    name: str
    description: str
    version: str
    has_script: bool


class StatusResponse(BaseModel):
    """系统状态响应"""
    status: str = "running"
    model: str = ""
    skills_count: int = 0
    sessions_count: int = 0
    gateway_port: int = 18789


class WebhookRequest(BaseModel):
    """Webhook 渠道入站消息"""
    sender_id: str = Field(..., description="外部系统的用户/群组标识，映射到 thread_id")
    message: str = Field(..., description="用户输入")
    session_id: Optional[str] = Field(None, description="显式指定会话 ID")
    metadata: Optional[dict] = Field(
        None, description="可选附加元信息（例如外部消息 ID、时间戳）"
    )


class WebhookResponse(BaseModel):
    """Webhook 渠道同步响应"""
    sender_id: str
    session_id: str
    response: str
