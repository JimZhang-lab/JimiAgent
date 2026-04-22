'''
Author: JimZhang
Date: 2026-04-19 18:15:00
LastEditors: JimZhang
LastEditTime: 2026-04-19 18:15:00
FilePath: /JimiAgent/server/core/agent_registry.py
Description: Agent 注册表。
'''
import logging
import re
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from server.core.agent import JimiAgent
    from server.config.settings import Settings

logger = logging.getLogger(__name__)


class RoutingRule:
    """一条路由规则：匹配条件 -> agent 名称"""

    def __init__(self, raw: dict):
        self.agent_name: str = raw.get("agent", "")
        match = raw.get("match") or {}
        self.channel: Optional[str] = match.get("channel")
        self.channel_name: Optional[str] = match.get("name")
        self.sender_prefix: Optional[str] = match.get("sender_id_prefix")
        self.sender_pattern: Optional[re.Pattern] = None
        pat = match.get("sender_id_regex")
        if pat:
            try:
                self.sender_pattern = re.compile(pat)
            except re.error:
                logger.warning(f"routing rule sender_id_regex 非法: {pat!r}")

    def matches(
        self,
        channel: Optional[str] = None,
        channel_name: Optional[str] = None,
        sender_id: Optional[str] = None,
    ) -> bool:
        """规则是否匹配当前请求上下文"""
        if self.channel and channel != self.channel:
            return False
        if self.channel_name and channel_name != self.channel_name:
            return False
        if self.sender_prefix and sender_id and not sender_id.startswith(self.sender_prefix):
            return False
        if self.sender_pattern and sender_id and not self.sender_pattern.search(sender_id):
            return False
        return True


class AgentRegistry:
    """多 Agent 注册表与路由器"""

    def __init__(self):
        self._agents: dict[str, "JimiAgent"] = {}
        self._rules: list[RoutingRule] = []
        self._default_name: str = "main"

    @property
    def default_agent(self) -> "JimiAgent":
        return self._agents[self._default_name]

    def get(self, name: str) -> Optional["JimiAgent"]:
        return self._agents.get(name)

    def all_agents(self) -> dict[str, "JimiAgent"]:
        return dict(self._agents)

    def register(self, name: str, agent: "JimiAgent") -> None:
        self._agents[name] = agent
        logger.info(f"AgentRegistry: 注册 agent '{name}'")

    def set_default(self, name: str) -> None:
        if name not in self._agents:
            raise KeyError(f"agent '{name}' 未注册")
        self._default_name = name

    def set_rules(self, rules: list[dict]) -> None:
        self._rules = [RoutingRule(r) for r in rules]
        logger.info(f"AgentRegistry: 加载 {len(self._rules)} 条路由规则")

    def resolve(
        self,
        channel: Optional[str] = None,
        channel_name: Optional[str] = None,
        sender_id: Optional[str] = None,
    ) -> "JimiAgent":
        """根据上下文匹配路由规则，返回对应 agent（无匹配则返回 default）"""
        for rule in self._rules:
            if rule.matches(channel, channel_name, sender_id):
                agent = self._agents.get(rule.agent_name)
                if agent:
                    return agent
                logger.warning(
                    f"路由规则指向不存在的 agent '{rule.agent_name}'，回退到 default"
                )
        return self.default_agent

    async def ainitialize_all(self) -> None:
        """异步初始化所有已注册的 agent"""
        for name, agent in self._agents.items():
            logger.info(f"初始化 agent '{name}'...")
            await agent.ainitialize()

    async def aclose_all(self) -> None:
        """关闭所有 agent"""
        for name, agent in self._agents.items():
            logger.info(f"关闭 agent '{name}'...")
            await agent.aclose()
