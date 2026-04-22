'''
Author: JimZhang
Date: 2026-04-19 17:55:00
LastEditors: JimZhang
LastEditTime: 2026-04-19 17:55:00
FilePath: /JimiAgent/server/plugin/channels/pairing.py
Description: Pairing/Allowlist。
'''
import json
import logging
import random
import string
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# 配对码有效期（秒）
PAIR_CODE_TTL = 600  # 10 分钟


class PairingStore:
    """Allowlist 和待配对码管理。"""

    def __init__(
        self,
        path: Path,
        channel_defaults: Optional[dict[str, list[str]]] = None,
    ):
        """初始化 PairingStore"""
        self._path = path
        self._data: dict = {"allowlist": {}, "pending": {}}
        self._load()

        # 合并 yaml allowFrom
        if channel_defaults:
            for channel, senders in channel_defaults.items():
                if not senders:
                    continue
                existing = self._data["allowlist"].setdefault(channel, [])
                for s in senders:
                    if s not in existing:
                        existing.append(s)

    # 持久化

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text("utf-8"))
            except Exception as e:
                logger.warning(f"加载 allowlist 失败，将使用空白: {e}")
                self._data = {"allowlist": {}, "pending": {}}
        self._data.setdefault("allowlist", {})
        self._data.setdefault("pending", {})

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Allowlist 查询

    def is_allowed(self, channel: str, sender_id: str) -> bool:
        """检查 sender 是否在白名单中。"""
        import fnmatch

        allowed = self._data["allowlist"].get(channel, [])
        for pat in allowed:
            if pat == "*" or pat == sender_id:
                return True
            if "*" in pat or "?" in pat:
                if fnmatch.fnmatch(sender_id, pat):
                    return True
        return False

    def add_to_allowlist(self, channel: str, sender_id: str) -> None:
        """将 sender 加入白名单"""
        lst = self._data["allowlist"].setdefault(channel, [])
        if sender_id not in lst:
            lst.append(sender_id)
            self._save()
            logger.info(f"Allowlist: {channel}/{sender_id} 已加入")

    def remove_from_allowlist(self, channel: str, sender_id: str) -> bool:
        """从白名单移除；返回是否实际移除了"""
        lst = self._data["allowlist"].get(channel, [])
        if sender_id in lst:
            lst.remove(sender_id)
            self._save()
            return True
        return False

    def list_allowed(self, channel: Optional[str] = None) -> dict:
        """返回 allowlist（可选按 channel 过滤）"""
        if channel:
            return {channel: self._data["allowlist"].get(channel, [])}
        return dict(self._data["allowlist"])

    # Pairing

    def create_pair_code(self, channel: str, sender_id: str) -> str:
        """为 (channel, sender_id) 生成 6 位配对码。"""
        key = f"{channel}:{sender_id}"
        existing = self._data["pending"].get(key)
        if existing:
            age = time.time() - existing.get("created_at", 0)
            if age < PAIR_CODE_TTL:
                return existing["code"]

        code = "".join(random.choices(string.ascii_uppercase + string.digits, k=6))
        self._data["pending"][key] = {
            "code": code,
            "created_at": time.time(),
        }
        self._save()
        return code

    def approve_code(self, channel: str, code: str) -> Optional[str]:
        """验证配对码并放行。"""
        now = time.time()
        for key, val in list(self._data["pending"].items()):
            if val["code"] == code.upper():
                ch, sid = key.split(":", 1)
                if ch != channel:
                    continue
                if now - val.get("created_at", 0) > PAIR_CODE_TTL:
                    # 过期
                    del self._data["pending"][key]
                    self._save()
                    return None
                # 放行
                del self._data["pending"][key]
                self.add_to_allowlist(channel, sid)
                return sid
        return None

    def list_pending(self) -> list[dict]:
        """列出所有待审批的配对请求"""
        result = []
        now = time.time()
        for key, val in list(self._data["pending"].items()):
            ch, sid = key.split(":", 1)
            age = now - val.get("created_at", 0)
            if age > PAIR_CODE_TTL:
                del self._data["pending"][key]
                continue
            result.append({
                "channel": ch,
                "sender_id": sid,
                "code": val["code"],
                "remaining_seconds": int(PAIR_CODE_TTL - age),
            })
        if len(result) != len(self._data["pending"]):
            self._save()  # 清理了过期项
        return result

    def revoke(self, channel: str, sender_id: str) -> bool:
        """吊销：从 allowlist 移除 + 清理 pending"""
        removed = self.remove_from_allowlist(channel, sender_id)
        key = f"{channel}:{sender_id}"
        if key in self._data["pending"]:
            del self._data["pending"][key]
            self._save()
            removed = True
        return removed
