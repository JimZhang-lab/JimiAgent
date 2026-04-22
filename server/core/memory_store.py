'''
Author: JimZhang
Date: 2026-04-19 18:45:00
LastEditors: JimZhang
LastEditTime: 2026-04-19 18:45:00
FilePath: /JimiAgent/server/core/memory_store.py
Description: 长期记忆存储。
'''
from __future__ import annotations

import datetime
import json
import logging
import math
import pickle
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from server.config.settings import Settings

logger = logging.getLogger(__name__)


# 数据结构

@dataclass
class Memory:
    id: int
    kind: str
    subject: str
    text: str
    created_at: str
    updated_at: str
    source_session: str = ""
    hits: int = 0
    score: float = 0.0  # 检索时填充，越大越相关
    # J 组扩展字段
    predicate: str = ""
    object: str = ""
    valid_from: str = ""
    valid_until: str = ""
    namespace: str = "default"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "subject": self.subject,
            "text": self.text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_session": self.source_session,
            "hits": self.hits,
            "score": self.score,
            "predicate": self.predicate,
            "object": self.object,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "namespace": self.namespace,
        }


KINDS = ("semantic", "episodic", "procedural", "triple")


# 工具函数

def _now_iso() -> str:
    """UTC ISO8601（Python 3.12+ 弃用 utcnow，改用 now(tz.utc)）"""
    return (
        datetime.datetime.now(datetime.timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# 过滤易变的 AI 能力 / 权限 / 沙箱描述，避免脏数据进入长期记忆。
_CAPABILITY_CLAIM_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"AI\s*(无法|不能|只能|仅能|只可|不可)(直接)?(访问|执行|读取|写入)"),
    re.compile(r"(无法|不能|只能)直接(访问|执行|读取)"),
    re.compile(r"AI\s*(被)?限制"),
    re.compile(r"权限(被)?限制"),
    re.compile(r"(系统|安全)规则.*限制"),
    re.compile(r"(仅限|只能).*工作区"),
    re.compile(r"工作区(目录)?.*(仅|只|无法|不能)"),
    re.compile(r"ls\s+-la.*(粘贴|输出)"),
    re.compile(r"粘贴(给|到).*AI"),
    re.compile(r"拖入工作区"),
    re.compile(r"用户(应|需|必须)(自行|在终端)"),
    re.compile(r"用户的?工作区(路径)?是"),
    re.compile(r"(confirm|确认流程|沙箱|白名单|黑名单)"),
    re.compile(r"uses_workspace|has_workspace|workspace_path", re.IGNORECASE),
)


def _looks_like_capability_claim(text: str) -> bool:
    """判断文本是否像能力边界或沙箱描述。"""
    if not text:
        return False
    for rx in _CAPABILITY_CLAIM_PATTERNS:
        if rx.search(text):
            return True
    return False


# MemoryStore

class MemoryStore:
    """跨会话长期记忆存储。"""

    def __init__(
        self,
        settings: "Settings",
        namespace_override: Optional[str] = None,
    ):
        self.settings = settings
        self.cfg = settings.memory
        self.db_path = Path(self.cfg.store_path).expanduser()
        if not self.db_path.is_absolute():
            from server.config.settings import PROJECT_ROOT
            self.db_path = (PROJECT_ROOT / self.db_path).resolve()

        # namespace: override > yaml > agent_name
        if namespace_override:
            self.namespace = str(namespace_override)
        else:
            ns_cfg = (self.cfg.namespace or "auto").strip()
            if ns_cfg == "auto" or not ns_cfg:
                self.namespace = str(getattr(settings, "agent_name", "default"))
            else:
                self.namespace = ns_cfg

        self._conn: Optional[sqlite3.Connection] = None
        self._embed_model = None  # LlamaIndex BaseEmbedding 或 None
        self._embed_dim: Optional[int] = None
        # 写操作统一串行，且允许同一连接跨线程复用
        self._lock = threading.RLock()

    # 初始化

    def initialize(self) -> None:
        if self._conn is not None:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False 允许 threadpool / 后台线程复用连接
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._create_schema()
        if self.cfg.embedding_model != "off":
            self._embed_model = self._resolve_embed_model()
        logger.info(
            f"MemoryStore 就绪: {self.db_path} "
            f"(embedding={'on' if self._embed_model else 'off/FTS'})"
        )

    def _create_schema(self) -> None:
        """创建或迁移到最新 schema。"""
        assert self._conn is not None
        c = self._conn.cursor()
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                subject TEXT NOT NULL DEFAULT 'user',
                text TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_session TEXT,
                deleted_at TEXT,
                hits INTEGER NOT NULL DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_memories_kind ON memories(kind);
            CREATE INDEX IF NOT EXISTS idx_memories_deleted ON memories(deleted_at);

            CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
                text, content='memories', content_rowid='id', tokenize='unicode61'
            );

            -- 触发器：同步 FTS
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text)
                    VALUES('delete', old.id, old.text);
            END;
            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memories_fts(memories_fts, rowid, text)
                    VALUES('delete', old.id, old.text);
                INSERT INTO memories_fts(rowid, text) VALUES (new.id, new.text);
            END;
            """
        )
        # J 组迁移：补齐新列
        existing = {
            r["name"] for r in c.execute("PRAGMA table_info(memories)").fetchall()
        }
        migrations = [
            ("predicate", "ALTER TABLE memories ADD COLUMN predicate TEXT"),
            ("object", "ALTER TABLE memories ADD COLUMN object TEXT"),
            ("valid_from", "ALTER TABLE memories ADD COLUMN valid_from TEXT"),
            ("valid_until", "ALTER TABLE memories ADD COLUMN valid_until TEXT"),
            (
                "namespace",
                "ALTER TABLE memories ADD COLUMN namespace TEXT NOT NULL DEFAULT 'default'",
            ),
        ]
        for col, sql in migrations:
            if col not in existing:
                try:
                    c.execute(sql)
                    logger.info(f"MemoryStore 迁移：补齐列 {col}")
                except sqlite3.OperationalError as e:
                    logger.warning(f"迁移列 {col} 失败: {e}")

        # 复合索引
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_ns_kind "
            "ON memories(namespace, kind, deleted_at)"
        )
        c.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_triple "
            "ON memories(namespace, subject, predicate)"
        )
        self._conn.commit()

    def _resolve_embed_model(self):
        """优先用 agent.embedding 构造 OpenAIEmbedding。"""
        try:
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError:
            logger.info("MemoryStore: 未装 llama-index-embeddings-openai，降级为 FTS")
            return None

        ec = self.settings.embedding
        if not ec.api_key:
            logger.info("MemoryStore: agent.embedding.api_key 空，降级为 FTS")
            return None

        try:
            kw = {"api_key": ec.api_key, "model": ec.model}
            if ec.base_url:
                kw["api_base"] = ec.base_url
            try:
                m = OpenAIEmbedding(**kw)
            except ValueError as ve:
                if "is not a valid OpenAIEmbeddingModelType" in str(ve):
                    safe = dict(kw)
                    safe["model"] = "text-embedding-ada-002"
                    m = OpenAIEmbedding(**safe)
                    object.__setattr__(m, "model_name", ec.model)
                    object.__setattr__(m, "_query_engine", ec.model)
                    object.__setattr__(m, "_text_engine", ec.model)
                else:
                    raise
            return m
        except Exception as e:
            logger.warning(f"MemoryStore embedding 初始化失败: {e}，降级为 FTS")
            return None

    def _embed(self, text: str) -> Optional[list[float]]:
        if not self._embed_model:
            return None
        try:
            vec = self._embed_model.get_text_embedding(text)
            if self._embed_dim is None and vec:
                self._embed_dim = len(vec)
            return list(vec)
        except Exception as e:
            logger.debug(f"embed 失败（跳过）: {e}")
            return None

    # CRUD

    def add(
        self,
        text: str,
        kind: str = "semantic",
        subject: str = "user",
        source_session: str = "",
        *,
        embedding: Optional[list[float]] = None,
    ) -> int:
        """写入一条记忆。"""
        assert self._conn is not None
        text = text.strip()
        if not text:
            return 0
        if kind not in KINDS:
            kind = "semantic"

        vec = embedding if embedding is not None else self._embed(text)
        with self._lock:
            dup_id = self._find_duplicate(text, vec, kind, subject)
            if dup_id is not None:
                now = _now_iso()
                self._conn.execute(
                    "UPDATE memories SET updated_at=?, hits=hits+1 WHERE id=?",
                    (now, dup_id),
                )
                self._conn.commit()
                return dup_id

            now = _now_iso()
            blob = pickle.dumps(vec) if vec else None
            cur = self._conn.execute(
                "INSERT INTO memories(kind, subject, text, embedding, "
                "created_at, updated_at, source_session, namespace) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                (kind, subject, text, blob, now, now, source_session, self.namespace),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_triple(
        self,
        subject: str,
        predicate: str,
        obj: str,
        valid_from: Optional[str] = None,
        source_session: str = "",
    ) -> int:
        """写入一条时间三元组 (S, P, O)。

        时间覆盖：若已存在同 namespace + subject + predicate 且 object 不同的有效项，
        把旧项的 valid_until 设为 新项的 valid_from（旧事实过期），再插入新项。
        相同 object 则只 touch 既有条目。
        """
        assert self._conn is not None
        subject = (subject or "user").strip()
        predicate = (predicate or "").strip()
        obj = (obj or "").strip()
        if not (subject and predicate and obj):
            return 0

        now = _now_iso()
        vf = (valid_from or now).strip()

        # select/update/insert 需在同一锁内完成
        with self._lock:
            existing = self._conn.execute(
                "SELECT id, object FROM memories "
                "WHERE namespace=? AND kind='triple' AND subject=? AND predicate=? "
                "  AND deleted_at IS NULL AND (valid_until IS NULL OR valid_until > ?)",
                (self.namespace, subject, predicate, now),
            ).fetchall()

            for row in existing:
                if (row["object"] or "") == obj:
                    # 相同事实只 touch
                    self._conn.execute(
                        "UPDATE memories SET updated_at=?, hits=hits+1 WHERE id=?",
                        (now, int(row["id"])),
                    )
                    self._conn.commit()
                    return int(row["id"])
                # object 变化时让旧事实过期
                self._conn.execute(
                    "UPDATE memories SET valid_until=?, updated_at=? WHERE id=?",
                    (vf, now, int(row["id"])),
                )

            # text 里顺带存 "S P O"，便于 FTS
            text = f"{subject} {predicate} {obj}"
            vec = self._embed(text)
            blob = pickle.dumps(vec) if vec else None
            cur = self._conn.execute(
                "INSERT INTO memories(kind, subject, text, embedding, "
                "created_at, updated_at, source_session, namespace, "
                "predicate, object, valid_from) "
                "VALUES('triple', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    subject, text, blob, now, now, source_session,
                    self.namespace, predicate, obj, vf,
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def triples_at(
        self,
        ts: Optional[str] = None,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        limit: int = 50,
    ) -> list[Memory]:
        """返回指定时间有效的三元组。"""
        assert self._conn is not None
        at = (ts or _now_iso()).strip()
        q = (
            "SELECT * FROM memories "
            "WHERE namespace=? AND kind='triple' AND deleted_at IS NULL "
            "  AND (valid_from IS NULL OR valid_from <= ?) "
            "  AND (valid_until IS NULL OR valid_until > ?)"
        )
        args: list = [self.namespace, at, at]
        if subject:
            q += " AND subject=?"
            args.append(subject)
        if predicate:
            q += " AND predicate=?"
            args.append(predicate)
        q += " ORDER BY updated_at DESC LIMIT ?"
        args.append(int(limit))
        rows = self._conn.execute(q, args).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def _find_duplicate(
        self, text: str, vec: Optional[list[float]], kind: str, subject: str,
    ) -> Optional[int]:
        assert self._conn is not None
        # 先查完全相同文本
        row = self._conn.execute(
            "SELECT id FROM memories "
            "WHERE deleted_at IS NULL AND namespace=? AND kind=? AND subject=? "
            "  AND text=? LIMIT 1",
            (self.namespace, kind, subject, text),
        ).fetchone()
        if row:
            return int(row["id"])

        if not vec:
            return None

        rows = self._conn.execute(
            "SELECT id, embedding FROM memories "
            "WHERE deleted_at IS NULL AND namespace=? AND kind=? AND subject=? "
            "  AND embedding IS NOT NULL",
            (self.namespace, kind, subject),
        ).fetchall()
        thr = float(self.cfg.dedup_threshold)
        for r in rows:
            try:
                other = pickle.loads(r["embedding"])
            except Exception:
                continue
            if _cosine(vec, other) >= thr:
                return int(r["id"])
        return None

    def delete(self, memory_id: int) -> bool:
        assert self._conn is not None
        now = _now_iso()
        with self._lock:
            cur = self._conn.execute(
                "UPDATE memories SET deleted_at=? "
                "WHERE id=? AND namespace=? AND deleted_at IS NULL",
                (now, memory_id, self.namespace),
            )
            self._conn.commit()
            return cur.rowcount > 0

    def get(self, memory_id: int) -> Optional[Memory]:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT * FROM memories "
            "WHERE id=? AND namespace=? AND deleted_at IS NULL",
            (memory_id, self.namespace),
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def list_recent(self, limit: int = 20, kinds: Optional[list[str]] = None) -> list[Memory]:
        assert self._conn is not None
        q = "SELECT * FROM memories WHERE deleted_at IS NULL AND namespace=?"
        args: list = [self.namespace]
        if kinds:
            placeholders = ",".join("?" * len(kinds))
            q += f" AND kind IN ({placeholders})"
            args.extend(kinds)
        q += " ORDER BY updated_at DESC LIMIT ?"
        args.append(int(limit))
        rows = self._conn.execute(q, args).fetchall()
        return [self._row_to_memory(r) for r in rows]

    # 检索

    def search(
        self,
        query: str,
        k: int = 5,
        kinds: Optional[list[str]] = None,
    ) -> list[Memory]:
        """混合检索：向量优先，FTS 兜底。"""
        assert self._conn is not None
        q = (query or "").strip()
        if not q:
            return []

        # 向量路径
        vec = self._embed(q)
        if vec is not None:
            return self._search_vector(vec, k, kinds)
        return self._search_fts(q, k, kinds)

    def _search_vector(
        self, qvec: list[float], k: int, kinds: Optional[list[str]],
    ) -> list[Memory]:
        assert self._conn is not None
        base = (
            "SELECT * FROM memories "
            "WHERE deleted_at IS NULL AND embedding IS NOT NULL AND namespace=?"
        )
        args: list = [self.namespace]
        if kinds:
            base += f" AND kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        rows = self._conn.execute(base, args).fetchall()
        scored: list[tuple[float, sqlite3.Row]] = []
        for r in rows:
            try:
                v = pickle.loads(r["embedding"])
            except Exception:
                continue
            scored.append((_cosine(qvec, v), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, r in scored[:k]:
            m = self._row_to_memory(r)
            m.score = score
            out.append(m)
        self._touch_hits([m.id for m in out])
        return out

    def _search_fts(
        self, q: str, k: int, kinds: Optional[list[str]],
    ) -> list[Memory]:
        """先走 FTS5，再用 LIKE 兜底。"""
        assert self._conn is not None
        tokens = [t for t in re.split(r"\s+", q) if t]
        if not tokens:
            return []
        escaped = " OR ".join(f'"{t}"' for t in tokens)
        base = (
            "SELECT m.*, bm25(memories_fts) AS rank FROM memories_fts "
            "JOIN memories m ON m.id = memories_fts.rowid "
            "WHERE memories_fts MATCH ? AND m.deleted_at IS NULL AND m.namespace=?"
        )
        args: list = [escaped, self.namespace]
        if kinds:
            base += f" AND m.kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        base += " ORDER BY rank LIMIT ?"
        args.append(int(k))
        try:
            rows = self._conn.execute(base, args).fetchall()
        except sqlite3.OperationalError as e:
            logger.debug(f"FTS 查询失败: {e}，尝试 LIKE 兜底")
            rows = []
        out = []
        for r in rows:
            m = self._row_to_memory(r)
            try:
                m.score = 1.0 / (1.0 + float(r["rank"]))
            except Exception:
                pass
            out.append(m)

        if not out:
            out = self._search_like(q, k, kinds)

        self._touch_hits([m.id for m in out])
        return out

    def _search_like(
        self, q: str, k: int, kinds: Optional[list[str]],
    ) -> list[Memory]:
        """CJK / 子串场景的 LIKE 兜底。"""
        assert self._conn is not None
        tokens = [t for t in re.split(r"\s+", q) if t]
        if not tokens:
            return []
        base = "SELECT * FROM memories WHERE deleted_at IS NULL AND namespace=?"
        args: list = [self.namespace]
        for t in tokens:
            base += " AND text LIKE ?"
            args.append(f"%{t}%")
        if kinds:
            base += f" AND kind IN ({','.join('?' * len(kinds))})"
            args.extend(kinds)
        base += " ORDER BY updated_at DESC LIMIT ?"
        args.append(int(k) * 2)
        try:
            rows = self._conn.execute(base, args).fetchall()
        except sqlite3.OperationalError:
            return []
        scored: list[tuple[float, Memory]] = []
        for r in rows:
            m = self._row_to_memory(r)
            hits = sum(1 for t in tokens if t in m.text)
            m.score = hits / max(1, len(tokens))
            scored.append((m.score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored[:k]]

    def _touch_hits(self, ids: list[int]) -> None:
        if not ids:
            return
        assert self._conn is not None
        with self._lock:
            self._conn.executemany(
                "UPDATE memories SET hits=hits+1 WHERE id=?",
                [(i,) for i in ids],
            )
            self._conn.commit()

    def _row_to_memory(self, row) -> Memory:
        def _g(key: str, default=""):
            try:
                v = row[key]
                return default if v is None else v
            except (IndexError, KeyError):
                return default

        return Memory(
            id=int(row["id"]),
            kind=str(row["kind"]),
            subject=str(row["subject"] or "user"),
            text=str(row["text"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            source_session=str(row["source_session"] or ""),
            hits=int(row["hits"] or 0),
            predicate=str(_g("predicate", "")),
            object=str(_g("object", "")),
            valid_from=str(_g("valid_from", "")),
            valid_until=str(_g("valid_until", "")),
            namespace=str(_g("namespace", "default") or "default"),
        )

    # 抽取

    async def extract_and_store(
        self,
        llm,
        messages: list,
        session_id: str = "",
    ) -> int:
        """用 LLM 从历史消息里抽结构化事实并入库。返回新增/更新计数。"""
        if not messages:
            return 0

        def _to_line(m) -> str:
            role = getattr(m, "type", "msg")
            content = getattr(m, "content", "")
            if isinstance(content, list):
                content = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            text = str(content)[:300]
            return f"[{role}] {text}"

        history = "\n".join(_to_line(m) for m in messages[-40:])  # 最多 40 条
        prompt = (
            "你在帮一个对话 Agent 做长期记忆抽取。\n"
            "从下面的对话里提取对未来对话有参考价值的事实/偏好/规则，"
            "以 JSON 数组返回：\n"
            "```\n"
            "[\n"
            "  {\"kind\": \"semantic|episodic|procedural\", \"text\": \"一句话事实\"},\n"
            "  {\"kind\": \"triple\", \"subject\": \"user\", "
            "\"predicate\": \"works_at\", \"object\": \"ACME\"}\n"
            "]\n"
            "```\n"
            "- semantic：客观事实（用户是谁、项目叫什么、XX 的定义等）\n"
            "- episodic：具体事件（X 日做了 Y；修好了 Z bug）\n"
            "- procedural：偏好/规则（写代码前先给 plan；用简体中文回复）\n"
            "- triple（可选）：关系型事实，可表达随时间变化的属性，"
            "subject/predicate/object 都用简短英文 snake_case\n"
            "- 每条 text ≤ 80 字；总数不超过 8 条；没有价值就返回 []\n"
            "- 只返回纯 JSON，不要 markdown fence。\n"
            "\n"
            "**严格禁止**抽以下内容（这些是易变的工具/权限状态，不是稳定事实）：\n"
            "- AI 自身的能力边界、权限、可/不可访问的目录\n"
            "- 工具使用规则（'AI 只能 X'、'AI 无法 Y'、'需要用户手动 Z'）\n"
            "- 安全策略、confirm 流程、沙箱边界相关描述\n"
            "- 'ls -la xxx 并粘贴输出' 这类临时指令\n"
            "只关心**用户是谁、在做什么、偏好什么、发生了什么**。\n\n"
            f"对话历史：\n{history}"
        )

        try:
            resp = await llm.ainvoke(prompt)
            text = getattr(resp, "content", str(resp))
            if isinstance(text, list):
                text = "".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in text
                )
        except Exception as e:
            logger.warning(f"事实抽取 LLM 调用失败: {e}")
            return 0

        items = _parse_json_array(text)
        if not items:
            return 0

        count = 0
        skipped = 0
        for item in items:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind", "semantic")).strip().lower()
            if kind not in KINDS:
                kind = "semantic"

            try:
                if kind == "triple":
                    s_ = str(item.get("subject", "user")).strip()
                    p_ = str(item.get("predicate", "")).strip()
                    o_ = str(item.get("object", "")).strip()
                    if not (s_ and p_ and o_):
                        continue
                    # post-filter：triple 形式的 AI 能力描述也要拦
                    if _looks_like_capability_claim(f"{s_} {p_} {o_}"):
                        skipped += 1
                        continue
                    self.add_triple(
                        subject=s_, predicate=p_, obj=o_,
                        source_session=session_id,
                    )
                    count += 1
                    continue

                t = str(item.get("text", "")).strip()
                if not t:
                    continue
                # post-filter：LLM 偶尔仍会冒出"AI 无法..."类幻觉事实 → 双重拦截
                if _looks_like_capability_claim(t):
                    skipped += 1
                    continue
                self.add(text=t, kind=kind, source_session=session_id)
                count += 1
            except Exception as e:
                logger.debug(f"记忆入库失败（跳过）: {e}")

        # 后台 hot_extract 在 TUI 下会打断用户正输入；改 debug 避免撕屏。
        # 要看统计用 `/memories` 命令或调高 log level 到 DEBUG。
        if skipped:
            logger.debug(
                f"MemoryStore: 抽取时过滤掉 {skipped} 条「AI 能力边界」幻觉事实"
            )
        if count:
            logger.debug(f"MemoryStore: 从 session={session_id} 抽入 {count} 条记忆")
        return count

    # 注入辅助

    def build_recall_block(self, query: str, k: int, kinds: Optional[list[str]] = None) -> str:
        """构建注入到 system prompt 的记忆块。"""
        if kinds is None:
            kinds = ["semantic", "episodic", "triple"]
        mems = self.search(query, k=k, kinds=kinds)
        if not mems:
            return ""
        lines = ["<memories>"]
        for m in mems:
            day = m.updated_at[:10]
            if m.kind == "triple":
                expired = ""
                if m.valid_until:
                    expired = f" (expired {m.valid_until[:10]})"
                lines.append(
                    f"- [triple] {m.subject} -{m.predicate}-> {m.object}"
                    f"{expired} ({day})"
                )
            else:
                lines.append(f"- [{m.kind}] {m.text} ({day})")
        lines.append("</memories>")
        return "\n".join(lines)

    def build_rules_block(self, limit: int = 10) -> str:
        """构建 procedural 规则块（<rules>），供 system prompt 顶部注入。"""
        mems = self.list_recent(limit=limit, kinds=["procedural"])
        if not mems:
            return ""
        lines = ["<rules>"]
        for m in mems:
            lines.append(f"- {m.text}")
        lines.append("</rules>")
        return "\n".join(lines)

    # 生命周期

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception as e:
                logger.warning(f"MemoryStore SQLite close 失败: {e}")
            self._conn = None


# 辅助

_JSON_ARRAY_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,?\s*)*\]", re.DOTALL)


def _parse_json_array(text: str) -> list:
    """容错解析 LLM 返回的 JSON 数组。"""
    text = text.strip()
    # 去掉可能的 markdown fence
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        v = json.loads(text)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    # 兜底：正则找第一个数组
    m = _JSON_ARRAY_RE.search(text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return []
    return []
