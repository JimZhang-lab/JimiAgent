'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-20 00:00:00
FilePath: /JimiAgent/server/core/skill_retriever.py
Description: Skill 召回引擎。
'''
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from langchain_core.tools import StructuredTool

from server.config.settings import get_settings
from server.core.skill_loader import (
    SkillMeta,
    parse_skill_md,
    build_tool_from_skill,
)

logger = logging.getLogger(__name__)


# Hybrid 检索工具函数

_EN_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _cjk_tokenizer(text: str) -> list[str]:
    """英文按词、中文按字的 BM25 分词器。"""
    if not text:
        return []
    text = text.lower()
    tokens = _EN_TOKEN_RE.findall(text)
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            tokens.append(ch)
    return tokens


class _TTLCache:
    """简易 TTL + LRU 缓存。"""
    __slots__ = ("maxsize", "ttl", "_d", "_order")

    def __init__(self, maxsize: int = 128, ttl_seconds: int = 30):
        self.maxsize = max(1, maxsize)
        self.ttl = max(0, ttl_seconds)
        self._d: dict[Any, tuple[Any, float]] = {}
        self._order: list = []

    def get(self, k):
        entry = self._d.get(k)
        if entry is None:
            return None
        if time.time() - entry[1] >= self.ttl:
            self._d.pop(k, None)
            try:
                self._order.remove(k)
            except ValueError:
                pass
            return None
        return entry[0]

    def put(self, k, v):
        if k in self._d:
            try:
                self._order.remove(k)
            except ValueError:
                pass
        self._d[k] = (v, time.time())
        self._order.append(k)
        while len(self._order) > self.maxsize:
            oldest = self._order.pop(0)
            self._d.pop(oldest, None)

    def clear(self):
        self._d.clear()
        self._order.clear()

    def __len__(self):
        return len(self._d)


class SkillRetriever:
    """基于 LlamaIndex 的 Skill 召回器。"""

    def __init__(
        self,
        skills_dir: Path,
        persist_dir: Optional[Path] = None,
        top_k: int = 3,
        similarity_threshold: float = 0.5,
        extra_skill_dirs: Optional[list[Path]] = None,
    ):
        self.skills_dir = skills_dir
        settings = get_settings()
        self.persist_dir = persist_dir or settings.skill_index_abs_path
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.extra_skill_dirs: list[Path] = list(extra_skill_dirs or [])

        # Hybrid 检索配置
        self._cfg = settings.skills

        # Skill 元信息缓存
        self._skill_metas: dict[str, SkillMeta] = {}

        # LlamaIndex 组件
        self._index = None
        self._vector_retriever = None
        self._bm25_retriever = None
        self._hybrid_retriever = None
        self._active_retriever = None        # 实际对外 retrieve 用的 retriever
        self._active_mode: str = "none"      # hybrid / vector_only / bm25_only / none
        self._initialized = False

        # Query 缓存
        if self._cfg.query_cache_ttl_seconds > 0:
            self._query_cache = _TTLCache(
                maxsize=self._cfg.query_cache_size,
                ttl_seconds=self._cfg.query_cache_ttl_seconds,
            )
        else:
            self._query_cache = None

    # skill 目录扫描

    def _iter_skill_dirs(self):
        """迭代所有 skill 目录。"""
        seen: set[Path] = set()

        if self.skills_dir.exists():
            for d in sorted(self.skills_dir.iterdir()):
                if d.is_dir() and d not in seen:
                    seen.add(d)
                    yield d

        for p in self.extra_skill_dirs:
            if not p.exists() or not p.is_dir():
                continue
            if (p / "SKILL.md").exists() or (p / "skill.md").exists():
                if p not in seen:
                    seen.add(p)
                    yield p
            else:
                for d in sorted(p.iterdir()):
                    if d.is_dir() and d not in seen:
                        if (d / "SKILL.md").exists() or (d / "skill.md").exists():
                            seen.add(d)
                            yield d

    def _compute_skills_hash(self) -> str:
        """计算技能内容哈希。"""
        h = hashlib.md5()
        for skill_dir in self._iter_skill_dirs():
            for name in ("SKILL.md", "skill.md"):
                f = skill_dir / name
                if f.exists():
                    h.update(str(skill_dir).encode("utf-8"))
                    h.update(b"\0")
                    h.update(f.read_bytes())
                    break
        return h.hexdigest()

    def _needs_rebuild(self) -> bool:
        """检查索引是否需要重建"""
        hash_file = self.persist_dir / "skills_hash.json"
        current_hash = self._compute_skills_hash()

        if not hash_file.exists():
            return True

        try:
            saved = json.loads(hash_file.read_text())
            return saved.get("hash") != current_hash
        except Exception:
            return True

    def _save_hash(self):
        """保存当前 Skills 哈希"""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        hash_file = self.persist_dir / "skills_hash.json"
        current_hash = self._compute_skills_hash()
        hash_file.write_text(json.dumps({"hash": current_hash}))

    def _load_metas_only(self):
        """仅扫描 Skills 元信息。"""
        for skill_dir in self._iter_skill_dirs():
            meta = parse_skill_md(skill_dir)
            if meta is not None:
                self._skill_metas[meta.name] = meta

    # 初始化

    def initialize(self):
        """初始化检索链，并按需降级。"""
        if self._initialized:
            return

        try:
            from llama_index.core import (
                VectorStoreIndex,
                Document,
                StorageContext,
                load_index_from_storage,
                Settings as LlamaSettings,
            )
            from llama_index.core.schema import TextNode
            from llama_index.embeddings.openai import OpenAIEmbedding
        except ImportError:
            logger.warning(
                "LlamaIndex 未安装，Skills 召回退化为全量加载模式。"
                "请运行: pip install llama-index-core llama-index-embeddings-openai"
            )
            self._load_metas_only()
            self._initialized = True
            return

        settings = get_settings()
        embed_cfg = settings.embedding
        mode_cfg = (self._cfg.retrieval_mode or "hybrid").lower()

        # 1. 扫 skill 元信息
        for skill_dir in self._iter_skill_dirs():
            meta = parse_skill_md(skill_dir)
            if meta is not None:
                self._skill_metas[meta.name] = meta

        if not self._skill_metas:
            logger.info("未找到任何 Skills，跳过索引构建。")
            self._initialized = True
            return

        metas = list(self._skill_metas.values())

        # 2. 构造 Vector 侧
        need_vector = mode_cfg in ("hybrid", "vector_only")
        vector_ok = False
        if need_vector:
            if not embed_cfg.api_key:
                logger.warning(
                    "未配置 agent.embedding.api_key，Skills Vector 通路禁用。"
                )
            else:
                try:
                    embed_model = self._build_embed_model(
                        OpenAIEmbedding, embed_cfg
                    )
                    LlamaSettings.embed_model = embed_model
                    self._build_or_load_vector_index(
                        VectorStoreIndex, StorageContext, load_index_from_storage, metas
                    )
                    if self._index is not None:
                        self._vector_retriever = self._index.as_retriever(
                            similarity_top_k=max(self.top_k * 2, self.top_k + 1),
                        )
                        vector_ok = True
                except Exception as e:
                    logger.warning(f"Vector 通路初始化失败: {e}")

        # 3. 构造 BM25 侧
        # CJK 走预分词 + token_pattern=\S+
        need_bm25 = mode_cfg in ("hybrid", "bm25_only")
        bm25_ok = False
        if need_bm25:
            try:
                from llama_index.retrievers.bm25 import BM25Retriever
                from llama_index.core.schema import QueryBundle

                class _CJKBM25Retriever(BM25Retriever):
                    """对 query 端做 CJK 预分词的 BM25Retriever 子类"""
                    def _retrieve(self, query_bundle):
                        tokenized = " ".join(_cjk_tokenizer(query_bundle.query_str))
                        new_bundle = QueryBundle(
                            query_str=tokenized,
                            custom_embedding_strs=list(
                                query_bundle.custom_embedding_strs or []
                            ),
                            embedding=query_bundle.embedding,
                        )
                        return super()._retrieve(new_bundle)

                weighted_nodes = self._build_weighted_bm25_nodes(TextNode, metas)
                self._bm25_retriever = _CJKBM25Retriever.from_defaults(
                    nodes=weighted_nodes,
                    similarity_top_k=max(self.top_k * 2, self.top_k + 1),
                    token_pattern=r"(?u)\S+",   # 每个空格分隔 token 独立
                    skip_stemming=True,         # 跳过英语 stemmer，避免把中文字符误处理
                    language="en",              # stopwords 用英语集合；对中文无影响
                    verbose=False,
                )
                bm25_ok = True
            except ImportError:
                logger.warning(
                    "llama-index-retrievers-bm25 未安装，BM25 通路禁用。"
                    "请运行: pip install llama-index-retrievers-bm25"
                )
            except Exception as e:
                logger.warning(f"BM25 通路初始化失败: {e}")

        # 4. 组装 active retriever
        self._assemble_active_retriever(mode_cfg, vector_ok, bm25_ok)

        self._initialized = True
        logger.info(
            f"Skills 召回引擎就绪 · mode={self._active_mode} · "
            f"skills={len(metas)} · cache={'on' if self._query_cache else 'off'}"
        )

    def _build_embed_model(self, OpenAIEmbedding, embed_cfg):
        """构造 OpenAIEmbedding；非 OpenAI 官方 enum 名走占位兼容模式"""
        kwargs = {"api_key": embed_cfg.api_key, "model": embed_cfg.model}
        if embed_cfg.base_url:
            kwargs["api_base"] = embed_cfg.base_url
        try:
            return OpenAIEmbedding(**kwargs)
        except ValueError as ve:
            if "is not a valid OpenAIEmbeddingModelType" in str(ve):
                logger.debug(
                    f"Embedding '{embed_cfg.model}' 不在 OpenAI 官方 enum 中，"
                    "走 OpenAI 兼容模式（占位名绕过 enum + 实际请求用真实名）"
                )
                safe = dict(kwargs)
                safe["model"] = "text-embedding-ada-002"
                m = OpenAIEmbedding(**safe)
                real = embed_cfg.model
                object.__setattr__(m, "model_name", real)
                object.__setattr__(m, "_query_engine", real)
                object.__setattr__(m, "_text_engine", real)
                return m
            raise

    def _build_or_load_vector_index(
        self, VectorStoreIndex, StorageContext, load_index_from_storage, metas
    ):
        """构建或从磁盘加载 VectorStoreIndex（无加权 documents）"""
        from llama_index.core import Document

        documents = [
            Document(
                text=(
                    f"Skill Name: {m.name}\n"
                    f"Description: {m.description}\n\n"
                    f"{m.instructions}"
                ),
                metadata={
                    "skill_name": m.name,
                    "skill_dir": str(m.skill_dir) if m.skill_dir else "",
                },
            )
            for m in metas
        ]

        self.persist_dir.mkdir(parents=True, exist_ok=True)

        if (
            not self._needs_rebuild()
            and (self.persist_dir / "docstore.json").exists()
        ):
            logger.info("从本地缓存加载 Skills 索引...")
            try:
                ctx = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
                self._index = load_index_from_storage(ctx)
                return
            except Exception as e:
                logger.warning(f"加载缓存索引失败: {e}，重新构建...")

        logger.info(f"构建 Skills 向量索引... ({len(documents)} 个 Skills)")
        self._index = VectorStoreIndex.from_documents(documents)
        self._index.storage_context.persist(persist_dir=str(self.persist_dir))
        self._save_hash()

    def _build_weighted_bm25_nodes(self, TextNode, metas):
        """为 BM25 生成加权 + 预分词的 nodes

        - CJK 预分词（英文整词 + 中文单字用空格分）让 token_pattern=r"\\S+" 吃到单字
        - 加权：把分词后 token 序列重复 N 次实现 tf 放大
        - 最终文本示例：
              name(wn=3): "web_search web_search web_search"
              description(wd=2): "联 网 搜 索 联 网 搜 索"
              instructions(×1): 正文分词结果
          全部 space-join
        """
        wn = max(1, int(self._cfg.bm25_weight_name))
        wd = max(1, int(self._cfg.bm25_weight_description))
        nodes = []
        for m in metas:
            name_tk = " ".join(_cjk_tokenizer(m.name))
            desc_tk = " ".join(_cjk_tokenizer(m.description or ""))
            inst_tk = " ".join(_cjk_tokenizer(m.instructions or ""))
            text = " ".join([name_tk] * wn + [desc_tk] * wd + [inst_tk]).strip()
            nodes.append(
                TextNode(
                    text=text,
                    metadata={
                        "skill_name": m.name,
                        "skill_dir": str(m.skill_dir) if m.skill_dir else "",
                    },
                )
            )
        return nodes

    def _assemble_active_retriever(self, mode_cfg: str, vector_ok: bool, bm25_ok: bool):
        """按 retrieval_mode + 各路径实际就绪情况选定 active_retriever"""
        if mode_cfg == "hybrid" and vector_ok and bm25_ok:
            try:
                from llama_index.core.retrievers import QueryFusionRetriever
                from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
                # 用 MockLLM 绕过构造期校验
                try:
                    from llama_index.core.llms import MockLLM
                    _fake_llm = MockLLM()
                except Exception:
                    _fake_llm = None
                self._hybrid_retriever = QueryFusionRetriever(
                    retrievers=[self._vector_retriever, self._bm25_retriever],
                    llm=_fake_llm,
                    similarity_top_k=self.top_k,
                    num_queries=1,                     # 不做 query expansion
                    mode=FUSION_MODES.RECIPROCAL_RANK,
                    use_async=False,
                    verbose=False,
                )
                self._active_retriever = self._hybrid_retriever
                self._active_mode = "hybrid"
                return
            except Exception as e:
                logger.warning(
                    f"QueryFusionRetriever 构造失败: {e!r}，降级为单路"
                )

        # 非 hybrid / 降级路径
        if vector_ok:
            self._active_retriever = self._vector_retriever
            self._active_mode = "vector_only"
        elif bm25_ok:
            self._active_retriever = self._bm25_retriever
            self._active_mode = "bm25_only"
        else:
            self._active_retriever = None
            self._active_mode = "none"
            logger.warning(
                "Vector 与 BM25 通路均不可用，Skills 召回退化为全量加载。"
            )

    # 检索主流程

    def _exact_name_matches(self, query: str) -> list[SkillMeta]:
        """query 里出现完整 skill_name / skill_key（长度 ≥3）→ 直接命中"""
        if not self._cfg.exact_match_short_circuit:
            return []
        ql = (query or "").lower()
        if not ql:
            return []
        out: list[SkillMeta] = []
        for meta in self._skill_metas.values():
            if meta.always:
                continue  # always=true 由单独分支处理
            keys = [meta.name.lower()]
            if meta.skill_key:
                keys.append(meta.skill_key.lower())
            for k in keys:
                if len(k) >= 3 and k in ql:
                    out.append(meta)
                    break
        return out

    def retrieve_skills(self, query: str) -> list[SkillMeta]:
        """五步检索流程"""
        if not self._initialized:
            self.initialize()

        results: list[SkillMeta] = []
        seen: set[str] = set()

        def _add(meta: SkillMeta):
            if meta.name not in seen:
                seen.add(meta.name)
                results.append(meta)

        # Step 1: 先加 always skill
        always_skills = [m for m in self._skill_metas.values() if m.always]
        for m in always_skills:
            _add(m)

        # S2：空 query 短路 —— 无检索依据，返回 always + 全量（供下游自行决定）
        q_stripped = (query or "").strip()
        if not q_stripped:
            for m in self._skill_metas.values():
                _add(m)
            return results

        # Step 2: 精确名匹配
        for m in self._exact_name_matches(query):
            _add(m)

        # Step 3/4: 缓存 -> 检索
        if self._active_retriever is None:
            # 全量降级时直接返回全部
            for m in self._skill_metas.values():
                _add(m)
            return results

        try:
            # P2：缓存 key 标准化，避免大小写 / 首尾空格导致重复 miss
            cache_key = q_stripped.lower()
            nodes = None
            if self._query_cache is not None:
                nodes = self._query_cache.get(cache_key)
            if nodes is None:
                nodes = self._active_retriever.retrieve(query)
                if self._query_cache is not None:
                    self._query_cache.put(cache_key, nodes)

            # Step 5: 阈值过滤 + 去重
            # threshold 仅在 vector_only 下生效
            apply_threshold = self._active_mode == "vector_only"

            for node in nodes:
                if (
                    apply_threshold
                    and getattr(node, "score", None) is not None
                    and node.score < self.similarity_threshold
                ):
                    continue
                skill_name = node.metadata.get("skill_name", "")
                if skill_name and skill_name in self._skill_metas:
                    _add(self._skill_metas[skill_name])

            # 如果全被阈值过滤，至少保底一条
            non_always_count = sum(1 for m in results if not m.always)
            if non_always_count == 0 and nodes:
                first_name = nodes[0].metadata.get("skill_name", "")
                if first_name in self._skill_metas:
                    _add(self._skill_metas[first_name])

            # 截断到 top_k + always
            max_len = self.top_k + len(always_skills)
            return results[:max_len]

        except Exception as e:
            logger.error(f"Skills 检索失败: {e}，返回全量 Skills。")
            return list(self._skill_metas.values())

    # ---------- 对外接口 ----------

    def get_tools_for_query(self, query: str) -> list[StructuredTool]:
        """根据查询返回最相关的 LangChain Tools"""
        matched_skills = self.retrieve_skills(query)
        tools = []
        for skill in matched_skills:
            tools.extend(build_tool_from_skill(skill))

        skill_names = [s.name for s in matched_skills]
        logger.info(f"查询 '{query[:50]}...' 召回 Skills: {skill_names}")
        return tools

    def get_all_tools(self) -> list[StructuredTool]:
        """获取所有 Skills 对应的 Tools（不做语义过滤）"""
        if not self._initialized:
            self.initialize()

        tools = []
        for skill in self._skill_metas.values():
            tools.extend(build_tool_from_skill(skill))
        return tools

    def rebuild_index(self):
        """强制重建索引 + 清查询缓存"""
        self._initialized = False
        self._index = None
        self._vector_retriever = None
        self._bm25_retriever = None
        self._hybrid_retriever = None
        self._active_retriever = None
        self._active_mode = "none"
        self._skill_metas.clear()
        if self._query_cache is not None:
            self._query_cache.clear()

        if self.persist_dir.exists():
            import shutil
            shutil.rmtree(self.persist_dir, ignore_errors=True)

        self.initialize()
