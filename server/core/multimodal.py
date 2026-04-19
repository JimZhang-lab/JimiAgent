'''多模态辅助。'''
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from server.core.agent import JimiAgent

logger = logging.getLogger(__name__)


_PROMPT = (
    "用一句中文（不超过 60 字）描述这张图片的核心内容：主要对象、场景、颜色、氛围。"
    "不要猜文件来源，不要加前缀如'这张图片'。"
)


def _build_vl_embedder(settings) -> Optional[object]:
    """构造用于视觉特征的文本 Embedding 模型实例"""
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError:
        return None

    ec = settings.effective_vl_embedding
    if not ec.api_key:
        return None

    kw = {"api_key": ec.api_key, "model": ec.model}
    if ec.base_url:
        kw["api_base"] = ec.base_url
    try:
        try:
            return OpenAIEmbedding(**kw)
        except ValueError as ve:
            if "is not a valid OpenAIEmbeddingModelType" in str(ve):
                safe = dict(kw)
                safe["model"] = "text-embedding-ada-002"
                m = OpenAIEmbedding(**safe)
                object.__setattr__(m, "model_name", ec.model)
                object.__setattr__(m, "_query_engine", ec.model)
                object.__setattr__(m, "_text_engine", ec.model)
                return m
            raise
    except Exception as e:
        logger.debug(f"VL-embedding 构造失败: {e}")
        return None


async def describe_image(agent: "JimiAgent", image_url: str) -> str:
    """调 VLM 对图片生成一句话描述。失败返回空串。"""
    from langchain_core.messages import HumanMessage

    try:
        vlm = getattr(agent, "vlm", None) or agent.llm
        msg = HumanMessage(content=[
            {"type": "text", "text": _PROMPT},
            {"type": "image_url", "image_url": {"url": image_url}},
        ])
        resp = await vlm.ainvoke([msg])
        content = getattr(resp, "content", "") or ""
        if isinstance(content, list):
            # 极少数 provider 会把回答放 list-of-parts
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict)]
            content = " ".join(t for t in text_parts if t)
        return str(content).strip()
    except Exception as e:
        logger.warning(f"VLM 描述失败: {e}")
        return ""


async def describe_and_remember(
    agent: "JimiAgent", image_url: str,
) -> dict:
    """对图片生成描述并存入长期记忆存储"""
    description = await describe_image(agent, image_url)
    result: dict = {
        "description": description,
        "memory_id": 0,
        "embedding_used": False,
    }
    if not description:
        return result

    store = getattr(agent, "memory_store", None)
    if store is None:
        return result

    embedding: Optional[list[float]] = None
    try:
        vle = _build_vl_embedder(agent.settings)
        if vle is not None:
            vec = vle.get_text_embedding(description)
            if vec:
                embedding = list(vec)
                result["embedding_used"] = True
    except Exception as e:
        logger.debug(f"VL-embedding 失败（回退 MemoryStore 自己的 embedding）: {e}")

    try:
        mid = store.add(
            text=f"[image] {description}\nsource: {image_url}",
            kind="episodic",
            subject="upload",
            source_session="",
            embedding=embedding,
        )
        result["memory_id"] = int(mid or 0)
    except Exception as e:
        logger.warning(f"写 MemoryStore 失败: {e}")

    return result
