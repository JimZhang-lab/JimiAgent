'''
Author: JimZhang
Date: 2026-04-18 22:10:00
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/workspace/skills/web_search/search.py
Description: Web 搜索 Skill。

'''
import httpx


async def search(query: str, max_results: int = 5) -> str:
    """执行网络搜索并返回结果摘要"""
    # DuckDuckGo 即时回答接口，无需 Key
    url = "https://api.duckduckgo.com/"
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1,
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        results = []

        # 即时回答
        if data.get("AbstractText"):
            results.append(f"摘要: {data['AbstractText']}")
            if data.get("AbstractURL"):
                results.append(f"来源: {data['AbstractURL']}")

        # 相关主题
        for topic in data.get("RelatedTopics", [])[:max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"- {topic['Text']}")

        if results:
            return "\n".join(results)
        return f"搜索 '{query}' 未找到相关结果，请尝试其他关键词。"

    except Exception as e:
        return f"搜索出错: {str(e)}"
