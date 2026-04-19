'''K5 - 插件槽位加载。'''
from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any, Optional

from server.core.plugin_registry import PluginRegistry, PluginState

logger = logging.getLogger(__name__)


def _load_plugin_module(state: PluginState, module_name: str = "runtime"):
    """从插件目录动态导入 runtime.py。失败返回 None。"""
    path = state.manifest.source / f"{module_name}.py"
    if not path.exists():
        logger.warning(
            f"slot 期望插件 {state.id} 提供 {module_name}.py，但不存在"
        )
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            f"openclaw_plugin_{state.id}_{module_name}", path,
        )
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    except Exception as e:
        logger.warning(f"加载插件 {state.id}/{module_name}.py 失败: {e}")
        return None


def resolve_memory_slot(
    registry: PluginRegistry, settings,
) -> tuple[str, Optional[Any]]:
    """解析并返回 Memory slot 策略与实例"""
    slots = (settings.plugins.slots or {}) if settings.plugins.enabled else {}
    sel = (slots.get("memory") or "").strip() if isinstance(slots, dict) else ""
    sel_low = sel.lower()

    if sel_low in ("", "memory-core", "default", "builtin"):
        return "builtin", None
    if sel_low == "none":
        return "none", None

    state = registry.get(sel)
    if state is None or not state.enabled:
        logger.warning(
            f"plugins.slots.memory={sel} 未找到或未启用，回退 builtin"
        )
        return "builtin", None
    if state.manifest.kind not in ("memory", ""):
        logger.warning(
            f"plugins.slots.memory={sel} 的 manifest.kind={state.manifest.kind!r} "
            f"与 memory slot 不匹配，回退 builtin"
        )
        return "builtin", None

    mod = _load_plugin_module(state, "runtime")
    factory = getattr(mod, "create_memory_store", None) if mod else None
    if factory is None:
        logger.warning(
            f"plugin {sel} 缺 create_memory_store(settings) 工厂，回退 builtin"
        )
        return "builtin", None

    try:
        inst = factory(settings)
    except Exception as e:
        logger.warning(f"plugin {sel}.create_memory_store 抛异常: {e}；回退 builtin")
        return "builtin", None

    return f"plugin:{sel}", inst


def resolve_context_engine_slot(
    registry: PluginRegistry, settings,
) -> tuple[str, Optional[Any]]:
    """解析并返回 Context Engine slot 策略与实例"""
    slots = (settings.plugins.slots or {}) if settings.plugins.enabled else {}
    sel = (slots.get("contextEngine") or "").strip() if isinstance(slots, dict) else ""
    sel_low = sel.lower()

    if sel_low in ("", "none", "default", "builtin"):
        return "builtin", None  # 由 SkillRetriever 继续扮演

    state = registry.get(sel)
    if state is None or not state.enabled:
        logger.warning(
            f"plugins.slots.contextEngine={sel} 未找到或未启用，回退 builtin"
        )
        return "builtin", None
    if state.manifest.kind not in ("context-engine", ""):
        logger.warning(
            f"plugins.slots.contextEngine={sel} 的 kind={state.manifest.kind!r} "
            f"与 contextEngine slot 不匹配，回退 builtin"
        )
        return "builtin", None

    mod = _load_plugin_module(state, "runtime")
    factory = getattr(mod, "create_context_engine", None) if mod else None
    if factory is None:
        logger.warning(
            f"plugin {sel} 缺 create_context_engine(settings) 工厂，回退 builtin"
        )
        return "builtin", None

    try:
        inst = factory(settings)
    except Exception as e:
        logger.warning(
            f"plugin {sel}.create_context_engine 抛异常: {e}；回退 builtin"
        )
        return "builtin", None

    return f"plugin:{sel}", inst
