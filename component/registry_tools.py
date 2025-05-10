from typing import Callable, Dict, Any, get_type_hints
from pydantic import BaseModel, Field, create_model
import inspect
import json

# 全局工具映射字典
tool_map: Dict[str, dict] = {}

def tool_registry(param_descriptions: Dict[str, str] = None):
    """
    装饰器：自动从函数参数注解生成 Pydantic 模型，并注册函数到 tool_map
    param_descriptions: 可选的参数描述字典
    """
    def decorator(func: Callable):
        # 获取函数签名和参数注解
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)

        # 构建字段字典用于创建 Pydantic 模型
        fields = {}
        for name, param in sig.parameters.items():
            annotation = type_hints.get(name, Any)
            default = param.default if param.default is not inspect.Parameter.empty else ...
            description = param_descriptions.get(name, f"{name} 参数") if param_descriptions else f"{name} 参数"
            fields[name] = (annotation, Field(default, description=description))

        # 创建 Pydantic 模型
        model_name = f"{func.__name__.capitalize()}Params"
        ParamModel = create_model(model_name, **fields)

        # 提取函数文档字符串作为描述
        doc = func.__doc__ or ""
        description = doc.strip().split('\n')[0]

        # 获取参数模型的 JSON Schema
        schema = ParamModel.model_json_schema()

        # 构建工具定义
        tool_map[func.__name__] = {
            "type": "function",
            "function": {
                "name": func.__name__,
                "description": description,
                "parameters": schema
            }
        }

        return func
    return decorator


def get_tools(tool_name_list: list):
    if not isinstance(tool_name_list, list):
        raise ValueError(f"tool_name_list type must be 'list', but got {type(tool_name_list)}")
    
    tools = []
    for tool_name in tool_name_list:
        if tool_name not in tool_map.keys():
            raise ValueError(f"tool name must select from {tool_map.keys()}")
        tools.append(tool_map[tool_name])
    
    return tools