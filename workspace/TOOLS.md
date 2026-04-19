# TOOLS — 环境信息

## 运行环境

- **操作系统**: macOS
- **Python**: 3.12 (conda: jimiAgent312)
- **项目路径**: /Users/jim/Desktop/Code/Project/JimiAgent

## 可用工具

以下工具通过 Skills 系统动态加载，可在对话中使用：

- **web_search**: 网络搜索信息
- **file_ops**: 读写文件、列出目录
- **shell_exec**: 执行 Shell 命令
- **datetime_info**: 获取当前日期时间信息

## API 配置

- **LLM Provider**: OpenAI 兼容接口 (qwen3-32B)
- **Embedding**: 复用 LLM Provider 的 Embedding 接口
