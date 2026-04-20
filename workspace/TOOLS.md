# TOOLS — 环境信息

## 运行环境

- **操作系统**: macOS
- **Python**: 3.12 (conda: jimiAgent312)
- **项目路径**: /Users/jim/Desktop/Code/Project/JimiAgent

## 可用工具

### 内置（builtin）工具 · 总是可用

- `read_file(path, offset, limit)`：读文件内容（支持行偏移）
- `write_file(path, content, confirm=False)`：写文件（workspace 外需 confirm=true）
- `edit_file(path, old_text, new_text, confirm=False)`：精确替换
- `list_dir(path, max_items)`：列目录（默认 workspace 根目录）
- `bash(command, confirm=False, cwd="")`：执行 shell 命令（读类直放；写类需 confirm=true；cwd 可指定）

### Skill 工具 · 按需召回

- `file_ops_*`：`read_file` / `write_file` / `list_directory`（与 builtin 等价，调同一安全层）
- `shell_exec_execute`：执行命令（与 builtin bash 等价）
- `web_search`：网络搜索
- `datetime_info`：获取当前时间

## 权限边界

- **读**：任意本地路径都能读（`~/Desktop`、`/tmp`、`/Users/xxx/` 等）。只有系统敏感目录（`/etc/`、`~/.ssh/`、`/System/`、`/usr/bin/` 等）硬拒
- **写 workspace 内**：直接放行，零摩擦
- **写 workspace 外**：安全层自动触发用户 confirm 弹框（TUI 按 y/N）；批准后执行、拒绝则返回 `[CANCELLED]`
- **毁灭性命令**（`rm -rf /`、`mkfs`、`dd of=/dev/*`、`shutdown` 等）：硬拒，`confirm=true` 也无效
- **重要**：所有 confirm 流程**由系统自动处理**，你只需正常调用工具。不要因为"可能需要权限"而不去调用

## API 配置

- **LLM Provider**: OpenAI 兼容接口 (qwen3-32B)
- **Embedding**: 复用 LLM Provider 的 Embedding 接口
