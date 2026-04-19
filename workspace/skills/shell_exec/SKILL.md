---
name: shell_exec
description: 在本地终端执行 Shell 命令
version: "1.0"
dependencies: []
---

# Shell Exec Skill

执行本地 Shell 命令的技能。

## 使用场景

- 运行系统命令获取信息（如 ls, cat, grep）
- 执行开发相关命令（如 git, pip, conda）
- 运行脚本或程序

## 安全规则

- **禁止** 执行以下危险命令: `rm -rf /`, `mkfs`, `dd if=`, `:(){:|:&};:`
- **需确认** 的命令: 含 `rm`, `sudo`, `chmod 777`, `kill -9` 的命令
- 命令超时限制: 30 秒
- 输出长度限制: 最多返回前 5000 字符
