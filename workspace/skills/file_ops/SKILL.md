---
name: file_ops
description: 读取、写入文件和列出目录内容
version: "1.0"
dependencies: []
---

# File Operations Skill

文件系统操作技能，支持读写文件和浏览目录。

## 使用场景

- 读取本地文件内容
- 创建或修改文件
- 列出目录结构
- 检查文件是否存在

## 可用操作

- `read_file`: 读取指定文件的内容
- `write_file`: 将内容写入指定文件
- `list_directory`: 列出指定目录的内容

## 安全注意

- 写入前确认用户意图
- 避免覆盖重要系统文件
- 操作路径限制在项目目录范围内
