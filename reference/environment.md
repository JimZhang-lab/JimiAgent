# 项目运行环境

## Python 环境
```bash
conda activate jimiAgent312
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 启动服务
```bash
# 方式一：直接启动 Gateway
python main.py

# 方式二：通过 CLI
python cli.py start

# 方式三：交互式对话（终端）
python cli.py chat
```

## 健康检查
```bash
python cli.py doctor
```

## 访问
- Web UI: http://localhost:18789
- API: http://localhost:18789/api/status
- WebSocket: ws://localhost:18789/ws/chat
