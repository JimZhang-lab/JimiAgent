'''
Author: JimZhang
Date: 2026-04-18 23:52:09
LastEditors: 很拉风的James
LastEditTime: 2026-04-19 13:30:00
FilePath: /JimiAgent/main.py
Description: Uvicorn 启动入口。

'''

import uvicorn
from server.config.settings import get_settings


def main():
    settings = get_settings()
    uvicorn.run(
        "server.api.gateway:app",
        host=settings.gateway.host,
        port=settings.gateway.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
