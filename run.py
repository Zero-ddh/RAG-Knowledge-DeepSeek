"""简单的启动入口：python run.py"""

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    print(f"服务启动中：http://127.0.0.1:{settings.port}  （Ctrl+C 停止）")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
