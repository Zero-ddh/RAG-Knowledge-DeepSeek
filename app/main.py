"""FastAPI 应用入口：注册路由、挂载静态前端。"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api import chat, documents, health
from app.config import BASE_DIR, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    settings = get_settings()
    settings.ensure_dirs()
    logger.info("Chroma 目录：%s", settings.chroma_dir)
    logger.info("Embedding 模型：%s", settings.embedding_model)
    logger.info("LLM：%s @ %s", settings.deepseek_model, settings.deepseek_base_url)
    if not settings.deepseek_configured:
        logger.warning("未检测到有效的 DEEPSEEK_API_KEY，问答接口将返回 503")
    yield


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=__version__,
    description="基于 LangChain + Chroma + BGE + DeepSeek 的 RAG 知识库问答服务",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_router = APIRouter(prefix="/api")
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
app.include_router(api_router)

static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
