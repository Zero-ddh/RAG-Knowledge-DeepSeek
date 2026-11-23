"""健康检查接口。"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from app.config import get_settings
from app.models.schemas import HealthResponse
from app.services import registry
from app.services.vector_store import collection_count, get_embeddings

logger = logging.getLogger(__name__)
router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse, summary="健康检查")
def health() -> HealthResponse:
    settings = get_settings()
    status = "ok"
    try:
        chunk_count = collection_count()
    except Exception:  # pragma: no cover
        logger.exception("读取向量库统计信息失败")
        chunk_count = 0
        status = "degraded"

    return HealthResponse(
        status=status,
        deepseek_configured=settings.deepseek_configured,
        llm_model=settings.deepseek_model,
        embedding_model=settings.embedding_model,
        embedding_loaded=get_embeddings().loaded,
        chroma_collection=settings.chroma_collection,
        chunk_count=chunk_count,
        document_count=len(registry.list_documents()),
    )
