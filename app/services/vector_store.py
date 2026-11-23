"""Chroma 向量库的创建、统计与删除操作。"""

from __future__ import annotations

import logging
import threading
from functools import lru_cache

from langchain_chroma import Chroma

from app.config import get_settings
from app.services.embeddings import BGEEmbeddings

logger = logging.getLogger(__name__)

# 写入 / 删除向量时共用同一把锁，避免并发修改 collection
store_lock = threading.Lock()


@lru_cache
def get_embeddings() -> BGEEmbeddings:
    return BGEEmbeddings(get_settings())


@lru_cache
def get_vector_store() -> Chroma:
    settings = get_settings()
    settings.ensure_dirs()

    import chromadb
    from chromadb.config import Settings as ChromaSettings

    client = chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False, allow_reset=False),
    )
    logger.info("Chroma 已就绪：%s / collection=%s", settings.chroma_dir, settings.chroma_collection)
    return Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def collection_count() -> int:
    store = get_vector_store()
    try:
        return int(store._collection.count())  # noqa: SLF001 - 公开 API 无计数接口
    except Exception:  # pragma: no cover
        return len(store.get(include=[])["ids"])


def delete_document_chunks(doc_id: str) -> int:
    store = get_vector_store()
    with store_lock:
        existing = store.get(where={"doc_id": doc_id}, include=[])
        ids = list(existing.get("ids") or [])
        if ids:
            store.delete(ids=ids)
    return len(ids)
