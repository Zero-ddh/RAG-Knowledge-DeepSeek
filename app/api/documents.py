"""文档管理接口：上传、列表、删除。"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.models.schemas import DocumentInfo, DocumentListResponse
from app.services import registry
from app.services.ingest import IngestError, save_and_ingest
from app.services.vector_store import delete_document_chunks

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["docs"])


@router.post("", response_model=DocumentInfo, summary="上传并入库文档")
async def upload_document(file: UploadFile = File(...)) -> DocumentInfo:
    settings = get_settings()
    if file.size is not None and file.size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413, detail=f"文件超过大小限制（{settings.max_upload_mb} MB）"
        )
    data = await file.read()
    filename = file.filename or ""
    try:
        return await run_in_threadpool(save_and_ingest, filename, data, file.content_type)
    except IngestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - 兜底
        logger.exception("上传文档失败")
        raise HTTPException(status_code=500, detail=f"文档入库失败：{exc}") from exc


@router.get("", response_model=DocumentListResponse, summary="获取已入库文档列表")
def list_documents() -> DocumentListResponse:
    items = registry.list_documents()
    return DocumentListResponse(total=len(items), items=items)


@router.delete("/{doc_id}", summary="删除文档及其向量片段")
def delete_document(doc_id: str) -> dict:
    document = registry.get_document(doc_id)
    if document is None:
        raise HTTPException(status_code=404, detail="文档不存在")

    removed_chunks = delete_document_chunks(doc_id)
    registry.remove_document(doc_id)

    settings = get_settings()
    for suffix in settings.allowed_extensions:
        (settings.upload_dir / f"{doc_id}{suffix}").unlink(missing_ok=True)

    logger.info("文档已删除：%s（%d 个片段）", document.filename, removed_chunks)
    return {"deleted": True, "doc_id": doc_id, "removed_chunks": removed_chunks}
