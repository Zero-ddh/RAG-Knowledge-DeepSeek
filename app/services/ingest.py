"""文档入库流程：保存 -> 解析 -> 切分 -> BGE 向量化 -> 写入 Chroma。"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.models.schemas import DocumentInfo
from app.services import registry
from app.services.loader import DocumentParseError, load_sections
from app.services.vector_store import delete_document_chunks, get_vector_store, store_lock

logger = logging.getLogger(__name__)

# 中文友好的递归切分分隔符
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


class IngestError(Exception):
    """入库失败（消息可直接展示给用户）。"""


def sanitize_filename(filename: str) -> str:
    name = Path(filename.replace("\\", "/")).name.strip()
    return name or "未命名文件"


def save_and_ingest(filename: str, data: bytes, content_type: str | None = None) -> DocumentInfo:
    settings = get_settings()
    settings.ensure_dirs()

    safe_name = sanitize_filename(filename)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise IngestError(
            f"不支持的文件类型 {suffix or '(无扩展名)'}，仅支持 txt / md / pdf / docx"
        )
    if not data:
        raise IngestError("文件内容为空")
    if len(data) > settings.max_upload_bytes:
        raise IngestError(f"文件超过大小限制（{settings.max_upload_mb} MB）")

    doc_id = uuid.uuid4().hex
    stored_path = settings.upload_dir / f"{doc_id}{suffix}"
    stored_path.write_bytes(data)

    try:
        sections = load_sections(stored_path)
        documents = _split_sections(sections, doc_id, safe_name)
        if not documents:
            raise IngestError("切分后没有产生有效文本片段，请检查文档内容")

        ids = [f"{doc_id}:{index}" for index in range(len(documents))]
        with store_lock:
            get_vector_store().add_documents(documents, ids=ids)
    except DocumentParseError as exc:
        stored_path.unlink(missing_ok=True)
        raise IngestError(str(exc)) from exc
    except Exception as exc:
        stored_path.unlink(missing_ok=True)
        try:
            delete_document_chunks(doc_id)
        except Exception:  # pragma: no cover - 清理失败不影响主错误
            logger.exception("清理失败的向量片段时出错：doc_id=%s", doc_id)
        if isinstance(exc, IngestError):
            raise
        logger.exception("文档入库失败：%s", safe_name)
        raise IngestError(f"文档向量化或写入向量库失败：{exc}") from exc

    document = DocumentInfo(
        doc_id=doc_id,
        filename=safe_name,
        size=len(data),
        chunk_count=len(documents),
        created_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        content_type=content_type,
    )
    registry.add_document(document)
    logger.info("文档入库完成：%s（%d 个片段）", safe_name, len(documents))
    return document


def _split_sections(sections, doc_id: str, filename: str) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=_SEPARATORS,
        length_function=len,
    )

    documents: list[Document] = []
    for section in sections:
        for text in splitter.split_text(section.text):
            content = text.strip()
            if not content:
                continue
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "doc_id": doc_id,
                        "filename": filename,
                        "chunk_index": len(documents),
                        "page": section.page or 0,
                    },
                )
            )
    return documents
