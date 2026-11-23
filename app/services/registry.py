"""文档登记表：以 JSON 文件记录已入库文档的元信息。"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from app.config import get_settings
from app.models.schemas import DocumentInfo

logger = logging.getLogger(__name__)

_lock = threading.Lock()


def _registry_path() -> Path:
    return get_settings().registry_path


def _read_all() -> list[dict]:
    path = _registry_path()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("文档登记表读取失败，将使用空列表：%s", exc)
        return []
    items = data.get("documents", []) if isinstance(data, dict) else []
    return [item for item in items if isinstance(item, dict)]


def _write_all(items: list[dict]) -> None:
    path = _registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({"documents": items}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def list_documents() -> list[DocumentInfo]:
    items = _read_all()
    documents = [DocumentInfo(**item) for item in items]
    documents.sort(key=lambda doc: doc.created_at, reverse=True)
    return documents


def get_document(doc_id: str) -> DocumentInfo | None:
    for item in _read_all():
        if item.get("doc_id") == doc_id:
            return DocumentInfo(**item)
    return None


def add_document(document: DocumentInfo) -> None:
    with _lock:
        items = _read_all()
        items = [item for item in items if item.get("doc_id") != document.doc_id]
        items.append(document.model_dump())
        _write_all(items)


def remove_document(doc_id: str) -> DocumentInfo | None:
    with _lock:
        items = _read_all()
        target = next((item for item in items if item.get("doc_id") == doc_id), None)
        if target is None:
            return None
        _write_all([item for item in items if item.get("doc_id") != doc_id])
    return DocumentInfo(**target)
