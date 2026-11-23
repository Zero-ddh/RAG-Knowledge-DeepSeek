"""API 请求 / 响应模型。"""

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    size: int
    chunk_count: int
    created_at: str
    content_type: str | None = None


class DocumentListResponse(BaseModel):
    total: int
    items: list[DocumentInfo]


class SourceSnippet(BaseModel):
    index: int
    doc_id: str
    filename: str
    chunk_index: int
    score: float
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceSnippet]


class HealthResponse(BaseModel):
    status: str
    deepseek_configured: bool
    llm_model: str
    embedding_model: str
    embedding_loaded: bool
    chroma_collection: str
    chunk_count: int
    document_count: int


class ErrorResponse(BaseModel):
    detail: str
