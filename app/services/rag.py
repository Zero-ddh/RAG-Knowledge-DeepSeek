"""检索增强生成：向量检索 -> 拼装上下文 -> 调用 DeepSeek -> 返回带引用的答案。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import get_settings
from app.core.prompts import NO_HIT_ANSWER, SYSTEM_PROMPT, build_user_prompt
from app.models.schemas import AskResponse, SourceSnippet
from app.services.llm import get_chat_model
from app.services.vector_store import get_vector_store


@dataclass(slots=True)
class RetrievedChunk:
    content: str
    doc_id: str
    filename: str
    chunk_index: int
    page: int
    score: float


def retrieve(question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    settings = get_settings()
    k = top_k or settings.top_k
    results = get_vector_store().similarity_search_with_score(question, k=k)

    chunks: list[RetrievedChunk] = []
    for document, distance in results:
        # collection 使用 cosine 空间，distance = 1 - cosine_similarity
        score = max(0.0, min(1.0, 1.0 - float(distance)))
        if score < settings.score_threshold:
            continue
        metadata = document.metadata or {}
        chunks.append(
            RetrievedChunk(
                content=document.page_content,
                doc_id=str(metadata.get("doc_id", "")),
                filename=str(metadata.get("filename", "未知文件")),
                chunk_index=int(metadata.get("chunk_index", 0)),
                page=int(metadata.get("page", 0) or 0),
                score=round(score, 4),
            )
        )
    return chunks


def build_context(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        location = f"{chunk.filename} 第 {chunk.page} 页" if chunk.page else chunk.filename
        parts.append(f"[{index}] 来源：{location}（片段 {chunk.chunk_index + 1}）\n{chunk.content}")
    return "\n\n".join(parts)


def to_sources(chunks: list[RetrievedChunk]) -> list[SourceSnippet]:
    return [
        SourceSnippet(
            index=index,
            doc_id=chunk.doc_id,
            filename=chunk.filename,
            chunk_index=chunk.chunk_index,
            score=chunk.score,
            content=chunk.content,
        )
        for index, chunk in enumerate(chunks, start=1)
    ]


def _message_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, str):
                pieces.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                pieces.append(str(item.get("text", "")))
        return "".join(pieces)
    return "" if content is None else str(content)


async def answer_question(question: str, top_k: int | None = None) -> AskResponse:
    question = question.strip()
    llm = get_chat_model()
    chunks = await asyncio.to_thread(retrieve, question, top_k)
    sources = to_sources(chunks)
    if not chunks:
        return AskResponse(answer=NO_HIT_ANSWER, sources=[])

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(question, build_context(chunks))),
    ]
    response = await llm.ainvoke(messages)
    return AskResponse(answer=_message_text(response.content).strip(), sources=sources)


async def stream_answer(
    question: str, top_k: int | None = None
) -> AsyncIterator[tuple[str, dict]]:
    """产出 (event, payload)：sources -> delta... -> done。"""
    question = question.strip()
    llm = get_chat_model(streaming=True)
    chunks = await asyncio.to_thread(retrieve, question, top_k)
    sources = to_sources(chunks)
    yield "sources", {"sources": [source.model_dump() for source in sources]}

    if not chunks:
        yield "delta", {"text": NO_HIT_ANSWER}
        yield "done", {"answer": NO_HIT_ANSWER}
        return

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_user_prompt(question, build_context(chunks))),
    ]
    collected: list[str] = []
    async for part in llm.astream(messages):
        text = _message_text(part.content)
        if text:
            collected.append(text)
            yield "delta", {"text": text}

    yield "done", {"answer": "".join(collected)}
