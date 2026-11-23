"""问答接口：普通 JSON 与 SSE 流式两种模式。"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import AskRequest, AskResponse
from app.services.llm import LLMNotConfiguredError
from app.services.rag import answer_question, stream_answer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _sse(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("", response_model=AskResponse, summary="知识库问答")
async def chat(request: AskRequest) -> AskResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    try:
        return await answer_question(question, request.top_k)
    except LLMNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("问答生成失败")
        raise HTTPException(status_code=502, detail=f"模型调用失败：{exc}") from exc


@router.post("/stream", summary="知识库问答（流式 SSE）")
async def chat_stream(request: AskRequest) -> StreamingResponse:
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")

    async def event_stream():
        try:
            async for event, payload in stream_answer(question, request.top_k):
                yield _sse(event, payload)
        except LLMNotConfiguredError as exc:
            yield _sse("error", {"detail": str(exc)})
        except Exception as exc:
            logger.exception("流式问答失败")
            yield _sse("error", {"detail": f"生成回答失败：{exc}"})

    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS)
