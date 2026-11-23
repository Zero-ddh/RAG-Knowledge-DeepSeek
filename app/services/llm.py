"""DeepSeek 模型客户端（OpenAI 兼容接口）。"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from app.config import get_settings


class LLMNotConfiguredError(RuntimeError):
    """未配置 API Key 时抛出。"""


def get_chat_model(streaming: bool = False) -> ChatOpenAI:
    settings = get_settings()
    if not settings.deepseek_configured:
        raise LLMNotConfiguredError(
            "未配置 DEEPSEEK_API_KEY，请复制 .env.example 为 .env 并填写真实密钥后重启服务"
        )
    return ChatOpenAI(
        model=settings.deepseek_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        timeout=settings.llm_timeout,
        max_retries=2,
        streaming=streaming,
    )
