"""BGE 中文向量模型（sentence-transformers）的 LangChain Embeddings 适配器。"""

from __future__ import annotations

import logging
import os
import threading

from langchain_core.embeddings import Embeddings

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class BGEEmbeddings(Embeddings):
    """懒加载 sentence-transformers 模型，避免服务启动时长时间阻塞。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = None
        self._lock = threading.Lock()
        self._device: str | None = None

    @property
    def model_name(self) -> str:
        return self._settings.embedding_model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str | None:
        return self._device

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model

            if self._settings.hf_endpoint:
                os.environ.setdefault("HF_ENDPOINT", self._settings.hf_endpoint)
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "缺少 sentence-transformers / torch 依赖，请先执行 "
                    "pip install -r requirements.txt"
                ) from exc

            device = self._settings.embedding_device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            logger.info("正在加载 Embedding 模型 %s（device=%s）", self.model_name, device)
            cache_folder = (
                str(self._settings.model_cache_dir) if self._settings.model_cache_dir else None
            )
            self._model = SentenceTransformer(
                self.model_name, device=device, cache_folder=cache_folder
            )
            self._device = device
            logger.info("Embedding 模型加载完成")
            return self._model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=self._settings.embedding_batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([text.replace("\n", " ") for text in texts])

    def embed_query(self, text: str) -> list[float]:
        instruction = self._settings.embedding_query_instruction.strip()
        query = f"{instruction}{text}" if instruction else text
        return self._encode([query.replace("\n", " ")])[0]
