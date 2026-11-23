"""全局配置：所有可调参数均通过环境变量 / .env 注入。"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- 服务 ----------
    app_name: str = "RAG 知识库问答系统"
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: str = "*"

    # ---------- DeepSeek ----------
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024
    llm_timeout: float = 60.0

    # ---------- Embedding ----------
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "auto"
    embedding_batch_size: int = 32
    embedding_query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    model_cache_dir: Path | None = None
    hf_endpoint: str | None = None

    # ---------- Chroma ----------
    chroma_dir: Path = BASE_DIR / "data" / "chroma"
    chroma_collection: str = "knowledge_base"

    # ---------- 文档切分与检索 ----------
    chunk_size: int = Field(default=700, ge=100, le=4000)
    chunk_overlap: int = Field(default=120, ge=0)
    top_k: int = Field(default=4, ge=1, le=20)
    score_threshold: float = Field(default=0.2, ge=0.0, le=1.0)

    # ---------- 上传 ----------
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    registry_path: Path = BASE_DIR / "data" / "documents.json"
    max_upload_mb: int = Field(default=20, ge=1, le=200)

    @field_validator("chroma_dir", "upload_dir", "registry_path", "model_cache_dir", mode="before")
    @classmethod
    def _resolve_path(cls, value: object) -> object:
        if value is None or value == "":
            return None
        path = Path(str(value)).expanduser()
        return path if path.is_absolute() else (BASE_DIR / path).resolve()

    @field_validator("hf_endpoint")
    @classmethod
    def _normalize_hf_endpoint(cls, value: str | None) -> str | None:
        if not value:
            return None
        return value.rstrip("/")

    @model_validator(mode="after")
    def _check_chunk_params(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP 必须小于 CHUNK_SIZE")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def deepseek_configured(self) -> bool:
        key = (self.deepseek_api_key or "").strip()
        return bool(key) and not key.startswith("sk-your-")

    @property
    def allowed_extensions(self) -> set[str]:
        return {".txt", ".md", ".markdown", ".pdf", ".docx"}

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for path in (self.chroma_dir, self.upload_dir, self.registry_path.parent):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
