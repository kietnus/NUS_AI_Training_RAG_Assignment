"""Application settings, loaded from environment variables or a .env file."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Looks for .env in the current directory and in the project root (when run from backend/).
    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Infrastructure -------------------------------------------------
    database_url: str = "postgresql://nusaskify:nusaskify@localhost:5433/nusaskify"
    ollama_host: str = "http://localhost:11434"
    request_timeout_s: float = 300.0

    # --- Models ---------------------------------------------------------
    embedding_model: str = "qwen3-embedding:0.6b"
    # Must match the embedding model's output dimension (qwen3-embedding:0.6b -> 1024).
    embedding_dim: int = 1024
    # Qwen3-Embedding works best when queries (not documents) carry a task instruction.
    # Set to an empty string for embedding models that do not use instructions.
    query_instruction: str = (
        "Given a question about internal company policies, "
        "retrieve the policy passages that answer the question"
    )
    chat_model: str = "qwen3:4b-instruct"
    llm_temperature: float = 0.0
    llm_num_ctx: int = 8192

    # --- Retrieval ------------------------------------------------------
    default_top_k: int = 5
    # If the best chunk scores below this cosine similarity, /api/ask answers
    # "not enough information" without calling the LLM. Calibrate with /api/search.
    min_score: float = 0.40

    # --- Chunking -------------------------------------------------------
    # Semantic chunking: break where similarity between adjacent pieces drops.
    semantic_threshold_type: Literal["percentile", "absolute"] = "percentile"
    # percentile mode: adjacent pairs whose similarity is in the lowest N% become breakpoints.
    semantic_breakpoint_percentile: float = 25.0
    # absolute mode: break when similarity < this value.
    semantic_similarity_threshold: float = 0.55
    # Each piece is embedded together with this many neighbours on each side (0 = piece only).
    semantic_buffer_size: int = 1
    max_chunk_chars: int = 1200
    min_chunk_chars: int = 150
    # Structure-based chunking (by heading / paragraph).
    structure_max_chunk_chars: int = 1200

    # --- Upload ---------------------------------------------------------
    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
