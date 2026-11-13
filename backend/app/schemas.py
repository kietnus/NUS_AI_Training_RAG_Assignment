"""Pydantic request/response models (also drive the OpenAPI / Swagger docs)."""

from datetime import date, datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

ChunkingMethod = Literal["semantic", "structure"]
EffectiveStatus = Literal["current", "superseded", "expired", "not_yet_valid"]


class ErrorResponse(BaseModel):
    detail: str = Field(examples=["Missing required metadata: valid_to. The document was not indexed."])
    error: str = Field(examples=["invalid_metadata"])


# --------------------------------------------------------------------------- documents


class DocumentOut(BaseModel):
    id: UUID
    tenant_id: str
    title: str
    version: str
    valid_from: date
    valid_to: date
    status: Literal["active", "expired"] = Field(description="Stored status ('expired' = superseded by a newer version).")
    effective_status: EffectiveStatus = Field(
        description="Status as used by retrieval today: only 'current' documents are searched.")
    superseded_by: UUID | None
    file_name: str
    file_type: Literal["pdf", "docx"]
    file_size: int
    char_count: int
    semantic_chunks: int
    structure_chunks: int
    created_at: datetime


class DocumentDetailOut(DocumentOut):
    ingestion_stats: dict[str, Any]


class VersionRef(BaseModel):
    id: UUID
    version: str


class UploadResponse(BaseModel):
    document: DocumentDetailOut
    chunks_created: dict[str, int]
    expired_documents: list[VersionRef] = Field(
        description="Older versions marked as expired because this upload is newer.")
    stored_as_expired_because_newer_version: VersionRef | None = Field(
        description="Set when the uploaded file is older than an existing active version.")
    ingestion_ms: int


class DeleteResponse(BaseModel):
    id: UUID
    title: str
    version: str
    chunks_deleted: int


class ChunkOut(BaseModel):
    id: UUID
    document_id: UUID
    chunking_method: ChunkingMethod
    chunk_index: int
    content: str
    char_count: int
    metadata: dict[str, Any]
    embedding_dim: int
    embedding_preview: list[float] = Field(description="First 8 values of the embedding vector.")
    created_at: datetime


class DocumentChunksOut(BaseModel):
    document: DocumentDetailOut
    chunks: list[ChunkOut]


# --------------------------------------------------------------------------- search / ask


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000,
                          examples=["How many days per week can I work from home?"])
    document_ids: list[UUID] = Field(min_length=1, description="Search scope: selected documents.")
    top_k: int = Field(default=5, ge=1, le=20)
    chunking_method: ChunkingMethod = Field(
        default="semantic", description="Which chunk set to search (for comparing strategies).")


class AskRequest(SearchRequest):
    pass


class RetrievedChunk(BaseModel):
    rank: int = Field(description="1-based rank; also the citation marker number [rank].")
    score: float = Field(description="Cosine similarity between question and chunk (higher = more similar).")
    chunk_id: UUID
    document_id: UUID
    document_title: str
    version: str
    valid_from: date
    valid_to: date
    chunk_index: int
    chunking_method: ChunkingMethod
    content: str


class SearchResponse(BaseModel):
    question: str
    chunking_method: ChunkingMethod
    top_k: int
    results: list[RetrievedChunk]
    query_log_id: UUID | None


class Citation(BaseModel):
    marker: int = Field(description="The [n] marker used in the answer; equals the chunk's rank.")
    chunk_id: UUID
    document_id: UUID
    document_title: str
    version: str
    chunk_index: int
    score: float
    quote: str | None = Field(description="Supporting quote given by the LLM.")
    quote_verified: bool | None = Field(
        description="True if the quote was found in the cited chunk's text; None if no quote.")


class AskResponse(BaseModel):
    status: Literal["answered", "not_enough_information"]
    answer: str
    citations: list[Citation]
    chunks: list[RetrievedChunk]
    reason: str | None = Field(description="Why 'not enough information' was returned.")
    top_score: float | None
    min_score: float
    dropped_citations: list[int] = Field(
        description="Citation numbers produced by the LLM that did not match a retrieved chunk (removed).")
    query_log_id: UUID | None


class QueryLogOut(BaseModel):
    id: UUID
    tenant_id: str
    endpoint: Literal["search", "ask"]
    question: str
    document_ids: list[UUID]
    chunking_method: str
    top_k: int
    retrieved_chunk_ids: list[UUID]
    retrieved_scores: list[float]
    status: str | None
    reason: str | None
    answer: str | None
    citations: list[dict[str, Any]] | None
    llm_raw_output: str | None
    embedding_model: str
    chat_model: str | None
    latency_ms: int | None
    created_at: datetime


class HealthOut(BaseModel):
    database: str
    ollama: str
    embedding_model: str
    embedding_model_available: bool | None
    embedding_dim: int
    chat_model: str
    chat_model_available: bool | None
    min_score: float
