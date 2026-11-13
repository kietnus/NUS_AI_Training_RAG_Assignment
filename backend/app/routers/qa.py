"""/api/search, /api/ask, /api/query-logs and /api/health endpoints."""

from typing import Annotated

import ollama
from fastapi import APIRouter, Query

from .. import repository as repo
from ..db import get_pool
from ..deps import ChatDep, EmbedderDep, SettingsDep, TenantId
from ..schemas import AskRequest, AskResponse, ErrorResponse, HealthOut, QueryLogOut, SearchRequest, SearchResponse
from ..services import qa

router = APIRouter(prefix="/api")

_ERRORS = {
    400: {"model": ErrorResponse, "description": "Missing / invalid X-Tenant-Id header"},
    503: {"model": ErrorResponse, "description": "Embedding or chat model (Ollama) unavailable"},
}


@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["search"],
    summary="Semantic search only (no LLM)",
    description=(
        "Embeds the question with the same embedding model used for the chunks and returns the "
        "top-k chunks with cosine-similarity scores. Filters (tenant from `X-Tenant-Id`, selected "
        "`document_ids`, document still valid today) are applied **inside the SQL query, before "
        "top-k**."
    ),
    responses=_ERRORS,
)
def search(body: SearchRequest, tenant_id: TenantId, embedder: EmbedderDep):
    return qa.search(tenant_id=tenant_id, question=body.question, document_ids=body.document_ids,
                     top_k=body.top_k, method=body.chunking_method, embedder=embedder)


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["search"],
    summary="Q&A with citations",
    description=(
        "Flow: identify tenant → filter (tenant, selected documents, still valid) inside the "
        "retrieval query → embed question → top-k → if the best score < `MIN_SCORE` return "
        "`not_enough_information` → LLM answers using only the numbered chunks → every citation "
        "is checked to point to a retrieved chunk (invalid ones are removed; an answer with no "
        "valid citation is discarded)."
    ),
    responses=_ERRORS,
)
def ask(body: AskRequest, tenant_id: TenantId, settings: SettingsDep,
        embedder: EmbedderDep, chat: ChatDep):
    return qa.ask(tenant_id=tenant_id, question=body.question, document_ids=body.document_ids,
                  top_k=body.top_k, method=body.chunking_method,
                  settings=settings, embedder=embedder, chat=chat)


@router.get("/query-logs", response_model=list[QueryLogOut], tags=["debug"],
            summary="Recent questions with retrieved chunk ids, scores and answers",
            responses={400: _ERRORS[400]})
def query_logs(tenant_id: TenantId, limit: Annotated[int, Query(ge=1, le=200)] = 20):
    with get_pool().connection() as conn:
        return repo.list_query_logs(conn, tenant_id, limit)


def _model_available(name: str, installed: set[str]) -> bool:
    return name in installed or (":" not in name and f"{name}:latest" in installed)


@router.get("/health", response_model=HealthOut, tags=["debug"],
            summary="Check database, Ollama and configured models")
def health(settings: SettingsDep):
    try:
        with get_pool().connection() as conn:
            conn.execute("SELECT 1")
        database = "ok"
    except Exception as exc:  # noqa: BLE001
        database = f"error: {exc}"

    embedding_ok = chat_ok = None
    try:
        listed = ollama.Client(host=settings.ollama_host, timeout=5).list()
        installed = set()
        for model in listed["models"]:
            name = getattr(model, "model", None) or (model.get("name") if isinstance(model, dict) else None)
            if name:
                installed.add(name)
        embedding_ok = _model_available(settings.embedding_model, installed)
        chat_ok = _model_available(settings.chat_model, installed)
        ollama_status = "ok"
    except Exception as exc:  # noqa: BLE001
        ollama_status = f"error: {exc}"

    return {
        "database": database,
        "ollama": ollama_status,
        "embedding_model": settings.embedding_model,
        "embedding_model_available": embedding_ok,
        "embedding_dim": settings.embedding_dim,
        "chat_model": settings.chat_model,
        "chat_model_available": chat_ok,
        "min_score": settings.min_score,
    }
