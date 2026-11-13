"""/api/documents endpoints: upload (ingestion), list, chunks (debug), delete."""

from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, File, Form, Query, UploadFile

from .. import repository as repo
from ..db import get_pool
from ..deps import EmbedderDep, SettingsDep, TenantId
from ..errors import NotFoundError
from ..schemas import (
    DeleteResponse,
    DocumentChunksOut,
    DocumentOut,
    ErrorResponse,
    UploadResponse,
)
from ..services.ingestion import ingest_document, validate_metadata

router = APIRouter(prefix="/api/documents", tags=["documents"])

_ERRORS = {
    400: {"model": ErrorResponse, "description": "Missing / invalid X-Tenant-Id header"},
    404: {"model": ErrorResponse, "description": "Document not found for this tenant"},
}


@router.post(
    "",
    response_model=UploadResponse,
    status_code=201,
    summary="Upload a file + metadata and run ingestion",
    description=(
        "Runs the ingestion pipeline: **validate metadata → parse (PDF/DOCX) → chunk "
        "(semantic + structure-based) → embed → index**.\n\n"
        "All metadata fields are **required**; a document without them is rejected and not indexed. "
        "Versions of the same document share the same `title` (case-insensitive) within a tenant: "
        "uploading a newer version (later `valid_from`) marks older active versions as expired."
    ),
    responses={
        400: _ERRORS[400],
        409: {"model": ErrorResponse, "description": "This version of the document already exists"},
        413: {"model": ErrorResponse, "description": "File too large"},
        415: {"model": ErrorResponse, "description": "Unsupported file type (only PDF and DOCX)"},
        422: {"model": ErrorResponse, "description": "Missing/invalid metadata or unreadable file"},
        503: {"model": ErrorResponse, "description": "Embedding model (Ollama) unavailable"},
    },
)
def upload_document(
    tenant_id: TenantId,
    settings: SettingsDep,
    embedder: EmbedderDep,
    file: Annotated[UploadFile, File(description="PDF (.pdf) or Word (.docx) file")],
    title: Annotated[str | None, Form(description="**Required.** e.g. `Annual Leave Policy`")] = None,
    version: Annotated[str | None, Form(description="**Required.** e.g. `2.0`")] = None,
    valid_from: Annotated[str | None, Form(description="**Required.** YYYY-MM-DD")] = None,
    valid_to: Annotated[str | None, Form(description="**Required.** YYYY-MM-DD")] = None,
):
    meta = validate_metadata(title, version, valid_from, valid_to)  # reject before any work
    data = file.file.read(settings.max_upload_mb * 1024 * 1024 + 1)
    return ingest_document(tenant_id=tenant_id, filename=file.filename or "", data=data,
                           meta=meta, settings=settings, embedder=embedder)


@router.get("", response_model=list[DocumentOut], summary="List the tenant's documents",
            responses={400: _ERRORS[400]})
def list_documents(tenant_id: TenantId):
    with get_pool().connection() as conn:
        return repo.list_documents(conn, tenant_id)


@router.get(
    "/{document_id}/chunks",
    response_model=DocumentChunksOut,
    summary="Show a document's chunks (debug)",
    description="Returns the document (with ingestion statistics, e.g. the semantic-chunking "
                "threshold and adjacent similarities) and its chunks with metadata explaining "
                "why each semantic chunk started (`break_reason`, `boundary_similarity`).",
    responses=_ERRORS,
)
def get_document_chunks(
    tenant_id: TenantId,
    document_id: UUID,
    method: Annotated[Literal["semantic", "structure", "all"], Query(description="Chunk set")] = "all",
):
    with get_pool().connection() as conn:
        document = repo.get_document(conn, tenant_id, document_id)
        if document is None:
            raise NotFoundError(f"Document {document_id} not found for tenant '{tenant_id}'.")
        rows = repo.list_chunks(conn, document_id, None if method == "all" else method)
    chunks = []
    for row in rows:
        embedding = row.pop("embedding")
        chunks.append({
            **row,
            "embedding_dim": len(embedding),
            "embedding_preview": [round(float(x), 5) for x in embedding[:8]],
        })
    return {"document": document, "chunks": chunks}


@router.delete("/{document_id}", response_model=DeleteResponse,
               summary="Delete a document and its chunks", responses=_ERRORS)
def delete_document(tenant_id: TenantId, document_id: UUID):
    with get_pool().connection() as conn:
        deleted = repo.delete_document(conn, tenant_id, document_id)
    if deleted is None:
        raise NotFoundError(f"Document {document_id} not found for tenant '{tenant_id}'.")
    return deleted
