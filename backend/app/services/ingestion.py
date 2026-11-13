"""Ingestion pipeline: validate metadata -> parse -> chunk -> embed -> index.

Order matters: cheap checks (metadata, file type, size, duplicate version) run first,
so a document that will be rejected is never parsed, embedded or indexed.
"""

import hashlib
import logging
import time
from dataclasses import dataclass
from datetime import date

import psycopg

from .. import repository as repo
from ..config import Settings
from ..db import get_pool
from ..errors import DuplicateError, FileTooLargeError, MetadataError, ParseError
from .chunking import semantic_chunk, structure_chunk
from .embeddings import EmbeddingService
from .parsing import detect_file_type, parse_document

logger = logging.getLogger(__name__)


@dataclass
class DocumentMetadata:
    title: str
    version: str
    valid_from: date
    valid_to: date


def _parse_date(name: str, value: str) -> date:
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise MetadataError(f"'{name}' must be a date in YYYY-MM-DD format, got '{value}'. "
                            "The document was not indexed.") from exc


def validate_metadata(title: str | None, version: str | None,
                      valid_from: str | None, valid_to: str | None) -> DocumentMetadata:
    fields = {"title": title, "version": version, "valid_from": valid_from, "valid_to": valid_to}
    missing = [name for name, value in fields.items() if value is None or not str(value).strip()]
    if missing:
        raise MetadataError(
            f"Missing required metadata: {', '.join(missing)}. The document was not indexed."
        )
    title, version = title.strip(), version.strip()
    if len(title) > 200:
        raise MetadataError("'title' must be at most 200 characters.")
    if len(version) > 50:
        raise MetadataError("'version' must be at most 50 characters.")
    start = _parse_date("valid_from", valid_from)
    end = _parse_date("valid_to", valid_to)
    if end < start:
        raise MetadataError(f"'valid_to' ({end}) must be on or after 'valid_from' ({start}). "
                            "The document was not indexed.")
    return DocumentMetadata(title, version, start, end)


def _check_duplicate_version(versions: list[dict], meta: DocumentMetadata, tenant_id: str) -> None:
    for existing in versions:
        if existing["version"].lower() == meta.version.lower():
            raise DuplicateError(
                f"Version '{meta.version}' of '{meta.title}' already exists for tenant "
                f"'{tenant_id}' (document {existing['id']}). Delete it first or use a new version label."
            )


def ingest_document(*, tenant_id: str, filename: str, data: bytes, meta: DocumentMetadata,
                    settings: Settings, embedder: EmbeddingService) -> dict:
    started = time.perf_counter()

    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise FileTooLargeError(f"File is larger than {settings.max_upload_mb} MB.")
    if not data:
        raise ParseError("The uploaded file is empty.")
    file_type = detect_file_type(filename, data)

    with get_pool().connection() as conn:  # fail fast before the expensive steps
        _check_duplicate_version(repo.find_versions(conn, tenant_id, meta.title), meta, tenant_id)

    # 1. Parse
    parsed = parse_document(file_type, data)

    # 2. Chunk (both strategies)
    semantic = semantic_chunk(
        parsed.blocks,
        embedder.embed_documents,
        threshold_type=settings.semantic_threshold_type,
        breakpoint_percentile=settings.semantic_breakpoint_percentile,
        similarity_threshold=settings.semantic_similarity_threshold,
        buffer_size=settings.semantic_buffer_size,
        max_chars=settings.max_chunk_chars,
        min_chars=settings.min_chunk_chars,
    )
    structure = structure_chunk(
        parsed.blocks, max_chars=settings.structure_max_chunk_chars, min_chars=settings.min_chunk_chars
    )
    if not semantic.chunks or not structure:
        raise ParseError("The document produced no chunks after parsing.")

    # 3. Embed each chunk (same model that will embed the questions)
    semantic_vectors = embedder.embed_documents([c.content for c in semantic.chunks])
    structure_vectors = embedder.embed_documents([c.content for c in structure])

    stats = {
        "parse": {"blocks": len(parsed.blocks), "chars": parsed.char_count, "pages": parsed.page_count,
                  "block_kinds": {k: sum(1 for b in parsed.blocks if b.kind == k)
                                  for k in ("heading", "paragraph", "list_item", "table_row")}},
        "semantic": semantic.stats,
        "structure": {"chunks": len(structure), "max_chars": settings.structure_max_chunk_chars},
        "embedding_model": embedder.model,
        "embedding_dim": embedder.dim,
    }

    # 4. Index: document row + chunks + versioning, in one transaction
    try:
        with get_pool().connection() as conn:
            repo.lock_document_title(conn, tenant_id, meta.title)
            versions = repo.find_versions(conn, tenant_id, meta.title)
            _check_duplicate_version(versions, meta, tenant_id)

            # If an active version with a later valid_from already exists, the uploaded
            # file is an OLD version: store it as expired. Otherwise it becomes current.
            newer = [v for v in versions if v["status"] == "active" and v["valid_from"] > meta.valid_from]
            newest = max(newer, key=lambda v: v["valid_from"]) if newer else None

            document_id = repo.insert_document(conn, {
                "tenant_id": tenant_id,
                "title": meta.title,
                "version": meta.version,
                "valid_from": meta.valid_from,
                "valid_to": meta.valid_to,
                "status": "expired" if newest else "active",
                "superseded_by": newest["id"] if newest else None,
                "file_name": filename,
                "file_type": file_type,
                "file_size": len(data),
                "content_sha256": hashlib.sha256(data).hexdigest(),
                "char_count": parsed.char_count,
                "ingestion_stats": stats,
            })
            expired = [] if newest else repo.expire_older_versions(
                conn, tenant_id, meta.title, document_id, meta.valid_from)

            repo.insert_chunks(conn, document_id, "semantic", semantic.chunks, semantic_vectors)
            repo.insert_chunks(conn, document_id, "structure", structure, structure_vectors)
            # The context manager commits here.
    except psycopg.errors.UniqueViolation as exc:
        raise DuplicateError(f"Version '{meta.version}' of '{meta.title}' already exists.") from exc

    with get_pool().connection() as conn:
        document = repo.get_document(conn, tenant_id, document_id)

    logger.info("Ingested %s (%s v%s) for tenant %s: %d semantic / %d structure chunks",
                filename, meta.title, meta.version, tenant_id, len(semantic.chunks), len(structure))
    return {
        "document": document,
        "chunks_created": {"semantic": len(semantic.chunks), "structure": len(structure)},
        "expired_documents": [{"id": r["id"], "version": r["version"]} for r in expired],
        "stored_as_expired_because_newer_version": (
            {"id": newest["id"], "version": newest["version"]} if newest else None
        ),
        "ingestion_ms": int((time.perf_counter() - started) * 1000),
    }
