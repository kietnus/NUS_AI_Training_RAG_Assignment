"""SQL queries. Every query is scoped by tenant_id."""

from datetime import date
from uuid import UUID

import numpy as np
import psycopg
from psycopg.types.json import Jsonb

_DOCUMENT_COLUMNS = """
    d.id, d.tenant_id, d.title, d.version, d.valid_from, d.valid_to, d.status,
    d.superseded_by, d.file_name, d.file_type, d.file_size, d.char_count,
    d.ingestion_stats, d.created_at,
    CASE
        WHEN d.status = 'expired' AND d.superseded_by IS NOT NULL THEN 'superseded'
        WHEN d.status = 'expired' THEN 'expired'
        WHEN %(today)s < d.valid_from THEN 'not_yet_valid'
        WHEN %(today)s > d.valid_to THEN 'expired'
        ELSE 'current'
    END AS effective_status,
    COALESCE(cc.semantic_chunks, 0) AS semantic_chunks,
    COALESCE(cc.structure_chunks, 0) AS structure_chunks
"""

_DOCUMENT_FROM = """
    FROM documents d
    LEFT JOIN LATERAL (
        SELECT count(*) FILTER (WHERE c.chunking_method = 'semantic')  AS semantic_chunks,
               count(*) FILTER (WHERE c.chunking_method = 'structure') AS structure_chunks
        FROM chunks c WHERE c.document_id = d.id
    ) cc ON true
"""


# --------------------------------------------------------------------------- documents


def list_documents(conn: psycopg.Connection, tenant_id: str) -> list[dict]:
    return conn.execute(
        f"SELECT {_DOCUMENT_COLUMNS} {_DOCUMENT_FROM} "
        "WHERE d.tenant_id = %(tenant)s ORDER BY lower(d.title), d.valid_from DESC, d.created_at DESC",
        {"tenant": tenant_id, "today": date.today()},
    ).fetchall()


def get_document(conn: psycopg.Connection, tenant_id: str, document_id: UUID) -> dict | None:
    return conn.execute(
        f"SELECT {_DOCUMENT_COLUMNS} {_DOCUMENT_FROM} WHERE d.tenant_id = %(tenant)s AND d.id = %(id)s",
        {"tenant": tenant_id, "id": document_id, "today": date.today()},
    ).fetchone()


def find_versions(conn: psycopg.Connection, tenant_id: str, title: str) -> list[dict]:
    return conn.execute(
        "SELECT id, version, valid_from, valid_to, status FROM documents "
        "WHERE tenant_id = %s AND lower(title) = lower(%s)",
        (tenant_id, title),
    ).fetchall()


def lock_document_title(conn: psycopg.Connection, tenant_id: str, title: str) -> None:
    """Serialise concurrent uploads of the same document (inside the current transaction)."""
    conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"{tenant_id}\x1f{title.lower()}",))


def insert_document(conn: psycopg.Connection, values: dict) -> UUID:
    row = conn.execute(
        """
        INSERT INTO documents (tenant_id, title, version, valid_from, valid_to, status,
                               superseded_by, file_name, file_type, file_size,
                               content_sha256, char_count, ingestion_stats)
        VALUES (%(tenant_id)s, %(title)s, %(version)s, %(valid_from)s, %(valid_to)s, %(status)s,
                %(superseded_by)s, %(file_name)s, %(file_type)s, %(file_size)s,
                %(content_sha256)s, %(char_count)s, %(ingestion_stats)s)
        RETURNING id
        """,
        {**values, "ingestion_stats": Jsonb(values["ingestion_stats"])},
    ).fetchone()
    return row["id"]


def expire_older_versions(conn: psycopg.Connection, tenant_id: str, title: str,
                          new_id: UUID, new_valid_from: date) -> list[dict]:
    return conn.execute(
        """
        UPDATE documents SET status = 'expired', superseded_by = %(new_id)s
        WHERE tenant_id = %(tenant)s AND lower(title) = lower(%(title)s)
          AND id <> %(new_id)s AND status = 'active' AND valid_from <= %(valid_from)s
        RETURNING id, version
        """,
        {"tenant": tenant_id, "title": title, "new_id": new_id, "valid_from": new_valid_from},
    ).fetchall()


def insert_chunks(conn: psycopg.Connection, document_id: UUID, method: str,
                  chunks: list, vectors: np.ndarray) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (document_id, chunking_method, chunk_index, content,
                                char_count, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (document_id, method, c.index, c.content, len(c.content), Jsonb(c.metadata), vec)
                for c, vec in zip(chunks, vectors)
            ],
        )


def delete_document(conn: psycopg.Connection, tenant_id: str, document_id: UUID) -> dict | None:
    return conn.execute(
        """
        WITH counted AS (SELECT count(*) AS n FROM chunks WHERE document_id = %(id)s)
        DELETE FROM documents d USING counted
        WHERE d.tenant_id = %(tenant)s AND d.id = %(id)s
        RETURNING d.id, d.title, d.version, counted.n AS chunks_deleted
        """,
        {"tenant": tenant_id, "id": document_id},
    ).fetchone()


def list_chunks(conn: psycopg.Connection, document_id: UUID, method: str | None) -> list[dict]:
    return conn.execute(
        """
        SELECT id, document_id, chunking_method, chunk_index, content, char_count, metadata,
               embedding, created_at
        FROM chunks
        WHERE document_id = %(id)s AND (%(method)s::text IS NULL OR chunking_method = %(method)s)
        ORDER BY chunking_method, chunk_index
        """,
        {"id": document_id, "method": method},
    ).fetchall()


# --------------------------------------------------------------------------- retrieval


def search_chunks(conn: psycopg.Connection, *, tenant_id: str, document_ids: list[UUID],
                  query_vector: np.ndarray, top_k: int, method: str) -> list[dict]:
    """Semantic search. ALL filters are inside the query, before ORDER BY ... LIMIT k:
    tenant, selected documents, document still valid (active + inside validity window),
    and chunking method. Score = cosine similarity = 1 - cosine distance (<=>)."""
    return conn.execute(
        """
        SELECT c.id AS chunk_id, c.document_id, c.chunk_index, c.chunking_method, c.content,
               d.title AS document_title, d.version, d.valid_from, d.valid_to,
               1 - (c.embedding <=> %(q)s) AS score
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.tenant_id = %(tenant)s
          AND d.id = ANY(%(doc_ids)s)
          AND d.status = 'active'
          AND d.valid_from <= %(today)s AND d.valid_to >= %(today)s
          AND c.chunking_method = %(method)s
        ORDER BY c.embedding <=> %(q)s
        LIMIT %(k)s
        """,
        {
            "q": query_vector,
            "tenant": tenant_id,
            "doc_ids": list(document_ids),
            "today": date.today(),
            "method": method,
            "k": top_k,
        },
    ).fetchall()


# --------------------------------------------------------------------------- query logs


def insert_query_log(conn: psycopg.Connection, values: dict) -> UUID:
    row = conn.execute(
        """
        INSERT INTO query_logs (tenant_id, endpoint, question, document_ids, chunking_method,
                                top_k, retrieved_chunk_ids, retrieved_scores, status, reason,
                                answer, citations, llm_raw_output, embedding_model, chat_model,
                                latency_ms)
        VALUES (%(tenant_id)s, %(endpoint)s, %(question)s, %(document_ids)s, %(chunking_method)s,
                %(top_k)s, %(retrieved_chunk_ids)s, %(retrieved_scores)s, %(status)s, %(reason)s,
                %(answer)s, %(citations)s, %(llm_raw_output)s, %(embedding_model)s, %(chat_model)s,
                %(latency_ms)s)
        RETURNING id
        """,
        {
            "status": None, "reason": None, "answer": None, "citations": None,
            "llm_raw_output": None, "chat_model": None,
            **values,
            "citations": Jsonb(values["citations"]) if values.get("citations") is not None else None,
        },
    ).fetchone()
    return row["id"]


def list_query_logs(conn: psycopg.Connection, tenant_id: str, limit: int) -> list[dict]:
    return conn.execute(
        "SELECT * FROM query_logs WHERE tenant_id = %s ORDER BY created_at DESC LIMIT %s",
        (tenant_id, limit),
    ).fetchall()
