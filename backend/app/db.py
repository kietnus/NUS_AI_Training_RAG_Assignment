"""Database setup: schema creation, embedding-dimension check and connection pool."""

import logging
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import Settings

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
# pgvector's HNSW index supports up to 2000 dimensions for the `vector` type.
HNSW_MAX_DIM = 2000

_pool: ConnectionPool | None = None


def init_schema(settings: Settings) -> None:
    """Create the extension/tables if needed and verify the vector dimension."""
    dim = int(settings.embedding_dim)
    sql = SCHEMA_PATH.read_text(encoding="utf-8").replace("{{EMBEDDING_DIM}}", str(dim))
    with psycopg.connect(settings.database_url, autocommit=True) as conn:
        conn.execute(sql)

        row = conn.execute(
            """
            SELECT atttypmod FROM pg_attribute
            WHERE attrelid = 'chunks'::regclass AND attname = 'embedding'
            """
        ).fetchone()
        existing_dim = row[0] if row else None
        if existing_dim not in (None, -1) and existing_dim != dim:
            raise RuntimeError(
                f"chunks.embedding is vector({existing_dim}) but EMBEDDING_DIM={dim}. "
                "The embedding model changed: reset the database "
                "(`docker compose down -v && docker compose up -d`) and re-upload documents."
            )

        if dim <= HNSW_MAX_DIM:
            conn.execute(
                "CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx "
                "ON chunks USING hnsw (embedding vector_cosine_ops)"
            )
        else:
            logger.warning("EMBEDDING_DIM=%s > %s: skipping HNSW index (exact search only).",
                           dim, HNSW_MAX_DIM)


def _configure_connection(conn: psycopg.Connection) -> None:
    register_vector(conn)
    try:
        # pgvector >= 0.8: when an HNSW index scan is combined with WHERE filters
        # (tenant, documents, validity), keep scanning until LIMIT k rows pass the
        # filter, so filtering really happens before top-k.
        conn.execute("SET hnsw.iterative_scan = strict_order")
        conn.commit()
    except psycopg.Error as exc:
        conn.rollback()
        logger.warning("Could not enable hnsw.iterative_scan (pgvector < 0.8?): %s", exc)


def open_pool(settings: Settings) -> ConnectionPool:
    global _pool
    _pool = ConnectionPool(
        conninfo=settings.database_url,
        min_size=1,
        max_size=10,
        kwargs={"row_factory": dict_row},
        configure=_configure_connection,
        open=True,
    )
    _pool.wait(timeout=30)
    return _pool


def get_pool() -> ConnectionPool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialised")
    return _pool


def close_pool() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
