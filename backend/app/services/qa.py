"""Retrieval (/api/search) and question answering (/api/ask).

/api/ask flow:
  tenant (header) -> embed question -> filtered top-k search (tenant, selected docs,
  still valid - inside the SQL query, before LIMIT k) -> score gate ("not enough
  information" if the best score is poor) -> LLM answers from numbered sources with
  citations -> citations validated (must point to a retrieved chunk; quote checked
  against the chunk text) -> query log.
"""

import logging
import re
import time
import unicodedata
from difflib import SequenceMatcher
from uuid import UUID

from .. import repository as repo
from ..config import Settings
from ..db import get_pool
from .embeddings import EmbeddingService
from .llm import ChatService

logger = logging.getLogger(__name__)

NOT_ENOUGH_INFO_ANSWER = (
    "Not enough information: the selected documents do not contain enough information "
    "to answer this question."
)

SYSTEM_PROMPT = """You are NUSAskify, an assistant that answers employees' questions about internal company policies.

Rules:
1. Use ONLY the numbered SOURCES given in the user message. Never use outside knowledge and never guess.
2. Every sentence of the answer must end with one or more citation markers such as [1] or [2][3] pointing to the sources that support it. Only use source numbers that exist in SOURCES.
3. For every source you cite, add an entry to "citations" with a short quote copied word-for-word from that source that supports the answer.
4. If the SOURCES do not contain the information needed to answer the question, return status "not_enough_information", an empty answer and an empty citations list. Sources that are only about a related topic are NOT enough.
5. Answer in the same language as the question. Be concise (1-4 sentences)."""

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["answered", "not_enough_information"]},
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source": {"type": "integer"}, "quote": {"type": "string"}},
                "required": ["source", "quote"],
            },
        },
    },
    "required": ["status", "answer", "citations"],
}

# Accepts [1], [1, 2], [1][2], [Source 1], [source #2] ...
_MARKER_RE = re.compile(
    r"\[\s*(?:sources?|chunks?)?\s*#?\s*(\d+(?:\s*(?:,|;|and|&)\s*#?\s*\d+)*)\s*\]", re.IGNORECASE
)


# --------------------------------------------------------------------------- retrieval


def _retrieve(*, tenant_id: str, question: str, document_ids: list[UUID], top_k: int,
              method: str, embedder: EmbeddingService) -> list[dict]:
    query_vector = embedder.embed_query(question)  # same model as the chunks
    with get_pool().connection() as conn:
        rows = repo.search_chunks(conn, tenant_id=tenant_id, document_ids=document_ids,
                                  query_vector=query_vector, top_k=top_k, method=method)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["score"] = round(float(row["score"]), 4)
    return rows


def _log(values: dict) -> UUID | None:
    try:
        with get_pool().connection() as conn:
            return repo.insert_query_log(conn, values)
    except Exception:  # noqa: BLE001 - logging must never break the answer
        logger.exception("Could not write query log")
        return None


def search(*, tenant_id: str, question: str, document_ids: list[UUID], top_k: int, method: str,
           embedder: EmbeddingService) -> dict:
    started = time.perf_counter()
    chunks = _retrieve(tenant_id=tenant_id, question=question, document_ids=document_ids,
                       top_k=top_k, method=method, embedder=embedder)
    log_id = _log({
        "tenant_id": tenant_id, "endpoint": "search", "question": question,
        "document_ids": document_ids, "chunking_method": method, "top_k": top_k,
        "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
        "retrieved_scores": [c["score"] for c in chunks],
        "embedding_model": embedder.model,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    })
    return {"question": question, "chunking_method": method, "top_k": top_k,
            "results": chunks, "query_log_id": log_id}


# --------------------------------------------------------------------------- answering


def _build_user_prompt(question: str, chunks: list[dict]) -> str:
    sources = []
    for c in chunks:
        sources.append(
            f"[{c['rank']}] Document: \"{c['document_title']}\" | version {c['version']} | "
            f"valid {c['valid_from']} to {c['valid_to']}\n{c['content']}"
        )
    return "SOURCES:\n\n" + "\n\n".join(sources) + f"\n\nQUESTION: {question}\n\nRespond with JSON only."


def _normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"(\.\.\.|…)", " ", text)
    text = re.sub(r"[\"'“”‘’`]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def quote_supported(quote: str | None, content: str) -> bool | None:
    """True if the quote really appears in the chunk (exact or >= 80% fuzzy overlap)."""
    if not quote or not quote.strip():
        return None
    q, c = _normalize_for_match(quote), _normalize_for_match(content)
    if not q:
        return None
    if q in c:
        return True
    match = SequenceMatcher(None, q, c, autojunk=False).find_longest_match(0, len(q), 0, len(c))
    return match.size / len(q) >= 0.8


def _normalize_markers(answer: str, n_sources: int) -> tuple[str, set[int], set[int]]:
    """Rewrite markers to [n] form and drop markers that point outside the retrieved chunks."""
    used: set[int] = set()
    invalid: set[int] = set()

    def replace(match: re.Match) -> str:
        numbers = [int(x) for x in re.findall(r"\d+", match.group(1))]
        valid = [x for x in numbers if 1 <= x <= n_sources]
        invalid.update(x for x in numbers if not 1 <= x <= n_sources)
        used.update(valid)
        return "".join(f"[{x}]" for x in valid)

    text = _MARKER_RE.sub(replace, answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text, used, invalid


def _not_enough(chunks: list[dict], reason: str, **extra) -> dict:
    return {"status": "not_enough_information", "answer": NOT_ENOUGH_INFO_ANSWER,
            "citations": [], "chunks": chunks, "reason": reason, **extra}


def _postprocess(data: dict | None, raw: str, chunks: list[dict]) -> dict:
    if data is None:  # model ignored the JSON format: salvage markers from plain text
        data = {"status": "answered", "answer": raw, "citations": []}

    status = str(data.get("status", "")).strip().lower()
    answer = str(data.get("answer") or "").strip()
    if status == "not_enough_information" or not answer:
        return _not_enough(chunks, "The LLM reported that the retrieved chunks do not answer the question.")

    answer, used, invalid = _normalize_markers(answer, len(chunks))

    quotes: dict[int, str] = {}
    for item in data.get("citations") or []:
        if not isinstance(item, dict):
            continue
        try:
            source = int(item.get("source"))
        except (TypeError, ValueError):
            continue
        if 1 <= source <= len(chunks):
            used.add(source)
            quotes.setdefault(source, str(item.get("quote") or "").strip())
        else:
            invalid.add(source)

    if not used:
        return _not_enough(
            chunks,
            "The LLM produced an answer without any valid citation to a retrieved chunk, "
            "so it was discarded as ungrounded.",
            dropped_citations=sorted(invalid),
        )

    citations = []
    for n in sorted(used):
        chunk = chunks[n - 1]
        quote = quotes.get(n) or None
        citations.append({
            "marker": n,
            "chunk_id": chunk["chunk_id"],
            "document_id": chunk["document_id"],
            "document_title": chunk["document_title"],
            "version": chunk["version"],
            "chunk_index": chunk["chunk_index"],
            "score": chunk["score"],
            "quote": quote,
            "quote_verified": quote_supported(quote, chunk["content"]),
        })
    return {"status": "answered", "answer": answer, "citations": citations, "chunks": chunks,
            "reason": None, "dropped_citations": sorted(invalid)}


def ask(*, tenant_id: str, question: str, document_ids: list[UUID], top_k: int, method: str,
        settings: Settings, embedder: EmbeddingService, chat: ChatService) -> dict:
    started = time.perf_counter()
    chunks = _retrieve(tenant_id=tenant_id, question=question, document_ids=document_ids,
                       top_k=top_k, method=method, embedder=embedder)
    raw_output = None
    top_score = chunks[0]["score"] if chunks else None

    if not chunks:
        result = _not_enough(chunks, "No valid chunks in scope: the selected documents are not "
                                     "available for this tenant, expired, or not yet valid.")
    elif top_score < settings.min_score:
        result = _not_enough(chunks, f"Best retrieval score {top_score:.3f} is below MIN_SCORE "
                                     f"{settings.min_score:.2f}; the LLM was not called.")
    else:
        data, raw_output = chat.chat_json(SYSTEM_PROMPT, _build_user_prompt(question, chunks), ANSWER_SCHEMA)
        result = _postprocess(data, raw_output, chunks)

    result["top_score"] = top_score
    result["min_score"] = settings.min_score
    result.setdefault("dropped_citations", [])
    result["query_log_id"] = _log({
        "tenant_id": tenant_id, "endpoint": "ask", "question": question,
        "document_ids": document_ids, "chunking_method": method, "top_k": top_k,
        "retrieved_chunk_ids": [c["chunk_id"] for c in chunks],
        "retrieved_scores": [c["score"] for c in chunks],
        "status": result["status"], "reason": result["reason"], "answer": result["answer"],
        "citations": [{**c, "chunk_id": str(c["chunk_id"]), "document_id": str(c["document_id"])}
                      for c in result["citations"]],
        "llm_raw_output": raw_output,
        "embedding_model": embedder.model,
        "chat_model": chat.model if raw_output is not None else None,
        "latency_ms": int((time.perf_counter() - started) * 1000),
    })
    return result
