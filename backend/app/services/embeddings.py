"""Embedding model client (Ollama). The SAME model embeds chunks and questions."""

import logging

import httpx
import numpy as np
import ollama

from ..config import Settings
from ..errors import ModelServiceError

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self, settings: Settings):
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self.host = settings.ollama_host
        self.query_instruction = settings.query_instruction.strip()
        self._client = ollama.Client(host=settings.ollama_host, timeout=settings.request_timeout_s)

    def _embed(self, inputs: list[str]) -> np.ndarray:
        try:
            response = self._client.embed(model=self.model, input=inputs)
        except ollama.ResponseError as exc:
            hint = f" Run `ollama pull {self.model}`." if exc.status_code == 404 else ""
            raise ModelServiceError(f"Embedding model '{self.model}' failed: {exc.error}.{hint}") from exc
        except (httpx.HTTPError, ConnectionError) as exc:
            raise ModelServiceError(
                f"Cannot reach Ollama at {self.host} for embeddings ({exc}). Is `ollama serve` running?"
            ) from exc

        vectors = np.asarray(response["embeddings"], dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(inputs):
            raise ModelServiceError(f"Unexpected embedding response shape {vectors.shape}.")
        if vectors.shape[1] != self.dim:
            raise ModelServiceError(
                f"Embedding model '{self.model}' returns {vectors.shape[1]}-dim vectors but "
                f"EMBEDDING_DIM={self.dim}. Set EMBEDDING_DIM={vectors.shape[1]} and reset the database."
            )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Embed chunk / piece texts (no instruction prefix)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        batches = [self._embed(texts[i:i + batch_size]) for i in range(0, len(texts), batch_size)]
        return np.vstack(batches)

    def format_query(self, question: str) -> str:
        """Qwen3-Embedding format: the instruction goes on the query side only."""
        if not self.query_instruction:
            return question
        return f"Instruct: {self.query_instruction}\nQuery:{question}"

    def embed_query(self, question: str) -> np.ndarray:
        return self._embed([self.format_query(question)])[0]
