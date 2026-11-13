"""Chat model client (Ollama) with JSON-schema constrained output."""

import json
import logging
import re

import httpx
import ollama

from ..config import Settings
from ..errors import ModelServiceError

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class ChatService:
    def __init__(self, settings: Settings):
        self.model = settings.chat_model
        self.host = settings.ollama_host
        self.temperature = settings.llm_temperature
        self.num_ctx = settings.llm_num_ctx
        self._client = ollama.Client(host=settings.ollama_host, timeout=settings.request_timeout_s)

    def chat_json(self, system: str, user: str, schema: dict) -> tuple[dict | None, str]:
        """Return (parsed JSON or None, raw model output)."""
        try:
            response = self._client.chat(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                format=schema,
                options={"temperature": self.temperature, "num_ctx": self.num_ctx},
            )
        except ollama.ResponseError as exc:
            hint = f" Run `ollama pull {self.model}`." if exc.status_code == 404 else ""
            raise ModelServiceError(f"Chat model '{self.model}' failed: {exc.error}.{hint}") from exc
        except (httpx.HTTPError, ConnectionError) as exc:
            raise ModelServiceError(
                f"Cannot reach Ollama at {self.host} for chat ({exc}). Is `ollama serve` running?"
            ) from exc

        raw = response["message"]["content"] or ""
        return _parse_json(raw), raw


def _parse_json(raw: str) -> dict | None:
    text = _THINK_RE.sub("", raw).strip()
    for candidate in (text, *_JSON_OBJECT_RE.findall(text)):
        try:
            value = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    logger.warning("LLM output is not valid JSON: %.200s", raw)
    return None
