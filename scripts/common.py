"""Shared helpers for the scripts (talk to the running NUSAskify API over HTTP)."""

import json
import os
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "sample_docs"
REPORTS_DIR = ROOT / "reports"
DEFAULT_BASE_URL = os.environ.get("NUSASKIFY_URL", "http://localhost:8000")


class Api:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 600.0):
        self.client = httpx.Client(base_url=base_url, timeout=timeout)

    def request(self, method: str, path: str, tenant: str, **kwargs) -> httpx.Response:
        headers = {"X-Tenant-Id": tenant, **kwargs.pop("headers", {})}
        return self.client.request(method, path, headers=headers, **kwargs)

    def _ok(self, response: httpx.Response) -> dict | list:
        if response.status_code >= 400:
            raise RuntimeError(f"{response.request.method} {response.request.url.path} -> "
                               f"{response.status_code}: {response.text}")
        return response.json()

    def documents(self, tenant: str) -> list[dict]:
        return self._ok(self.request("GET", "/api/documents", tenant))

    def chunks(self, tenant: str, document_id: str, method: str = "all") -> dict:
        return self._ok(self.request("GET", f"/api/documents/{document_id}/chunks", tenant,
                                     params={"method": method}))

    def delete(self, tenant: str, document_id: str) -> dict:
        return self._ok(self.request("DELETE", f"/api/documents/{document_id}", tenant))

    def upload(self, tenant: str, path: Path, metadata: dict) -> httpx.Response:
        with path.open("rb") as fh:
            return self.request("POST", "/api/documents", tenant,
                                files={"file": (path.name, fh)}, data=metadata)

    def search(self, tenant: str, question: str, document_ids: list[str],
               top_k: int = 5, method: str = "semantic") -> dict:
        return self._ok(self.request("POST", "/api/search", tenant, json={
            "question": question, "document_ids": document_ids, "top_k": top_k,
            "chunking_method": method}))

    def ask(self, tenant: str, question: str, document_ids: list[str],
            top_k: int = 5, method: str = "semantic") -> dict:
        return self._ok(self.request("POST", "/api/ask", tenant, json={
            "question": question, "document_ids": document_ids, "top_k": top_k,
            "chunking_method": method}))


def load_manifest() -> list[dict]:
    path = SAMPLE_DIR / "manifest.json"
    if not path.exists():
        raise SystemExit("sample_docs/manifest.json not found - run scripts/generate_sample_docs.py first.")
    return json.loads(path.read_text(encoding="utf-8"))


def find_doc(docs: list[dict], title: str, version: str | None = None) -> dict:
    for d in docs:
        if d["title"].lower() == title.lower() and (version is None or d["version"] == version):
            return d
    raise SystemExit(f"Document '{title}' {version or ''} not found - run scripts/seed_sample_docs.py first.")


def md_escape(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def shorten(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
