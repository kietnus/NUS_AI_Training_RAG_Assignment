"""NUSAskify - Internal Policy Q&A (RAG).

Run from the backend/ directory:
    uvicorn app.main:app --reload --port 8000
Then open http://localhost:8000 (web UI) and http://localhost:8000/docs (Swagger UI).
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import get_settings
from .db import close_pool, init_schema, open_pool
from .errors import AppError
from .routers import documents, qa
from .services.embeddings import EmbeddingService
from .services.llm import ChatService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("nusaskify")

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    try:
        init_schema(settings)
        open_pool(settings)
    except psycopg.OperationalError as exc:
        raise RuntimeError(
            f"Cannot connect to PostgreSQL ({exc}). Start it with `docker compose up -d` "
            "from the project root and check DATABASE_URL."
        ) from exc
    app.state.embedder = EmbeddingService(settings)
    app.state.chat = ChatService(settings)
    logger.info("NUSAskify ready: embedding=%s (%d dims), chat=%s, MIN_SCORE=%.2f",
                settings.embedding_model, settings.embedding_dim, settings.chat_model, settings.min_score)
    yield
    close_pool()


app = FastAPI(
    title="NUSAskify API",
    version="1.0.0",
    description=(
        "Internal policy Q&A with Retrieval-Augmented Generation.\n\n"
        "Every endpoint is multi-tenant: send the tenant in the **`X-Tenant-Id`** header "
        "(e.g. `company-a`). Typical flow: upload documents → list them → `/api/search` "
        "(retrieval only) or `/api/ask` (answer with citations)."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "documents", "description": "Upload (ingestion), list, inspect chunks, delete"},
        {"name": "search", "description": "Semantic search and question answering"},
        {"name": "debug", "description": "Query logs and health checks"},
    ],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AppError)
async def app_error_handler(_: Request, exc: AppError):
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message, "error": exc.code})


@app.exception_handler(RequestValidationError)
async def validation_error_handler(_: Request, exc: RequestValidationError):
    errors = exc.errors()
    parts = []
    for err in errors:
        loc = [str(p) for p in err.get("loc", []) if p not in ("body", "query", "path", "header")]
        parts.append(f"{'.'.join(loc) or 'request'}: {err.get('msg')}")
    return JSONResponse(
        status_code=422,
        content={"detail": "Invalid request - " + "; ".join(parts), "error": "validation_error",
                 "errors": jsonable_encoder(errors)},
    )


app.include_router(documents.router)
app.include_router(qa.router)

# The web UI is a static page served by the same server (mounted last so /api and /docs win).
if FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
else:  # pragma: no cover
    logger.warning("Frontend directory %s not found; only the API is served.", FRONTEND_DIR)
