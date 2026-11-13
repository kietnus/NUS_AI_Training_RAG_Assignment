# NUSAskify - Internal Policy Q&A (RAG)

Upload policy documents → select documents → ask a question → get an answer **with the source chunks**,
or a clear **"not enough information"** when the documents do not cover the question.

- Multi-tenant (`X-Tenant-Id` header), document versioning (old versions expire), validity dates.
- Two chunking strategies stored side by side (**semantic** and **structure-based**) so they can be compared.
- Every answer is checked: citations must point to retrieved chunks, quotes are verified against the chunk text.
- Interactive API docs (Swagger UI) at `/docs`; every question is logged in `query_logs` for debugging.

---

## 1. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Backend | **Python 3.12 + FastAPI** (sync endpoints, Pydantic v2) | Swagger UI generated from the code; Pydantic validates requests |
| Database / vector store | **PostgreSQL 17 + pgvector 0.8** (Docker) | Relational tables + vectors in one place; metadata filters are plain SQL `WHERE` inside the vector query |
| DB driver | psycopg 3 + psycopg-pool + pgvector-python | Raw SQL keeps the "filter before top-k" logic explicit |
| Embedding model | **`qwen3-embedding:0.6b`** via Ollama (1024 dims) | Small, multilingual (EN + VI), local |
| Chat model | **`qwen3:4b-instruct`** via Ollama, JSON-schema output, temperature 0 | Runs on a laptop GPU; structured output simplifies citation checks |
| Parsing | pypdf (PDF), python-docx (DOCX incl. headings, lists, tables) | Assignment requirement: PDF + DOCX, others rejected |
| Chunking | **Own implementation** with numpy (semantic) + own heading/paragraph splitter (structure) | Every step of the algorithm is visible and explainable |
| Frontend | Plain HTML + CSS + JavaScript (no build step), served by FastAPI | Nothing extra to install |

## 2. Architecture and flows

```
Web UI (static, /)  ⇄  FastAPI backend (/api/*, /docs)  ⇄  PostgreSQL + pgvector
                                  ⇅
                 Ollama: qwen3-embedding:0.6b + qwen3:4b-instruct
```

### 2.1 Ingestion flow - `POST /api/documents`

```mermaid
flowchart LR
    A[Upload file + metadata<br/>X-Tenant-Id] --> B{Metadata complete?<br/>title, version,<br/>valid_from, valid_to}
    B -- no --> R1[422 - not indexed]
    B -- yes --> C{PDF or DOCX?<br/>extension + signature}
    C -- no --> R2[415 - not indexed]
    C -- yes --> D{Version already<br/>exists?}
    D -- yes --> R3[409]
    D -- no --> E[Parse → blocks<br/>heading / paragraph /<br/>list item / table row]
    E --> F1[Semantic chunking]
    E --> F2[Structure-based chunking]
    F1 --> G[Embed each chunk<br/>qwen3-embedding]
    F2 --> G
    G --> H[(One transaction:<br/>insert document + chunks,<br/>expire older versions)]
```

### 2.2 Question flow - `POST /api/ask`

```mermaid
flowchart LR
    Q[Question + document_ids<br/>X-Tenant-Id] --> E[Embed question<br/>same model + query instruction]
    E --> S["SQL: WHERE tenant AND selected docs<br/>AND status='active' AND today in validity<br/>ORDER BY cosine distance LIMIT k"]
    S --> G{Any chunk and<br/>best score ≥ MIN_SCORE?}
    G -- no --> N[not_enough_information<br/>LLM not called]
    G -- yes --> L[LLM: answer only from<br/>numbered chunks, JSON with<br/>citations + quotes]
    L --> V{At least one citation<br/>to a retrieved chunk?}
    V -- no --> N
    V -- yes --> A[answered: answer + citations<br/>quote verified against chunk]
    A --> LOG[(query_logs)]
    N --> LOG
```

`POST /api/search` is the same flow up to the SQL query (no LLM), so retrieval can be inspected on its own.

**Filter before top-k.** All filters are in the same SQL statement as `ORDER BY embedding <=> query LIMIT k`,
so the k results are the k best chunks *among the allowed ones* (not the k best overall, filtered afterwards).
With the HNSW index, `SET hnsw.iterative_scan = strict_order` (pgvector ≥ 0.8) makes the index scan continue
until k rows pass the filter.

## 3. Database schema

```mermaid
erDiagram
    documents ||--o{ chunks : "has (ON DELETE CASCADE)"
    documents |o--o| documents : "superseded_by"
    documents {
        uuid id PK
        text tenant_id "filter"
        text title "versions share (tenant, lower(title))"
        text version "unique per (tenant, title)"
        date valid_from "required"
        date valid_to "required, >= valid_from"
        text status "active | expired"
        uuid superseded_by FK
        text file_name
        text file_type "pdf | docx"
        int file_size
        text content_sha256
        int char_count
        jsonb ingestion_stats "parse + chunking statistics"
        timestamptz created_at
    }
    chunks {
        uuid id PK
        uuid document_id FK
        text chunking_method "semantic | structure"
        int chunk_index
        text content
        int char_count
        jsonb metadata "break reason, boundary similarity, heading..."
        vector_1024 embedding "dimension = EMBEDDING_DIM"
        timestamptz created_at
    }
    query_logs {
        uuid id PK
        text tenant_id
        text endpoint "search | ask"
        text question
        uuid_array document_ids
        text chunking_method
        int top_k
        uuid_array retrieved_chunk_ids
        float_array retrieved_scores
        text status "answered | not_enough_information"
        text reason
        text answer
        jsonb citations
        text llm_raw_output
        text embedding_model
        text chat_model
        int latency_ms
        timestamptz created_at
    }
```

Keys and indexes:

| Object | Purpose |
|---|---|
| `documents_tenant_title_version_uq` UNIQUE `(tenant_id, lower(title), lower(version))` | the same version cannot be uploaded twice |
| `documents_tenant_status_validity_idx` `(tenant_id, status, valid_from, valid_to)` | retrieval filter and document list |
| `chunks_document_method_index_uq` UNIQUE `(document_id, chunking_method, chunk_index)` | chunk order; also the index for "chunks of a document" |
| `chunks_embedding_hnsw_idx` HNSW `(embedding vector_cosine_ops)` | approximate nearest-neighbour search (cosine) |
| `query_logs_tenant_created_idx` `(tenant_id, created_at DESC)` | recent questions per tenant |

The vector column is created as `vector(EMBEDDING_DIM)`. At startup the app checks the existing column against
`EMBEDDING_DIM` and refuses to start if they differ (e.g. after switching embedding model).

## 4. Ingestion details

**Parse.** PDF text is extracted page by page; wrapped lines are re-joined into paragraphs and headings are
detected heuristically (short, no final period, Title Case / numbered). DOCX is read in document order with
heading styles, list styles and tables (each table row becomes `Header: value; Header: value`). Other types
(and `.pdf`/`.docx` files whose content does not match) are rejected with **415** and a clear message. Files
without extractable text (scanned PDFs) are rejected with **422**.

**Semantic chunking** (`backend/app/services/chunking.py`, following the assignment's algorithm):

1. Split the text into small pieces: sentences of paragraphs, list items, table rows, headings.
2. Embed each piece. With `SEMANTIC_BUFFER_SIZE=1` a piece is embedded together with its neighbours (smooths very
   short sentences; set `0` for the literal algorithm).
3. Compute the cosine similarity between each piece and the next one.
4. Walk the pieces in order: similarity high → same chunk; similarity below the threshold (and chunk ≥ `MIN_CHUNK_CHARS`)
   or chunk would exceed `MAX_CHUNK_CHARS` → new chunk. The threshold is the 25th percentile of the document's adjacent
   similarities by default (`SEMANTIC_THRESHOLD_TYPE=absolute` uses a fixed value). A heading is never left at the end of
   a chunk, and a tiny last chunk is merged into the previous one.

Each chunk stores *why* it started (`break_reason`, `boundary_similarity`, `threshold`), visible in
`GET /api/documents/{id}/chunks` and in the UI's **Chunks** dialog.

**Structure-based chunking.** Every heading starts a section; a section is packed paragraph by paragraph up to
`STRUCTURE_MAX_CHUNK_CHARS` and never crosses a heading; tiny sections (e.g. the title) are merged forward.

**Embed + index.** Each chunk is embedded with the same model used for questions (questions additionally get the
Qwen3 query instruction). Document row, both chunk sets and version updates are written in one transaction.

**Versioning.** Versions of a document share the title (case-insensitive) within a tenant. Uploading a version whose
`valid_from` is later than (or equal to) the active one marks the older one `expired` with `superseded_by`. Uploading an
*older* version than the active one stores it directly as `expired`. Retrieval only uses documents with
`status = 'active'` **and** today within `[valid_from, valid_to]`.

## 5. API

Interactive docs: **http://localhost:8000/docs** (Swagger UI) - every endpoint has an `X-Tenant-Id` field.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/documents` | Upload a file + metadata (multipart) → run ingestion |
| GET | `/api/documents` | List the tenant's documents (with `effective_status` and chunk counts) |
| GET | `/api/documents/{id}/chunks?method=semantic\|structure\|all` | Show a document's chunks and chunking statistics (debug) |
| DELETE | `/api/documents/{id}` | Delete a document and its chunks |
| POST | `/api/search` | `{question, document_ids, top_k, chunking_method}` → chunks with scores (no LLM) |
| POST | `/api/ask` | `{question, document_ids, top_k, chunking_method}` → `{status, answer, citations, chunks, reason, …}` |
| GET | `/api/query-logs?limit=20` | Recent questions with retrieved chunk ids, scores, answers, raw LLM output |
| GET | `/api/health` | Database / Ollama / model availability |

Example:

```bash
curl -s -X POST localhost:8000/api/ask -H 'X-Tenant-Id: company-a' -H 'Content-Type: application/json' \
  -d '{"question":"How many days per week can I work from home?","document_ids":["<uuid>"],"top_k":5}'
```

Errors are JSON `{"detail": "...", "error": "<code>"}`: 400 missing tenant, 404 not found, 409 duplicate version,
413 too large, 415 unsupported type, 422 missing/invalid metadata or unreadable file, 503 Ollama unavailable.

## 6. Web UI

http://localhost:8000 - tenant selector (sent as `X-Tenant-Id`), upload form, document list with checkboxes
(search scope, status badges, chunk viewer, delete), question box with top-k and chunking-method selectors,
**Ask** / **Search only** buttons, answer panel with clickable citation markers and quote verification, and the list of
retrieved chunks (document, version, chunk, score, content; cited chunks highlighted).

## 7. Sample documents

`scripts/generate_sample_docs.py` creates 6 files in `sample_docs/` (+ `manifest.json` with their metadata):

| File | Tenant | Title / version | Validity | Role in tests |
|---|---|---|---|---|
| `remote_work_policy_v2.pdf` | company-a | Remote Work Policy 2.0 | 2025-01-01 → 2027-12-31 | says "remote work", never "work from home" (TC1/TC2) |
| `annual_leave_policy_v1.docx` | company-a | Annual Leave Policy 1.0 (14 days) | 2022-01-01 → 2026-12-31 | old version; dates still valid, so only versioning excludes it (TC4) |
| `annual_leave_policy_v2.pdf` | company-a | Annual Leave Policy 2.0 (18 days) | 2025-01-01 → 2027-12-31 | current version (TC4) |
| `business_travel_expense_policy.docx` | company-a | Business Travel and Expense Policy 1.0 | 2024-07-01 → 2027-12-31 | table parsing; "remote project sites" + "site allowance" trap (TC6/TC8) |
| `information_security_policy.pdf` | company-a | Information Security Policy 1.0 | 2024-01-01 → 2027-12-31 | overlaps with remote work (VPN, lost devices) |
| `company_b_remote_work_policy.docx` | **company-b** | Remote Work Policy 1.0 (4 days) | 2025-01-01 → 2027-12-31 | other tenant (TC5) |
| `unsupported_example.txt` | - | - | - | must be rejected |

No document covers dress code or maternity leave (TC3).

## 8. Semantic vs structure-based chunking

Method: `python scripts/compare_chunking.py` runs 15 questions with a known answer (document + phrase) through
`/api/search` for both chunk sets and reports Hit@1, Hit@3, MRR@5, scores and chunk statistics →
`reports/chunking_comparison.md`.

How the two strategies differ:
- Policy documents already have one topic per heading, so structure-based chunks often match topics well; semantic
  chunking can merge two short related sections (e.g. *Requesting Remote Days* + *Working Hours*) or split a section
  whose sentences use different vocabulary.
- Bigger chunks "dilute" the embedding: a chunk mixing several topics has a lower score for each of them.
- Semantic chunking costs one extra embedding call per piece at ingestion time.
- Semantic chunking adapts to documents without headings (e.g. PDFs where heading detection fails); structure-based
  chunking depends on headings being detected.

## 9. Test results TC1-TC8

Run `python scripts/run_test_cases.py` (and `--method structure`) → `reports/test_results_<method>.md` with the
top chunks, scores, answers, citations and automatic checks (PASS / FAIL, or REVIEW where a human judgement is
needed).

**Diagnosis rule:** first look at the retrieved chunks, then at the answer.
Right chunk not in top-k → **retrieval** error (chunking, embedding, filter, top-k). Right chunk retrieved but
wrong/unsupported answer or citation → **generation** error (prompt, model). `query_logs` keeps both
`retrieved_chunk_ids` and `llm_raw_output` for this.

| # | Scenario | Expected |
|---|---|---|
| TC1 | "How many days per week can I work from home?" (doc says "remote work") | Remote Work chunk retrieved; answer "2 days" with citation |
| TC2 | `/api/search` with the TC1 question | top 3 with scores + explanation |
| TC3 | "What is the dress code for client meetings?" / "How many weeks of maternity leave can I take?" | not enough information (stopped by the score gate or by the LLM) |
| TC4 | Annual leave question with v1.0 **and** v2.0 selected | only v2.0 used (18 days) |
| TC5 | Same question as company-b (request also contains a company-a document id) | no company-a chunks; answer 4 days |
| TC6 | Only the Travel policy selected, annual leave question | no Annual Leave chunks; not enough information |
| TC7 | Check every citation of TC1/TC4/TC5/hotel question | each points to a retrieved chunk that supports the claim |
| TC8 | "Is there an allowance for working remotely?" (and 2 other candidates) | a case where the most similar chunk is wrong |

**TC2 - why semantic search finds the chunk although the words differ:**
the question and the chunk share few words ("days", "week", "work") and the key phrase differs ("work from home" vs
"work remotely"). A keyword search would rank by shared words; the embedding model instead maps both texts to vectors by
*meaning*. During training it saw "work from home", "WFH", "remote work" in the same contexts, so their vectors are
close and the cosine similarity of the question and the chunk is high, higher than for chunks that share words
but not meaning.

**TC8 - failure mode of similarity:** similarity measures "same topic / same words",
not "answers the question". Candidates designed into the sample set:
- *"Is there an allowance for working remotely?"* - the Travel policy's "remote project sites … site allowance of SGD 50
  per day" shares "remote" and "allowance" with the question, but the right answer is the Remote Work Policy's
  "internet allowance of SGD 30 per month".
- *"Which employees are not allowed to work remotely?"* - negation: embeddings largely ignore "not", so the chunk saying
  who **may** work remotely can outrank the eligibility chunk.
- *"Can I work remotely from Vietnam for a whole month?"* - the general "two days per week" chunk can outrank the
  specific "overseas: max ten working days per year" chunk.

Possible fixes: re-ranking with a cross-encoder, hybrid search (BM25 + vectors), smaller chunks, larger top-k
(the LLM may still pick the right chunk), or metadata (section headings) in the chunk text.

## 10. How to run

### Prerequisites
- Docker (with Compose) - for PostgreSQL + pgvector
- [Ollama](https://ollama.com/download) - local models
- Python 3.10+ with `venv` (Ubuntu: `sudo apt install python3-venv`)

### Steps

```bash
# 0) project root
cd nusaskify
cp .env.example .env                       # defaults work as-is

# 1) database (PostgreSQL 17 + pgvector on host port 5433)
docker compose up -d

# 2) models
ollama pull qwen3-embedding:0.6b
ollama pull qwen3:4b-instruct
# (Ollama runs as a service after install; otherwise: ollama serve)

# 3) backend
python3 -m venv backend/.venv
source backend/.venv/bin/activate
pip install -r backend/requirements.txt
cd backend && uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** (UI) and **http://localhost:8000/docs** (API docs).

```bash
# 4) sample data and reports (in a second terminal, venv active, from the project root)
python scripts/generate_sample_docs.py
python scripts/seed_sample_docs.py              # --reset to delete the sample tenants' documents first
python scripts/run_test_cases.py                # -> reports/test_results_semantic.md
python scripts/run_test_cases.py --method structure
python scripts/compare_chunking.py              # -> reports/chunking_comparison.md
```

### Configuration (`.env`)

| Variable | Default | Notes |
|---|---|---|
| `DATABASE_URL` | `postgresql://nusaskify:nusaskify@localhost:5433/nusaskify` | |
| `OLLAMA_HOST` | `http://localhost:11434` | |
| `EMBEDDING_MODEL` / `EMBEDDING_DIM` | `qwen3-embedding:0.6b` / `1024` | change both together, then reset the DB |
| `QUERY_INSTRUCTION` | Qwen3 retrieval instruction | empty for models without instructions |
| `CHAT_MODEL` | `qwen3:4b-instruct` | |
| `MIN_SCORE` | `0.40` | score gate for "not enough information" - calibrate with `/api/search` |
| `DEFAULT_TOP_K` | `5` | |
| `SEMANTIC_THRESHOLD_TYPE` | `percentile` | or `absolute` (uses `SEMANTIC_SIMILARITY_THRESHOLD`) |
| `SEMANTIC_BREAKPOINT_PERCENTILE` | `25` | lowest 25% adjacent similarities become breakpoints |
| `SEMANTIC_BUFFER_SIZE` | `1` | neighbours embedded with each piece |
| `MAX_CHUNK_CHARS` / `MIN_CHUNK_CHARS` | `1200` / `150` | |
| `STRUCTURE_MAX_CHUNK_CHARS` | `1200` | |

Changing chunking settings only affects documents uploaded afterwards (`seed_sample_docs.py --reset` re-ingests).

**Calibrating `MIN_SCORE`:** run `/api/search` for answerable questions (TC1, TC4) and unanswerable ones (TC3) and put the
threshold between the best scores of the two groups. Too high → correct answers are refused; too low → the LLM sees
irrelevant chunks (it can still refuse, which is the second safety layer).

### Troubleshooting

| Symptom | Fix |
|---|---|
| `Cannot connect to PostgreSQL` at startup | `docker compose up -d`; check port 5433 |
| `chunks.embedding is vector(N) but EMBEDDING_DIM=M` | embedding model changed: `docker compose down -v && docker compose up -d`, re-upload |
| 503 `Cannot reach Ollama` / `model not found` | start Ollama; `ollama pull <model>`; check `/api/health` |
| Upload is slow | first call loads the model into memory; semantic chunking embeds every sentence |

## 11. Known limitations

- No OCR: scanned PDFs are rejected. PDF heading detection is heuristic.
- Deleting the current version does not re-activate the previous one (re-upload it instead).
- Versions are grouped by title; renaming a document between versions creates a new document.
- The quote check verifies the quote exists in the chunk, not that it logically supports the claim (TC7 needs a human check).
- No authentication: the tenant header is trusted (simulation, as required by the assignment).

## Project layout

```
nusaskify/
├── docker-compose.yml        PostgreSQL + pgvector
├── .env.example
├── backend/
│   ├── requirements.txt
│   └── app/
│       ├── main.py           FastAPI app, error handlers, static UI
│       ├── config.py         settings (.env)
│       ├── db.py, schema.sql schema, dimension check, connection pool
│       ├── repository.py     all SQL (tenant-scoped; retrieval query)
│       ├── schemas.py        request/response models (OpenAPI)
│       ├── routers/          documents.py, qa.py
│       └── services/         parsing, chunking, embeddings, llm, ingestion, qa
├── frontend/                 index.html, app.js, styles.css
├── scripts/                  generate_sample_docs, seed_sample_docs, run_test_cases, compare_chunking
├── sample_docs/              generated sample files + manifest.json
└── reports/                  generated test / comparison reports
```
