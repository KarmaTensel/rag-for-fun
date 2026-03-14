# LlamaIndex Document API

A FastAPI service that ingests documents, generates OpenAI embeddings, stores them in pgvector (manually — no LlamaIndex write access), and exposes semantic search.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              FastAPI App                │
                        │                                         │
                        │  POST /ingest    POST /search           │
                        └────────┬──────────────┬─────────────────┘
                                 │              │
                    ┌────────────▼───┐     ┌────▼──────────────┐
                    │IngestionService│     │ RetrievalService  │
                    │                │     │                   │
                    │ LlamaIndex     │     │ LlamaIndex        │
                    │  - parse       │     │  - embed query    │
                    │  - chunk       │     │  - cosine search  │
                    │  - embed       │     │                   │
                    └──────┬─────────┘     └──────┬────────────┘
                           │                      │
                    ┌──────▼──────┐        ┌──────▼──────┐
                    │  Write DB   │        │  Read-only  │
                    │ (psycopg2)  │        │  DB conn    │
                    │  INSERT     │        │  (pgvector) │
                    └─────────────┘        └─────────────┘
                           │                      │
                           └────────────┬─────────┘
                                        │
                              ┌─────────▼─────────┐
                              │   PostgreSQL DB   │
                              │   + pgvector      │
                              └───────────────────┘
```

**Key design**: LlamaIndex only gets a **read-only** DB connection. All writes go through our own `psycopg2` connection.

## Setup

### 1. PostgreSQL

Create DB user:

```sql
-- User for ingestion service and LlamaIndex query service
CREATE USER app_user WITH PASSWORD 'password';
GRANT CONNECT ON DATABASE ai_db TO app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON data_document_embeddings TO app_user;

-- Check grants on specific table
\z data_document_embeddings

-- Check for installed extensions
\dx
```

### 2. Environment

```bash
cp .env.example .env
```

### 3. Install & Run

```bash
# Setup venv
uv venv

source .venv/bin/activate

pip install -r requirements.txt

uvicorn app.main:app --reload --port 8000
```

## API Endpoints

### `POST /ingest/`
Upload a single document with access metadata.

```bash
curl -X POST http://localhost:8000/ingest/ \
  -H "X-API-Key: api_key" \
  -F "file=@report.pdf" \
  -F "user_access=company1,company2" \
  -F "klass=Company" \
  -F "record_id=42"
```

### `POST /ingest/batch`
Upload multiple documents with shared access metadata.

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -H "X-API-Key: api_key" \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.docx" \
  -F "user_access=company1" \
  -F "klass=Company" \
  -F "record_id=42"
```

### `POST /ingest/text`
Ingest raw text (e.g. DB record summaries from Rails).

```bash
curl -X POST http://localhost:8000/ingest/text \
  -H "X-API-Key: api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Customer policy text...",
    "ref_doc_id": "policy-2024",
    "user_access": ["company1", "company2"],
    "klass": "Policy",
    "record_id": "99"
  }'
```

### `POST /search/`
Scoped semantic search.

```bash
curl -X POST http://localhost:8000/search/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: api_key" \
  -d '{
    "query": "what is the refund policy?",
    "top_k": 5,
    "user_access": ["company1", "company2"]
  }'
```

### `POST /query/`
RAG — scoped search + GPT answer generation.

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: api_key" \
  -d '{
    "query": "What is the refund policy?",
    "top_k": 5,
    "user_access": ["company1"],
    "chat_history": [
      {"role": "user", "content": "Hello"},
      {"role": "assistant", "content": "Hi, how can I help?"}
    ]
  }'
```

### `GET /health`
Health check.

## Supported File Types

| Format | Extension |
|--------|-----------|
| PDF    | `.pdf`    |
| Word   | `.docx`   |
| Text   | `.txt`    |
| Markdown | `.md`   |
| CSV    | `.csv`    |
| HTML   | `.html`   |

## DB Schema

The table schema matches LlamaIndex's `PGVectorStore` format so the read-only retriever works correctly:

```sql
CREATE TABLE data_document_embeddings (
    id          VARCHAR PRIMARY KEY,
    embedding   VECTOR(1536),       -- ada-002 dimensions
    text        TEXT NOT NULL,
    metadata   JSONB,
    user_access TEXT[],
    node_id     VARCHAR,
    ref_doc_id  VARCHAR             -- maps to original filename
);
```

## Cost Notes

- Embeddings: `text-embedding-ada-002` at **$0.0001 / 1K tokens**
- Re-uploading the same filename replaces existing embeddings (deduplication by filename)
- No LlamaIndex Cloud costs — fully local OSS library
