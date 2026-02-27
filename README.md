# LlamaIndex Document API

A FastAPI service that ingests documents, generates OpenAI embeddings, stores them in pgvector (manually — no LlamaIndex write access), and exposes semantic search.

## Architecture

```
                        ┌─────────────────────────────────────────┐
                        │              FastAPI App                │
                        │                                         │
                        │  POST /ingest    POST /search           │
                        └────────┬─────────────┬──────────────────┘
                                 │             │
                    ┌────────────▼──┐     ┌────▼──────────────┐
                    │IngestionService│     │ RetrievalService  │
                    │               │     │                   │
                    │ LlamaIndex    │     │ LlamaIndex        │
                    │  - parse      │     │  - embed query    │
                    │  - chunk      │     │  - cosine search  │
                    │  - embed      │     │                   │
                    └──────┬────────┘     └──────┬────────────┘
                           │                     │
                    ┌──────▼──────┐       ┌──────▼──────┐
                    │  Write DB   │       │  Read-only  │
                    │ (psycopg2)  │       │  DB conn    │
                    │  INSERT     │       │  (pgvector) │
                    └─────────────┘       └─────────────┘
                           │                     │
                    └───────────────┬─────────────┘
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
# Fill in your values
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
Upload a single document.

```bash
curl -X POST http://localhost:8000/ingest/ \
  -H "X-API-Key: api_key" \
  -F "file=file_path"
```

**Response:**
```json
{
  "status": "success",
  "message": "Document 'document.pdf' ingested successfully.",
  "details": {
    "filename": "document.pdf",
    "ref_doc_id": "document.pdf",
    "chunks": 24,
    "pages": 5
  }
}
```

### `POST /ingest/batch`
Upload multiple documents at once.

```bash
curl -X POST http://localhost:8000/ingest/batch \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.docx"
```

### `POST /search/`
Semantic search across all ingested documents.

```bash
curl -X POST http://localhost:8000/query/ \
  -H "Content-Type: application/json" \
  -H "X-API-Key: api_key" \
  -d '{"query": "who was first person to walk on the moon?", "top_k": 5}'
```

**Response:**
```json
{
  "query": "what is the refund policy?",
  "top_k": 5,
  "results": [
    {
      "text": "Customers may request a refund within 30 days...",
      "score": 0.9231,
      "metadata": { "filename": "policy.pdf", "page_label": "3" },
      "node_id": "abc-123",
      "ref_doc_id": "policy.pdf"
    }
  ]
}
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
    metadata_   JSONB,
    node_id     VARCHAR,
    ref_doc_id  VARCHAR             -- maps to original filename
);
```

## Cost Notes

- Embeddings: `text-embedding-ada-002` at **$0.0001 / 1K tokens**
- Re-uploading the same filename replaces existing embeddings (deduplication by filename)
- No LlamaIndex Cloud costs — fully local OSS library
