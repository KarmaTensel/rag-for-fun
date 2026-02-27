CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS data_document_embeddings (
    id          VARCHAR PRIMARY KEY,
    -- Dimension must match EMBEDDING_DIM in .env
    -- OpenAI text-embedding-ada-002 = 1536 | Ollama nomic-embed-text = 768
    embedding   VECTOR(768),
    text        TEXT NOT NULL,
    metadata_   JSONB DEFAULT '{}'::jsonb,
    node_id     VARCHAR,
    ref_doc_id  VARCHAR
);

CREATE INDEX IF NOT EXISTS idx_data_document_embeddings_embedding
    ON data_document_embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_data_document_embeddings_ref_doc
    ON data_document_embeddings (ref_doc_id);

-- Create app user and grant only what it needs
CREATE USER app_user WITH PASSWORD 'your_password';
GRANT CONNECT ON DATABASE vertex_ai_db TO vertex_app_user;
GRANT SELECT, INSERT, UPDATE, DELETE ON data_document_embeddings TO vertex_app_user;
