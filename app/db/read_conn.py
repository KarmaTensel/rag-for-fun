"""
LlamaIndex vector store — used only for querying (semantic search).
perform_setup=False ensures LlamaIndex never creates or alters tables.
"""

from llama_index.vector_stores.postgres import PGVectorStore

from app.config import settings


def get_vector_store() -> PGVectorStore:
    async_conn = settings.database_url.replace(
        "postgresql://", "postgresql+asyncpg://"
    )
    return PGVectorStore.from_params(
        connection_string=settings.database_url,
        async_connection_string=async_conn,
        table_name=settings.embeddings_table,
        embed_dim=settings.embedding_dim,
        perform_setup=False,
    )
