from pydantic_settings import BaseSettings
from llama_index.core.embeddings import BaseEmbedding

# LlamaIndex's PGVectorStore hardcodes a "data_" prefix on the table name.

class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str | None = None
    database_url: str
    embeddings_table: str = "document_embeddings"
    embedding_model: str = "text-embedding-ada-002"
    embedding_dim: int = 1536
    chunk_size: int = 512
    chunk_overlap: int = 50
    gpt_model: str = "gpt-4o-mini"
    api_key: str

    class Config:
        env_file = ".env"

settings = Settings()

# LlamaIndex prep5ends "data_" to table_name internally.
# Use this for direct SQL (write_conn).
DB_TABLE_NAME = f"data_{settings.embeddings_table}"


def get_embed_model() -> BaseEmbedding:
    if settings.openai_base_url:
        from llama_index.embeddings.ollama import OllamaEmbedding

        return OllamaEmbedding(
            model_name=settings.embedding_model,
            base_url=settings.openai_base_url.removesuffix("/v1"),
        )

    from llama_index.embeddings.openai import OpenAIEmbedding

    return OpenAIEmbedding(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
