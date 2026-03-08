"""
Scoped vector search — async SQL with pgvector via asyncpg.

Uses cosine distance (<=> operator) — the same similarity function
LlamaIndex calls internally — with metadata-based ACL filtering.
"""

from app.config import DB_TABLE_NAME
from app.db.pool import get_pool


async def scoped_vector_search(
    query_embedding: list[float],
    user_access: list[str],
    top_k: int = 7,
) -> list[dict]:
    """Cosine similarity search filtered by user_access.

    Matches the querying user's access list against the 'user_access'
    array stored in each document's metadata_ during ingestion.
    A document is visible if at least one value overlaps.
    """
    sql = f"""
        SELECT
            id,
            text,
            metadata_,
            node_id,
            ref_doc_id,
            user_access,
            1 - (embedding <=> $1::vector) AS score
        FROM {DB_TABLE_NAME}
        WHERE user_access && $2
        ORDER BY embedding <=> $1::vector
        LIMIT $3
    """

    pool = get_pool()
    rows = await pool.fetch(sql, query_embedding, user_access, top_k)

    return [dict(row) for row in rows]
