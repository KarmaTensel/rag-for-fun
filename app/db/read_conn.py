"""
Scoped vector search — direct SQL with pgvector.

Uses cosine distance (<=> operator) — the same similarity function
LlamaIndex calls internally — with metadata-based ACL filtering.
"""

import psycopg

from app.config import settings, DB_TABLE_NAME

def get_conn():
    return psycopg.connect(settings.database_url)

def scoped_vector_search(
    query_embedding: list[float],
    user_access: list[str],
    top_k: int = 7,
) -> list[dict]:
    """Cosine similarity search filtered by user_access.

    Matches the querying user's access list against the 'user_access'
    array stored in each document's metadata_ during ingestion.
    A document is visible if at least one value overlaps.
    """
    params: dict = {
        "embedding": query_embedding,
        "user_access": user_access,
        "top_k": top_k,
    }

    where_clause = "user_access && %(user_access)s"

    sql = f"""
        SELECT
            id,
            text,
            metadata_,
            node_id,
            ref_doc_id,
            user_access,
            1 - (embedding <=> %(embedding)s::vector) AS score
        FROM {DB_TABLE_NAME}
        WHERE {where_clause}
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(top_k)s
    """

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [desc[0] for desc in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

    return rows
