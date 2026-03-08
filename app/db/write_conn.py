"""
DB operations for ingestion — INSERT, UPDATE, DELETE on embeddings table.
LlamaIndex never uses this connection.

Note: Table and extension setup is handled by setup.sql, not here.
"""

import json

from app.config import DB_TABLE_NAME
from app.db.pool import get_pool


async def insert_nodes(nodes: list[dict]):
    if not nodes:
        return

    sql = f"""
        INSERT INTO {DB_TABLE_NAME}
            (id, embedding, text, metadata_, node_id, ref_doc_id, user_access)
        VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7)
        ON CONFLICT (id) DO UPDATE SET
            embedding   = EXCLUDED.embedding,
            text        = EXCLUDED.text,
            metadata_   = EXCLUDED.metadata_,
            node_id     = EXCLUDED.node_id,
            ref_doc_id  = EXCLUDED.ref_doc_id,
            user_access = EXCLUDED.user_access
    """

    rows = [
        (
            n["id"],
            n["embedding"],
            n["text"],
            json.dumps(n["metadata_"]),
            n["node_id"],
            n["ref_doc_id"],
            n["user_access"],
        )
        for n in nodes
    ]

    pool = get_pool()
    await pool.executemany(sql, rows)


async def delete_doc_nodes(ref_doc_id: str):
    sql = f"DELETE FROM {DB_TABLE_NAME} WHERE ref_doc_id = $1"
    pool = get_pool()
    await pool.execute(sql, ref_doc_id)
