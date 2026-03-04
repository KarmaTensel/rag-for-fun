"""
DB connection for ingestion — INSERT, UPDATE, DELETE on embeddings table.
LlamaIndex never uses this connection.

Note: Table and extension setup is handled by setup.sql, not here.
"""

import psycopg
from psycopg.types.json import Jsonb

from app.config import settings, DB_TABLE_NAME


def get_conn():
    return psycopg.connect(settings.database_url)


def insert_nodes(nodes: list[dict]):
    if not nodes:
        return

    sql = f"""
        INSERT INTO {DB_TABLE_NAME}
            (id, embedding, text, metadata_, node_id, ref_doc_id, user_access)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
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
            Jsonb(n["metadata_"]),
            n["node_id"],
            n["ref_doc_id"],
            n["user_access"],
        )
        for n in nodes
    ]

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.executemany(sql, rows)
        conn.commit()


def delete_doc_nodes(ref_doc_id: str):
    sql = f"DELETE FROM {DB_TABLE_NAME} WHERE ref_doc_id = %s"
    with get_conn() as conn:
        conn.execute(sql, (ref_doc_id,))
        conn.commit()
