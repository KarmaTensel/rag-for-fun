"""
Retrieval service.
Embeds the query and performs cosine similarity search via pgvector,
filtered by user_access.

Post-retrieval pipeline:
  1. Score threshold — drops chunks below a similarity cutoff.
  2. Content-based dedup — removes near-duplicate text (normalized prefix match)
     while preserving distinct chunks from the same document.
"""

import hashlib

from app.config import get_embed_model
from app.db.read_conn import scoped_vector_search

embed_model = get_embed_model()

SIMILARITY_CUTOFF = 0.35
DEDUP_PREFIX_LENGTH = 256

def _content_hash(text: str) -> str:
    """Hash the first N chars of whitespace-normalised text."""
    normalised = " ".join(text.split())[:DEDUP_PREFIX_LENGTH]
    return hashlib.md5(normalised.encode()).hexdigest()


def _deduplicate(results: list[dict]) -> list[dict]:
    """Remove results whose leading content is identical."""
    seen: set[str] = set()
    unique = []
    for r in results:
        h = _content_hash(r["text"])
        if h not in seen:
            seen.add(h)
            unique.append(r)
    return unique


async def semantic_search(query: str, user_access: list[str], top_k: int = 7) -> list[dict]:
    query_embedding = await embed_model.aget_text_embedding(query)

    results = await scoped_vector_search(
        query_embedding=query_embedding,
        user_access=user_access,
        top_k=top_k,
    )

    # this does SimilarityPostprocessor alike mechanism
    results = [r for r in results if r["score"] >= SIMILARITY_CUTOFF]
    results = _deduplicate(results)

    return [
        {
            "text":       r["text"],
            "score":      round(r["score"], 4),
            "metadata":   r["metadata_"],
            "node_id":    r["node_id"],
            "ref_doc_id": r["ref_doc_id"],
        }
        for r in results
    ]
