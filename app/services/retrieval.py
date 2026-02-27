"""
Retrieval service.
LlamaIndex embeds the query and performs cosine similarity search
against stored embeddings via the read-only pgvector connection.

Post-retrieval pipeline:
  1. SimilarityPostprocessor — drops chunks below a score threshold.
  2. Content-based dedup — removes near-duplicate text (normalized prefix match)
     while preserving distinct chunks from the same document.
"""

import hashlib

from llama_index.core import VectorStoreIndex
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.retrievers import VectorIndexRetriever

from app.config import get_embed_model
from app.db.read_conn import get_vector_store

embed_model = get_embed_model()

SIMILARITY_CUTOFF = 0.35
DEDUP_PREFIX_LENGTH = 256

similarity_filter = SimilarityPostprocessor(similarity_cutoff=SIMILARITY_CUTOFF)

def _content_hash(text: str) -> str:
    """Hash the first N chars of whitespace-normalised text."""
    normalised = " ".join(text.split())[:DEDUP_PREFIX_LENGTH]
    return hashlib.md5(normalised.encode()).hexdigest()


def _deduplicate(nodes):
    """Remove nodes whose leading content is identical."""
    seen: set[str] = set()
    unique = []
    for node in nodes:
        h = _content_hash(node.node.get_content())
        if h not in seen:
            seen.add(h)
            unique.append(node)
    return unique


async def semantic_search(query: str, top_k: int = 7) -> list[dict]:
    vector_store = get_vector_store()

    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=top_k,
    )

    nodes = await retriever.aretrieve(query)

    # Post-retrieval: filter low-quality results, then deduplicate
    nodes = similarity_filter.postprocess_nodes(nodes)
    nodes = _deduplicate(nodes)

    return [
        {
            "text":       node.node.get_content(),
            "score":      round(node.score or 0.0, 4),
            "metadata":   node.node.metadata,
            "node_id":    node.node.node_id,
            "ref_doc_id": node.node.ref_doc_id,
        }
        for node in nodes
    ]
