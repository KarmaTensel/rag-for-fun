"""
Ingestion service.

File ingestion pipeline:
  1. Save uploaded bytes to a temp file
  2. LlamaIndex parses the file into Documents
  3. SentenceSplitter chunks Documents into Nodes
  4. OpenAI embeds each chunk
  5. Our code inserts rows into pgvector — LlamaIndex never writes

Text ingestion pipeline (for Rails DB record summaries):
  Same as above but skips file parsing — wraps raw text in a Document directly.
"""

import os
import uuid
import tempfile
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, Document
from app.config import settings, get_embed_model
from app.db.write_conn import insert_nodes, delete_doc_nodes

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".html"}

embed_model = get_embed_model()

splitter = SentenceSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)


def _dedup_lines(text: str) -> str:
    """Remove consecutive duplicate lines (common pypdf artifact with forms)."""
    lines = text.split("\n")
    deduped = [lines[0]] if lines else []
    for line in lines[1:]:
        if line.strip() != deduped[-1].strip():
            deduped.append(line)
    return "\n".join(deduped)


async def ingest_file(filename: str, file_bytes: bytes) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, filename)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        reader = SimpleDirectoryReader(input_files=[tmp_path])
        documents = reader.load_data()

    documents = [
        Document(text=_dedup_lines(doc.get_content()), metadata=doc.metadata)
        for doc in documents
    ]

    if not documents:
        raise ValueError("No content could be extracted from the file.")

    ref_doc_id = filename
    delete_doc_nodes(ref_doc_id)

    nodes: list[TextNode] = splitter.get_nodes_from_documents(documents)

    if not nodes:
        raise ValueError("Document produced no chunks after splitting.")

    texts = [node.get_content() for node in nodes]
    embeddings = await embed_model.aget_text_embedding_batch(texts)

    rows = []
    for node, embedding in zip(nodes, embeddings):
        rows.append({
            "id":        str(uuid.uuid4()),
            "embedding": embedding,
            "text":      node.get_content(),
            "metadata_": {**node.metadata, "filename": filename},
            "node_id":    node.node_id,
            "ref_doc_id": ref_doc_id,
        })

    insert_nodes(rows)

    return {
        "filename":   filename,
        "ref_doc_id": ref_doc_id,
        "chunks":     len(rows),
        "pages":      len(documents),
    }


async def ingest_text(content: str, ref_doc_id: str, metadata: dict = {}) -> dict:
    if not content.strip():
        raise ValueError("Content cannot be empty.")
    if not ref_doc_id.strip():
        raise ValueError("ref_doc_id cannot be empty.")

    document = Document(text=content, metadata={"ref_doc_id": ref_doc_id, **metadata})

    delete_doc_nodes(ref_doc_id)

    nodes = splitter.get_nodes_from_documents([document])

    if not nodes:
        raise ValueError("Text produced no chunks after splitting.")

    texts = [node.get_content() for node in nodes]
    embeddings = await embed_model.aget_text_embedding_batch(texts)

    rows = []
    for node, embedding in zip(nodes, embeddings):
        rows.append({
            "id":        str(uuid.uuid4()),
            "embedding": embedding,
            "text":      node.get_content(),
            "metadata_": {**node.metadata, **metadata},
            "node_id":    node.node_id,
            "ref_doc_id": ref_doc_id,
        })

    insert_nodes(rows)

    return {
        "ref_doc_id": ref_doc_id,
        "chunks":     len(rows),
    }
