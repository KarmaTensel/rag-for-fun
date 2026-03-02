"""
Ingestion service.

File ingestion pipeline:
  1. Save uploaded bytes to a temp file
  2. Parse the file:
     - PDFs → Docling (preserves table structure as Markdown)
     - Other formats → LlamaIndex SimpleDirectoryReader
  3. Table-aware chunking (keeps Markdown tables intact)
  4. OpenAI embeds each chunk
  5. Our code inserts rows into pgvector — LlamaIndex never writes

Text ingestion pipeline (for Rails DB record summaries):
  Same as above but skips file parsing — wraps raw text in a Document directly.
"""

import os
import re
import uuid
import tempfile
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode, Document
from app.config import settings, get_embed_model
from app.db.write_conn import insert_nodes, delete_doc_nodes

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".html"}

# Max tokens for a single table chunk before splitting at row boundaries
TABLE_CHUNK_MAX_TOKENS = 2048

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

def _parse_pdf_with_docling(file_path: str) -> list[Document]:
    """Parse a PDF using Docling, preserving table structure as Markdown."""
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(file_path)
    md_text = result.document.export_to_markdown()

    if not md_text or not md_text.strip():
        return []

    return [Document(text=md_text, metadata={"source": file_path})]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


def _split_table_at_rows(table_text: str, max_tokens: int) -> list[str]:
    """Split a large Markdown table into smaller chunks at row boundaries."""
    lines = table_text.strip().split("\n")
    if len(lines) < 3:
        return [table_text]

    # First two lines are header + separator (e.g. | Col1 | Col2 |\n|---|---|)
    header_lines = lines[:2]
    header = "\n".join(header_lines)
    data_rows = lines[2:]

    chunks = []
    current_rows = []
    for row in data_rows:
        current_rows.append(row)
        candidate = header + "\n" + "\n".join(current_rows)
        if _estimate_tokens(candidate) >= max_tokens:
            # Flush current chunk (excluding the row that pushed over limit)
            if len(current_rows) > 1:
                flush = header + "\n" + "\n".join(current_rows[:-1])
                chunks.append(flush)
                current_rows = [row]
            else:
                # Single row exceeds limit — include it anyway
                chunks.append(candidate)
                current_rows = []

    if current_rows:
        chunks.append(header + "\n" + "\n".join(current_rows))

    return chunks


def _split_with_table_awareness(documents: list[Document]) -> list[TextNode]:
    """Split documents while keeping Markdown tables intact.

    Segments text into prose and table blocks. Prose is chunked with
    SentenceSplitter; tables are kept whole (or split at row boundaries
    if they exceed TABLE_CHUNK_MAX_TOKENS).
    """
    all_nodes: list[TextNode] = []

    # Regex: a block of consecutive lines starting with '|'
    table_pattern = re.compile(r"((?:^\|.+$\n?)+)", re.MULTILINE)

    for doc in documents:
        text = doc.get_content()
        metadata = doc.metadata

        # Split text into alternating prose / table segments
        segments = table_pattern.split(text)

        for segment in segments:
            segment_stripped = segment.strip()
            if not segment_stripped:
                continue

            is_table = segment_stripped.startswith("|")

            if is_table:
                # Keep table intact or split at row boundaries if too large
                if _estimate_tokens(segment_stripped) > TABLE_CHUNK_MAX_TOKENS:
                    table_chunks = _split_table_at_rows(segment_stripped, TABLE_CHUNK_MAX_TOKENS)
                else:
                    table_chunks = [segment_stripped]

                for chunk_text in table_chunks:
                    node = TextNode(
                        text=chunk_text,
                        metadata={**metadata, "content_type": "table"},
                    )
                    all_nodes.append(node)
            else:
                # Prose — use standard sentence splitter
                prose_doc = Document(text=segment_stripped, metadata=metadata)
                prose_nodes = splitter.get_nodes_from_documents([prose_doc])
                all_nodes.extend(prose_nodes)

    return all_nodes


async def ingest_file(filename: str, file_bytes: bytes) -> dict:
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, filename)
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        if ext == ".pdf":
            documents = _parse_pdf_with_docling(tmp_path)
        else:
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

    if ext == ".pdf":
        nodes = _split_with_table_awareness(documents)
    else:
        nodes = splitter.get_nodes_from_documents(documents)

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
