"""
RAG service — retrieves relevant chunks then generates a grounded answer via GPT.
"""

from openai import AsyncOpenAI

from app.config import settings
from app.services.retrieval import semantic_search

client = AsyncOpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions strictly based on the provided context.
If the answer cannot be found in the context, say "I don't have enough information to answer this."
If the answer is found, dont include phrases or expression like "According to provided context" that indicates some references or dont mention the term context to make it sound natural.
Do not make up information or use knowledge outside of the provided context.
If the user refers to something from the conversation history, use it to understand their intent but still ground your answer in the provided context."""

async def rag_query(query: str, user_access: list[str], top_k: int = 7, chat_history: list[dict] | None = None) -> dict:
    chunks = await semantic_search(query, user_access=user_access, top_k=top_k)

    if not chunks:
        return {
            "answer":  "No relevant documents found. Please ingest documents first.",
            "sources": [],
            "usage":   {},
        }

    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("filename", chunk.get("ref_doc_id", "unknown"))
        context_parts.append(f"[{i}] (Source: {source})\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        messages.extend(chat_history)

    messages.append({
        "role": "user",
        "content": f"Context:\n\n{context}\n\n---\n\nQuestion: {query}",
    })

    response = await client.chat.completions.create(
        model=settings.gpt_model,
        messages=messages,
        temperature=0.2,
    )

    return {
        "answer":  response.choices[0].message.content,
        "sources": chunks,
        "usage": {
            "prompt_tokens":     getattr(response.usage, "prompt_tokens", None) or 0,
            "completion_tokens": getattr(response.usage, "completion_tokens", None) or 0,
            "total_tokens":      getattr(response.usage, "total_tokens", None) or 0,
        } if response.usage else {},
    }
