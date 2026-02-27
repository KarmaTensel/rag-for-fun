import traceback

from typing import Optional

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.auth import verify_api_key
from app.services.rag import rag_query

router = APIRouter(
    prefix="/query",
    tags=["RAG Query"],
    dependencies=[Depends(verify_api_key)],
)


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(7, ge=1, le=20)
    chat_history: Optional[list[ChatMessage]] = None


class SourceChunk(BaseModel):
    text:       str
    score:      float
    metadata:   dict
    ref_doc_id: str | None


class QueryResponse(BaseModel):
    query:   str
    answer:  str
    sources: list[SourceChunk]
    usage:   dict


@router.post("/", response_model=QueryResponse, summary="RAG — retrieve context and generate answer")
async def query(request: QueryRequest):
    try:
        history = [msg.model_dump() for msg in request.chat_history] if request.chat_history else None
        result = await rag_query(request.query, top_k=request.top_k, chat_history=history)
        return QueryResponse(
            query=request.query,
            answer=result["answer"],
            sources=result["sources"],
            usage=result["usage"],
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"RAG query failed: {str(e)}")
