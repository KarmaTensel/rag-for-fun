from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.auth import verify_api_key
from app.services.retrieval import semantic_search

router = APIRouter(
    prefix="/search",
    tags=["Search"],
    dependencies=[Depends(verify_api_key)],
)

class SearchRequest(BaseModel):
    query:       str = Field(..., min_length=1)
    top_k:       int = Field(7, ge=1, le=50)
    user_access: list[str] = Field(..., min_length=1)


class SearchResult(BaseModel):
    text:       str
    score:      float
    metadata:   dict
    node_id:    str
    ref_doc_id: str | None


class SearchResponse(BaseModel):
    query:   str
    top_k:   int
    results: list[SearchResult]


@router.post("/", response_model=SearchResponse, summary="Semantic search over ingested documents")
async def search(request: SearchRequest):
    try:
        results = await semantic_search(
            request.query,
            user_access=request.user_access,
            top_k=request.top_k,
        )
        return SearchResponse(query=request.query, top_k=request.top_k, results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
