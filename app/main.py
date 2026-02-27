from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import ingest, search, query

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(
    title="LlamaIndex Document API",
    description="Document ingestion and semantic search / RAG API.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router)
app.include_router(search.router)
app.include_router(query.router)


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
