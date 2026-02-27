from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.auth import verify_api_key
from app.services.ingestion import ingest_file, ingest_text

router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/", summary="Ingest a single document")
async def ingest_document(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        result = await ingest_file(file.filename, file_bytes)
        return JSONResponse(status_code=200, content={
            "status":  "success",
            "message": f"Document '{result['filename']}' ingested successfully.",
            "details": result,
        })
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/batch", summary="Ingest multiple documents")
async def ingest_batch(files: list[UploadFile] = File(...)):
    results, errors = [], []
    for file in files:
        try:
            file_bytes = await file.read()
            result = await ingest_file(file.filename, file_bytes)
            results.append(result)
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    return JSONResponse(status_code=200, content={
        "status":    "completed",
        "succeeded": results,
        "failed":    errors,
    })

class TextIngestRequest(BaseModel):
    content:    str
    ref_doc_id: str
    metadata:   dict = {}


@router.post("/text", summary="Ingest raw text (e.g. DB record summaries from Rails)")
async def ingest_text_endpoint(payload: TextIngestRequest):
    try:
        result = await ingest_text(
            content=payload.content,
            ref_doc_id=payload.ref_doc_id,
            metadata=payload.metadata,
        )
        return JSONResponse(status_code=200, content={
            "status":  "success",
            "details": result,
        })
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
