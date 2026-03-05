from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.auth import verify_api_key
from app.services.ingestion import ingest_file, ingest_text

router = APIRouter(
    prefix="/ingest",
    tags=["Ingestion"],
    dependencies=[Depends(verify_api_key)],
)


@router.post("/", summary="Ingest a single document")
async def ingest_document(
    file: UploadFile = File(...),
    user_access: str = Form(...),
    klass: str = Form(...),
    record_id: str = Form(...),
):
    try:
        file_bytes = await file.read()
        metadata = {"klass": klass, "record_id": record_id}
        result = await ingest_file(
            file.filename, file_bytes,
            user_access=user_access.split(","),
            metadata=metadata,
        )
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
async def ingest_batch(
    files: list[UploadFile] = File(...),
    user_access: str = Form(...),
    klass: str = Form(...),
    record_id: str = Form(...),
):
    access = user_access.split(",")
    metadata = {"klass": klass, "record_id": record_id}
    results, errors = [], []
    for file in files:
        try:
            file_bytes = await file.read()
            result = await ingest_file(file.filename, file_bytes, user_access=access, metadata=metadata)
            results.append(result)
        except Exception as e:
            errors.append({"filename": file.filename, "error": str(e)})

    return JSONResponse(status_code=200, content={
        "status":    "completed",
        "succeeded": results,
        "failed":    errors,
    })


class TextIngestRequest(BaseModel):
    content:     str
    ref_doc_id:  str | None = None
    user_access: list[str] = Field(..., min_length=1)
    klass:       str
    record_id:   str
    metadata:    dict = {}


@router.post("/text", summary="Ingest raw text (e.g. DB record summaries from Rails)")
async def ingest_text_endpoint(payload: TextIngestRequest):
    try:
        ref_doc_id = payload.ref_doc_id or f"{payload.klass}_{payload.record_id}"
        meta = {"klass": payload.klass, "record_id": payload.record_id, **payload.metadata}
        result = await ingest_text(
            content=payload.content,
            ref_doc_id=ref_doc_id,
            user_access=payload.user_access,
            metadata=meta
        )
        return JSONResponse(status_code=200, content={
            "status":  "success",
            "details": result,
        })
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")
