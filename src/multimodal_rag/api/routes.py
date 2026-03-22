from __future__ import annotations

import json

from fastapi import APIRouter, File, Form, UploadFile

from multimodal_rag.models.schemas import HealthResponse, IngestResponse, QueryRequest, QueryResponse
from multimodal_rag.services.rag import rag_service

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/ingest", response_model=IngestResponse)
async def ingest_documents(
    text_files: list[UploadFile] = File(default=[]),
    image_files: list[UploadFile] = File(default=[]),
    image_captions_json: str | None = Form(default=None),
) -> IngestResponse:
    captions = json.loads(image_captions_json) if image_captions_json else {}
    documents_indexed, chunks_indexed, sources = await rag_service.ingest(text_files, image_files, captions)
    return IngestResponse(
        documents_indexed=documents_indexed,
        chunks_indexed=chunks_indexed,
        sources=sources,
    )


@router.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest) -> QueryResponse:
    return rag_service.answer(payload.query, payload.query_image_description, payload.top_k)
