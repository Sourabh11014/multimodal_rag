from typing import Literal

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int
    sources: list[str]


class QueryRequest(BaseModel):
    query: str = Field(min_length=3)
    query_image_description: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=10)


class RetrievedChunk(BaseModel):
    document_id: str
    chunk_id: str
    modality: Literal["text", "image"]
    score: float
    source_name: str
    content: str


class QueryResponse(BaseModel):
    answer: str
    retrieved_contexts: list[RetrievedChunk]


class HealthResponse(BaseModel):
    status: str
