from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from multimodal_rag.core.config import settings
from multimodal_rag.models.schemas import QueryResponse, RetrievedChunk
from multimodal_rag.services.chunker import chunk_text
from multimodal_rag.services.image_parser import ImageTextExtractor
from multimodal_rag.services.store import LocalVectorStore


class MultimodalRAGService:
    def __init__(self) -> None:
        self.store = LocalVectorStore(settings.index_dir / "chunks.json")
        self.image_extractor = ImageTextExtractor()

    async def ingest(
        self,
        text_files: list[UploadFile],
        image_files: list[UploadFile],
        captions: dict[str, str] | None = None,
    ) -> tuple[int, int, list[str]]:
        captions = captions or {}
        all_chunks = []
        sources: list[str] = []
        documents_indexed = 0

        for upload in text_files:
            raw_bytes = await upload.read()
            text = raw_bytes.decode("utf-8")
            document_id = str(uuid4())
            file_chunks = [
                self.store.new_chunk(document_id, "text", upload.filename or "text-upload", chunk)
                for chunk in chunk_text(text, settings.chunk_size, settings.chunk_overlap)
            ]
            if file_chunks:
                all_chunks.extend(file_chunks)
                sources.append(upload.filename or "text-upload")
                documents_indexed += 1

        for upload in image_files:
            raw_bytes = await upload.read()
            target_path = settings.raw_dir / (upload.filename or f"image-{uuid4()}.png")
            target_path.write_bytes(raw_bytes)
            extracted_text = self.image_extractor.extract(target_path, captions.get(upload.filename or ""))
            document_id = str(uuid4())
            image_chunks = [
                self.store.new_chunk(document_id, "image", upload.filename or target_path.name, chunk)
                for chunk in chunk_text(extracted_text, settings.chunk_size, settings.chunk_overlap)
            ]
            if image_chunks:
                all_chunks.extend(image_chunks)
                sources.append(upload.filename or target_path.name)
                documents_indexed += 1

        if all_chunks:
            self.store.add_chunks(all_chunks)

        return documents_indexed, len(all_chunks), sources

    def answer(self, query: str, query_image_description: str | None = None, top_k: int | None = None) -> QueryResponse:
        composed_query = query.strip()
        if query_image_description:
            composed_query = f"{composed_query}\nRelated image description: {query_image_description.strip()}"

        results = self.store.search(composed_query, top_k or settings.top_k)
        retrieved = [
            RetrievedChunk(
                document_id=record.document_id,
                chunk_id=record.chunk_id,
                modality=record.modality,
                score=round(score, 4),
                source_name=record.source_name,
                content=record.content,
            )
            for record, score in results
        ]

        if not retrieved:
            return QueryResponse(
                answer="I could not find relevant text or image context in the RAG index for that query.",
                retrieved_contexts=[],
            )

        answer = self._compose_answer(query, retrieved)
        return QueryResponse(answer=answer, retrieved_contexts=retrieved)

    def _compose_answer(self, query: str, contexts: list[RetrievedChunk]) -> str:
        query_terms = {term for term in re.findall(r"\w+", query.lower()) if len(term) > 2}
        candidate_sentences: list[tuple[int, str, str]] = []
        for context in contexts:
            sentences = re.split(r"(?<=[.!?])\s+", context.content)
            for sentence in sentences:
                lowered = sentence.lower()
                overlap = sum(term in lowered for term in query_terms)
                if sentence.strip():
                    candidate_sentences.append((overlap, sentence.strip(), context.source_name))

        top_sentences = sorted(candidate_sentences, key=lambda item: item[0], reverse=True)[:3]
        evidence = " ".join(sentence for _, sentence, _ in top_sentences).strip()
        source_list = ", ".join(dict.fromkeys(context.source_name for context in contexts))

        if not evidence:
            evidence = " ".join(context.content for context in contexts[:2])

        return (
            f"Answer grounded in retrieved context: {evidence} "
            f"Sources consulted: {source_list}."
        )


rag_service = MultimodalRAGService()
