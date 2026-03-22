from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(slots=True)
class ChunkRecord:
    document_id: str
    chunk_id: str
    modality: Literal["text", "image"]
    source_name: str
    content: str


class LocalVectorStore:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = storage_path
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.records: list[ChunkRecord] = []
        self.matrix = None
        self._load()

    def _load(self) -> None:
        if not self.storage_path.exists():
            return
        payload = json.loads(self.storage_path.read_text())
        self.records = [ChunkRecord(**item) for item in payload]
        self._rebuild_matrix()

    def _persist(self) -> None:
        self.storage_path.write_text(json.dumps([asdict(record) for record in self.records], indent=2))

    def _rebuild_matrix(self) -> None:
        if not self.records:
            self.matrix = None
            return
        self.matrix = self.vectorizer.fit_transform([record.content for record in self.records])

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        self.records.extend(chunks)
        self._rebuild_matrix()
        self._persist()

    def search(self, query: str, top_k: int) -> list[tuple[ChunkRecord, float]]:
        if not self.records or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.matrix)[0]
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        return [
            (self.records[index], float(similarities[index]))
            for index in ranked_indices
            if similarities[index] > 0
        ]

    @staticmethod
    def new_chunk(
        document_id: str | None,
        modality: Literal["text", "image"],
        source_name: str,
        content: str,
    ) -> ChunkRecord:
        resolved_document_id = document_id or str(uuid4())
        return ChunkRecord(
            document_id=resolved_document_id,
            chunk_id=str(uuid4()),
            modality=modality,
            source_name=source_name,
            content=content,
        )
