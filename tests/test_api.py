import base64

from fastapi.testclient import TestClient

from multimodal_rag.main import app
from multimodal_rag.services.rag import rag_service

client = TestClient(app)

VALID_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aF9sAAAAASUVORK5CYII="
)


def setup_function() -> None:
    rag_service.store.records.clear()
    rag_service.store.matrix = None
    if rag_service.store.storage_path.exists():
        rag_service.store.storage_path.unlink()



def test_healthcheck() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}



def test_ingest_and_query_text_and_image() -> None:
    files = [
        ("text_files", ("guide.txt", b"FastAPI powers the application layer for the multimodal RAG stack.", "text/plain")),
        ("image_files", ("diagram.png", VALID_PNG, "image/png")),
    ]
    data = {"image_captions_json": '{"diagram.png": "Architecture diagram showing image and text retrieval flow"}'}

    response = client.post("/ingest", files=files, data=data)
    assert response.status_code == 200
    payload = response.json()
    assert payload["documents_indexed"] == 2
    assert payload["chunks_indexed"] >= 2

    query_response = client.post(
        "/query",
        json={
            "query": "What powers the application layer?",
            "query_image_description": "diagram of the system architecture",
            "top_k": 3,
        },
    )
    assert query_response.status_code == 200
    answer = query_response.json()
    assert "FastAPI" in answer["answer"]
    assert len(answer["retrieved_contexts"]) >= 1
