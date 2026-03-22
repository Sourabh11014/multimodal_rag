# multimodal_rag

A cookiecutter-style FastAPI project that delivers an end-to-end multimodal RAG workflow for **text + image** knowledge sources.

## What is included

- **FastAPI API** with ingestion and query endpoints.
- **Multimodal ingestion** for UTF-8 text files and image files.
- **Image understanding layer** based on image captions, filename metadata, and optional OCR when `pytesseract` is installed.
- **Local RAG index** using TF-IDF retrieval over chunked text and image-derived descriptions.
- **Extractive answer composer** that keeps answers grounded in retrieved evidence.
- **Cookiecutter-friendly layout** with `src/`, `tests/`, `data/`, and `cookiecutter.json`.

## Project structure

```text
multimodal_rag/
├── cookiecutter.json
├── data/
│   ├── index/
│   └── raw/
├── pyproject.toml
├── README.md
├── src/
│   └── multimodal_rag/
│       ├── api/
│       ├── core/
│       ├── models/
│       ├── services/
│       └── main.py
└── tests/
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Optional OCR support:

```bash
pip install -e .[ocr]
```

## Run the API

```bash
uvicorn multimodal_rag.main:app --reload
```

The interactive docs will be available at `http://127.0.0.1:8000/docs`.

## API workflow

### 1. Ingest text and image knowledge

Use multipart form uploads:

- `text_files`: one or more `.txt` or `.md` files.
- `image_files`: one or more image files.
- `image_captions_json`: optional JSON map from filename to caption/description.

Example:

```bash
curl -X POST http://127.0.0.1:8000/ingest \
  -F 'text_files=@examples/guide.txt' \
  -F 'image_files=@examples/diagram.png' \
  -F 'image_captions_json={"diagram.png": "System architecture diagram for the multimodal RAG pipeline"}'
```

### 2. Ask a grounded question

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "What powers the application layer?",
    "query_image_description": "diagram of the architecture",
    "top_k": 3
  }'
```

### 3. Get a response grounded in retrieved context

The service returns:

- a synthesized answer
- the retrieved text/image chunks used as evidence
- similarity scores and source file names

## Design notes

- Images become searchable through a text surrogate made from caption text, filename hints, dimensions, and OCR text when available.
- Retrieval is intentionally lightweight and local, making the project easy to run without an external vector database.
- The answer generation step is extractive, which keeps the response tied to retrieved evidence and demonstrates the RAG pattern end-to-end.

## Next improvements

- Add PDF/document parsers.
- Swap TF-IDF for embedding-based retrieval.
- Add LLM-based answer generation behind an optional provider.
- Persist image thumbnails and expose source previews.
