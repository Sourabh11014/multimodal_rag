from fastapi import FastAPI

from multimodal_rag.api.routes import router
from multimodal_rag.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Cookiecutter-style FastAPI service for text-and-image multimodal retrieval augmented generation.",
)
app.include_router(router)
