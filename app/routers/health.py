"""Health check routes."""

from fastapi import APIRouter

router = APIRouter(tags=["Health"])

@router.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "StudyRAG"}