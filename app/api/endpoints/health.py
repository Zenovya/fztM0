from fastapi import APIRouter
from ...core.config import settings

router = APIRouter(prefix="/health", tags=["health"])

@router.get("", summary="Liveness probe")
def health():
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}