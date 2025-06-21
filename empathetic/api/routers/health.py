"""Health check endpoints."""
from datetime import datetime

from fastapi import APIRouter

from ..config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.VERSION,
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check for deployment."""
    checks = {
        "api": True,
        "openai_configured": bool(settings.OPENAI_API_KEY),
        "anthropic_configured": bool(settings.ANTHROPIC_API_KEY),
    }

    all_ready = all(checks.values())

    return {
        "ready": all_ready,
        "checks": checks,
        "timestamp": datetime.utcnow().isoformat(),
    }
