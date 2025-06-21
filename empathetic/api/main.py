"""FastAPI backend for Empathetic framework."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import health, reports, testing

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown."""
    print(f"Starting Empathetic API v{settings.VERSION}")
    yield
    print("Shutting down Empathetic API")


app = FastAPI(
    title="Empathetic API",
    description="Human-Centered AI Evaluation Framework API",
    version=settings.VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(testing.router, prefix="/api/testing", tags=["testing"])
app.include_router(reports.router, prefix="/api/reports", tags=["reports"])



@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Empathetic API",
        "version": settings.VERSION,
        "description": "Human-Centered AI Evaluation Framework",
        "endpoints": {
            "health": "/api/health",
            "testing": "/api/testing",
            "reports": "/api/reports",
            "docs": "/docs",
        }
    }
