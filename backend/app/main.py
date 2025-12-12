"""
FastAPI main application for Kramer Discovery Platform.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.config import settings
from app.core.database import init_db, close_db
from app.core.kramer_bridge import get_bridge

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events."""
    # Startup
    print("🚀 Starting Kramer Discovery Platform...")

    # Initialize database
    try:
        await init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️ Database initialization failed: {e}")
        print("   Running in memory-only mode")

    # Initialize bridge and load existing discoveries
    try:
        bridge = get_bridge()
        await bridge.startup()
        print("✅ Bridge initialized")
    except Exception as e:
        print(f"⚠️ Bridge initialization failed: {e}")

    yield

    # Shutdown
    print("👋 Shutting down Kramer Discovery Platform...")
    await close_db()


app = FastAPI(
    title="Kramer Discovery Platform",
    description="Autonomous Scientific Discovery System with Real-time Monitoring",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def root():
    """Root endpoint."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "name": "Kramer Discovery Platform",
    }


@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
        },
    )
