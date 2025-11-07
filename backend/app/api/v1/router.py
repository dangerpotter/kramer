"""
Main API router that includes all v1 endpoints.
"""

from fastapi import APIRouter

from app.api.v1 import discovery, world_model, datasets, reports, websocket

api_router = APIRouter()

# Include all sub-routers
api_router.include_router(discovery.router, prefix="/discovery", tags=["discovery"])
api_router.include_router(world_model.router, prefix="/world-model", tags=["world-model"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(reports.router, prefix="/reports", tags=["reports"])
api_router.include_router(websocket.router, tags=["websocket"])
