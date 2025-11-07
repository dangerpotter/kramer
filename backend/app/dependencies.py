"""
Dependency injection for FastAPI endpoints.
"""

from typing import Annotated
from fastapi import Depends, HTTPException, status

from app.services.discovery_service import DiscoveryService
from app.services.world_model_service import WorldModelService
from app.services.file_service import FileService


def get_discovery_service() -> DiscoveryService:
    """Get discovery service instance."""
    return DiscoveryService()


def get_world_model_service() -> WorldModelService:
    """Get world model service instance."""
    return WorldModelService()


def get_file_service() -> FileService:
    """Get file service instance."""
    return FileService()


# Type aliases for dependency injection
DiscoveryServiceDep = Annotated[DiscoveryService, Depends(get_discovery_service)]
WorldModelServiceDep = Annotated[WorldModelService, Depends(get_world_model_service)]
FileServiceDep = Annotated[FileService, Depends(get_file_service)]
