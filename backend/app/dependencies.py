"""
Dependency injection for FastAPI endpoints.
"""

from typing import Annotated
from fastapi import Depends, HTTPException, status

from app.services.discovery_service import DiscoveryService
from app.services.world_model_service import WorldModelService
from app.services.file_service import FileService


# Singleton instances
_discovery_service: DiscoveryService | None = None


def get_discovery_service() -> DiscoveryService:
    """Get discovery service singleton instance."""
    global _discovery_service
    if _discovery_service is None:
        _discovery_service = DiscoveryService()
    return _discovery_service


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
