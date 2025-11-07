"""
Discovery API endpoints for managing discovery sessions.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List

from app.dependencies import DiscoveryServiceDep
from app.models.discovery import (
    DiscoveryConfig,
    DiscoveryResponse,
    DiscoveryStatus,
    DiscoveryDetail,
    CycleInfo,
    MetricsResponse,
)
from app.models.responses import SuccessResponse

router = APIRouter()


@router.post("/start", response_model=DiscoveryResponse)
async def start_discovery(
    config: DiscoveryConfig,
    background_tasks: BackgroundTasks,
    service: DiscoveryServiceDep,
):
    """
    Start a new discovery session.

    Args:
        config: Discovery configuration
        background_tasks: FastAPI background tasks
        service: Discovery service

    Returns:
        Discovery response with ID and status
    """
    try:
        # Create discovery
        discovery_id = await service.create_discovery(config)

        # Start in background
        background_tasks.add_task(service.start_discovery, discovery_id)

        return DiscoveryResponse(
            discovery_id=discovery_id,
            status=DiscoveryStatus.PENDING,
            message="Discovery created and starting...",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{discovery_id}/status")
async def get_status(
    discovery_id: str,
    service: DiscoveryServiceDep,
):
    """
    Get current status of a discovery.

    Args:
        discovery_id: Discovery ID
        service: Discovery service

    Returns:
        Status information
    """
    try:
        status = await service.get_status(discovery_id)
        return status
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{discovery_id}/stop", response_model=SuccessResponse)
async def stop_discovery(
    discovery_id: str,
    service: DiscoveryServiceDep,
):
    """
    Stop a running discovery.

    Args:
        discovery_id: Discovery ID
        service: Discovery service

    Returns:
        Success response
    """
    try:
        await service.stop_discovery(discovery_id)
        return SuccessResponse(
            success=True,
            message="Discovery stopped successfully",
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{discovery_id}/cycles", response_model=List[CycleInfo])
async def get_cycles(
    discovery_id: str,
    service: DiscoveryServiceDep,
):
    """
    Get all cycles for a discovery.

    Args:
        discovery_id: Discovery ID
        service: Discovery service

    Returns:
        List of cycle information
    """
    try:
        cycles = await service.get_cycles(discovery_id)
        return cycles
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{discovery_id}/metrics", response_model=MetricsResponse)
async def get_metrics(
    discovery_id: str,
    service: DiscoveryServiceDep,
):
    """
    Get real-time metrics for a discovery.

    Args:
        discovery_id: Discovery ID
        service: Discovery service

    Returns:
        Metrics response
    """
    try:
        metrics = await service.get_metrics(discovery_id)
        return metrics
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[dict])
async def list_discoveries(
    service: DiscoveryServiceDep,
):
    """
    List all discoveries.

    Args:
        service: Discovery service

    Returns:
        List of discoveries
    """
    try:
        discoveries = await service.list_discoveries()
        return discoveries
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
