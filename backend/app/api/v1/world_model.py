"""
World model API endpoints for querying the knowledge graph.
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional

from app.dependencies import WorldModelServiceDep
from app.models.world_model import (
    GraphData,
    Finding,
    Hypothesis,
    Paper,
    NodeDetail,
)

router = APIRouter()


@router.get("/{discovery_id}/graph", response_model=GraphData)
async def get_graph(
    discovery_id: str,
    node_type: Optional[str] = Query(None, description="Filter by node type"),
    service: WorldModelServiceDep = None,
):
    """
    Get world model graph data for visualization.

    Args:
        discovery_id: Discovery ID
        node_type: Optional filter by node type
        service: World model service

    Returns:
        Graph data with nodes and edges
    """
    try:
        graph_data = await service.get_graph_data(discovery_id, node_type)
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{discovery_id}/nodes/{node_id}", response_model=NodeDetail)
async def get_node(
    discovery_id: str,
    node_id: str,
    service: WorldModelServiceDep = None,
):
    """
    Get detailed information about a specific node.

    Args:
        discovery_id: Discovery ID
        node_id: Node ID
        service: World model service

    Returns:
        Node detail
    """
    try:
        node_detail = await service.get_node(discovery_id, node_id)
        if not node_detail:
            raise HTTPException(status_code=404, detail="Node not found")
        return node_detail
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{discovery_id}/findings", response_model=List[Finding])
async def get_findings(
    discovery_id: str,
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    service: WorldModelServiceDep = None,
):
    """
    Get all findings from the world model.

    Args:
        discovery_id: Discovery ID
        min_confidence: Minimum confidence threshold
        service: World model service

    Returns:
        List of findings
    """
    try:
        findings = await service.get_findings(discovery_id, min_confidence)
        return findings
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{discovery_id}/hypotheses", response_model=List[Hypothesis])
async def get_hypotheses(
    discovery_id: str,
    tested_only: bool = Query(False, description="Only return tested hypotheses"),
    service: WorldModelServiceDep = None,
):
    """
    Get all hypotheses from the world model.

    Args:
        discovery_id: Discovery ID
        tested_only: Only return tested hypotheses
        service: World model service

    Returns:
        List of hypotheses
    """
    try:
        hypotheses = await service.get_hypotheses(discovery_id, tested_only)
        return hypotheses
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{discovery_id}/papers", response_model=List[Paper])
async def get_papers(
    discovery_id: str,
    service: WorldModelServiceDep = None,
):
    """
    Get all papers from the world model.

    Args:
        discovery_id: Discovery ID
        service: World model service

    Returns:
        List of papers
    """
    try:
        papers = await service.get_papers(discovery_id)
        return papers
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
