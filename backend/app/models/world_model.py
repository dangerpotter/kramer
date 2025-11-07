"""
Pydantic models for world model and graph-related endpoints.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    """A node in the world model graph."""
    node_id: str
    node_type: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    confidence: Optional[float] = None
    created_at: datetime


class GraphEdge(BaseModel):
    """An edge in the world model graph."""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class GraphData(BaseModel):
    """Complete graph data structure."""
    nodes: List[GraphNode]
    edges: List[GraphEdge]
    node_count: int
    edge_count: int


class Finding(BaseModel):
    """A scientific finding from the discovery process."""
    finding_id: str
    text: str
    confidence: float
    supporting_evidence: List[str] = Field(default_factory=list)
    source: Optional[str] = None
    cycle_discovered: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class Hypothesis(BaseModel):
    """A hypothesis generated during discovery."""
    hypothesis_id: str
    text: str
    confidence: Optional[float] = None
    status: str  # "untested", "testing", "supported", "refuted"
    supporting_findings: List[str] = Field(default_factory=list)
    refuting_findings: List[str] = Field(default_factory=list)
    test_results: Optional[Dict[str, Any]] = None
    cycle_generated: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class Paper(BaseModel):
    """A research paper in the knowledge base."""
    paper_id: str
    title: str
    authors: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    url: Optional[str] = None
    year: Optional[int] = None
    relevance_score: Optional[float] = None
    key_findings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class NodeDetail(BaseModel):
    """Detailed information about a specific node."""
    node: GraphNode
    connected_nodes: List[GraphNode]
    edges: List[GraphEdge]
    related_findings: List[Finding] = Field(default_factory=list)
    related_hypotheses: List[Hypothesis] = Field(default_factory=list)
