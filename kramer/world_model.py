"""World model for representing knowledge graph of papers, claims, and hypotheses."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
from enum import Enum
import uuid


class NodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    PAPER = "paper"
    CLAIM = "claim"
    HYPOTHESIS = "hypothesis"
    CONCEPT = "concept"


class EdgeType(str, Enum):
    """Types of relationships between nodes."""
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    CITES = "cites"
    RELATED_TO = "related_to"
    DERIVED_FROM = "derived_from"


@dataclass
class Node:
    """A node in the knowledge graph."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: NodeType = NodeType.CONCEPT
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        return self.id == other.id


@dataclass
class Edge:
    """A directed edge between nodes in the knowledge graph."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    target_id: str = ""
    type: EdgeType = EdgeType.RELATED_TO
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        if not isinstance(other, Edge):
            return False
        return self.id == other.id


@dataclass
class Finding:
    """A finding (claim/observation) extracted from a source."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    claim_text: str = ""
    source_node_id: str = ""  # ID of the paper node
    confidence: float = 0.5  # How central/important to the source
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_node(self) -> Node:
        """Convert finding to a claim node."""
        return Node(
            id=self.id,
            type=NodeType.CLAIM,
            content=self.claim_text,
            metadata={
                **self.metadata,
                "confidence": self.confidence,
                "source_node_id": self.source_node_id,
            }
        )


class WorldModel:
    """Knowledge graph managing papers, claims, hypotheses, and their relationships."""

    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, Edge] = {}
        self.findings: Dict[str, Finding] = {}

    def add_node(self, node: Node) -> Node:
        """Add a node to the graph."""
        self.nodes[node.id] = node
        return node

    def add_edge(self, edge: Edge) -> Edge:
        """Add an edge to the graph."""
        if edge.source_id not in self.nodes:
            raise ValueError(f"Source node {edge.source_id} not found")
        if edge.target_id not in self.nodes:
            raise ValueError(f"Target node {edge.target_id} not found")
        self.edges[edge.id] = edge
        return edge

    def add_finding(self, finding: Finding) -> Finding:
        """Add a finding and create corresponding claim node."""
        # Store the finding
        self.findings[finding.id] = finding

        # Create claim node
        claim_node = finding.to_node()
        self.add_node(claim_node)

        # Create edge from claim to source paper
        if finding.source_node_id in self.nodes:
            edge = Edge(
                source_id=claim_node.id,
                target_id=finding.source_node_id,
                type=EdgeType.DERIVED_FROM,
                weight=finding.confidence,
                metadata={"finding_id": finding.id}
            )
            self.add_edge(edge)

        return finding

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[Node]:
        """Get all nodes of a specific type."""
        return [node for node in self.nodes.values() if node.type == node_type]

    def get_edges_from_node(self, node_id: str) -> List[Edge]:
        """Get all edges originating from a node."""
        return [edge for edge in self.edges.values() if edge.source_id == node_id]

    def get_edges_to_node(self, node_id: str) -> List[Edge]:
        """Get all edges pointing to a node."""
        return [edge for edge in self.edges.values() if edge.target_id == node_id]

    def get_findings_for_paper(self, paper_node_id: str) -> List[Finding]:
        """Get all findings derived from a specific paper."""
        return [
            finding for finding in self.findings.values()
            if finding.source_node_id == paper_node_id
        ]

    def create_paper_node(
        self,
        title: str,
        authors: List[str],
        year: int,
        doi: str,
        abstract: str = "",
        **extra_metadata
    ) -> Node:
        """Create and add a paper node to the world model."""
        node = Node(
            type=NodeType.PAPER,
            content=title,
            metadata={
                "title": title,
                "authors": authors,
                "year": year,
                "doi": doi,
                "abstract": abstract,
                **extra_metadata
            }
        )
        return self.add_node(node)

    def __len__(self) -> int:
        """Return number of nodes in the graph."""
        return len(self.nodes)

    def summary(self) -> Dict[str, int]:
        """Get summary statistics of the world model."""
        return {
            "total_nodes": len(self.nodes),
            "total_edges": len(self.edges),
            "total_findings": len(self.findings),
            "papers": len(self.get_nodes_by_type(NodeType.PAPER)),
            "claims": len(self.get_nodes_by_type(NodeType.CLAIM)),
            "hypotheses": len(self.get_nodes_by_type(NodeType.HYPOTHESIS)),
        }
