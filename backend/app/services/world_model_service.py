"""
World model service for querying the knowledge graph.
"""

from typing import List, Optional
from datetime import datetime

from app.core.kramer_bridge import get_bridge
from app.models.world_model import (
    GraphData,
    GraphNode,
    GraphEdge,
    Finding,
    Hypothesis,
    Paper,
    NodeDetail,
)
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.world_model.graph import NodeType, EdgeType


class WorldModelService:
    """Service for querying the world model graph."""

    def __init__(self):
        """Initialize world model service."""
        self.bridge = get_bridge()

    async def get_graph_data(
        self,
        discovery_id: str,
        node_type: Optional[str] = None,
    ) -> GraphData:
        """
        Get graph data for visualization.

        Args:
            discovery_id: Discovery ID
            node_type: Optional filter by node type

        Returns:
            Graph data with nodes and edges
        """
        world_model = self.bridge.get_world_model(discovery_id)
        if not world_model:
            return GraphData(nodes=[], edges=[], node_count=0, edge_count=0)

        graph = world_model.graph

        # Get nodes
        nodes = []
        for node_id in graph.nodes():
            node_data = graph.nodes[node_id]

            # Filter by type if specified
            if node_type and node_data.get("node_type") != node_type:
                continue

            nodes.append(
                GraphNode(
                    node_id=node_id,
                    node_type=node_data.get("node_type", "unknown"),
                    text=node_data.get("text", ""),
                    metadata=node_data.get("metadata", {}),
                    confidence=node_data.get("confidence"),
                    created_at=datetime.fromisoformat(
                        node_data.get("created_at", datetime.utcnow().isoformat())
                    ),
                )
            )

        # Get edges
        edges = []
        for source, target, edge_data in graph.edges(data=True):
            # Only include edges between nodes we're showing
            node_ids = {n.node_id for n in nodes}
            if source not in node_ids or target not in node_ids:
                continue

            edges.append(
                GraphEdge(
                    edge_id=f"{source}-{target}",
                    source_id=source,
                    target_id=target,
                    edge_type=edge_data.get("edge_type", "unknown"),
                    metadata=edge_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(
                        edge_data.get("created_at", datetime.utcnow().isoformat())
                    ),
                )
            )

        return GraphData(
            nodes=nodes,
            edges=edges,
            node_count=len(nodes),
            edge_count=len(edges),
        )

    async def get_node(self, discovery_id: str, node_id: str) -> Optional[NodeDetail]:
        """
        Get detailed information about a specific node.

        Args:
            discovery_id: Discovery ID
            node_id: Node ID

        Returns:
            Node detail or None if not found
        """
        world_model = self.bridge.get_world_model(discovery_id)
        if not world_model or node_id not in world_model.graph:
            return None

        graph = world_model.graph
        node_data = graph.nodes[node_id]

        # Get the node
        node = GraphNode(
            node_id=node_id,
            node_type=node_data.get("node_type", "unknown"),
            text=node_data.get("text", ""),
            metadata=node_data.get("metadata", {}),
            confidence=node_data.get("confidence"),
            created_at=datetime.fromisoformat(
                node_data.get("created_at", datetime.utcnow().isoformat())
            ),
        )

        # Get connected nodes
        connected_nodes = []
        for neighbor in graph.neighbors(node_id):
            neighbor_data = graph.nodes[neighbor]
            connected_nodes.append(
                GraphNode(
                    node_id=neighbor,
                    node_type=neighbor_data.get("node_type", "unknown"),
                    text=neighbor_data.get("text", ""),
                    metadata=neighbor_data.get("metadata", {}),
                    confidence=neighbor_data.get("confidence"),
                    created_at=datetime.fromisoformat(
                        neighbor_data.get("created_at", datetime.utcnow().isoformat())
                    ),
                )
            )

        # Get edges
        edges = []
        for source, target, edge_data in graph.edges(node_id, data=True):
            edges.append(
                GraphEdge(
                    edge_id=f"{source}-{target}",
                    source_id=source,
                    target_id=target,
                    edge_type=edge_data.get("edge_type", "unknown"),
                    metadata=edge_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(
                        edge_data.get("created_at", datetime.utcnow().isoformat())
                    ),
                )
            )

        return NodeDetail(
            node=node,
            connected_nodes=connected_nodes,
            edges=edges,
        )

    async def get_findings(
        self,
        discovery_id: str,
        min_confidence: float = 0.0,
    ) -> List[Finding]:
        """
        Get all findings from the world model.

        Args:
            discovery_id: Discovery ID
            min_confidence: Minimum confidence threshold

        Returns:
            List of findings
        """
        world_model = self.bridge.get_world_model(discovery_id)
        if not world_model:
            return []

        findings = []
        for node_id in world_model.graph.nodes():
            node_data = world_model.graph.nodes[node_id]

            if node_data.get("node_type") != NodeType.FINDING.value:
                continue

            confidence = node_data.get("confidence", 0.0)
            if confidence < min_confidence:
                continue

            findings.append(
                Finding(
                    finding_id=node_id,
                    text=node_data.get("text", ""),
                    confidence=confidence,
                    source=node_data.get("provenance"),
                    cycle_discovered=node_data.get("metadata", {}).get("cycle", 0),
                    metadata=node_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(
                        node_data.get("created_at", datetime.utcnow().isoformat())
                    ),
                )
            )

        return findings

    async def get_hypotheses(
        self,
        discovery_id: str,
        tested_only: bool = False,
    ) -> List[Hypothesis]:
        """
        Get all hypotheses from the world model.

        Args:
            discovery_id: Discovery ID
            tested_only: Only return tested hypotheses

        Returns:
            List of hypotheses
        """
        world_model = self.bridge.get_world_model(discovery_id)
        if not world_model:
            return []

        hypotheses = []
        for node_id in world_model.graph.nodes():
            node_data = world_model.graph.nodes[node_id]

            if node_data.get("node_type") != NodeType.HYPOTHESIS.value:
                continue

            status = node_data.get("metadata", {}).get("status", "untested")
            if tested_only and status == "untested":
                continue

            hypotheses.append(
                Hypothesis(
                    hypothesis_id=node_id,
                    text=node_data.get("text", ""),
                    confidence=node_data.get("confidence"),
                    status=status,
                    cycle_generated=node_data.get("metadata", {}).get("cycle", 0),
                    metadata=node_data.get("metadata", {}),
                    created_at=datetime.fromisoformat(
                        node_data.get("created_at", datetime.utcnow().isoformat())
                    ),
                )
            )

        return hypotheses

    async def get_papers(self, discovery_id: str) -> List[Paper]:
        """
        Get all papers from the world model.

        Args:
            discovery_id: Discovery ID

        Returns:
            List of papers
        """
        world_model = self.bridge.get_world_model(discovery_id)
        if not world_model:
            return []

        papers = []
        for node_id in world_model.graph.nodes():
            node_data = world_model.graph.nodes[node_id]

            if node_data.get("node_type") != NodeType.PAPER.value:
                continue

            metadata = node_data.get("metadata", {})

            papers.append(
                Paper(
                    paper_id=node_id,
                    title=node_data.get("text", ""),
                    authors=metadata.get("authors", []),
                    abstract=metadata.get("abstract"),
                    url=metadata.get("url"),
                    year=metadata.get("year"),
                    relevance_score=node_data.get("confidence"),
                    metadata=metadata,
                    created_at=datetime.fromisoformat(
                        node_data.get("created_at", datetime.utcnow().isoformat())
                    ),
                )
            )

        return papers
