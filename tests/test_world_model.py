"""Tests for the world model."""

import pytest
from kramer.world_model import (
    WorldModel,
    Node,
    Edge,
    Finding,
    NodeType,
    EdgeType
)


class TestNode:
    """Tests for Node class."""

    def test_create_node(self):
        """Test creating a node."""
        node = Node(
            type=NodeType.PAPER,
            content="Test Paper",
            metadata={"authors": ["Alice", "Bob"]}
        )
        assert node.type == NodeType.PAPER
        assert node.content == "Test Paper"
        assert node.metadata["authors"] == ["Alice", "Bob"]
        assert node.id  # Should have auto-generated ID

    def test_node_equality(self):
        """Test node equality based on ID."""
        node1 = Node(type=NodeType.PAPER, content="Paper 1")
        node2 = Node(type=NodeType.PAPER, content="Paper 2")
        node3 = node1  # Same reference

        assert node1 != node2
        assert node1 == node3


class TestEdge:
    """Tests for Edge class."""

    def test_create_edge(self):
        """Test creating an edge."""
        edge = Edge(
            source_id="node1",
            target_id="node2",
            type=EdgeType.CITES,
            weight=0.8
        )
        assert edge.source_id == "node1"
        assert edge.target_id == "node2"
        assert edge.type == EdgeType.CITES
        assert edge.weight == 0.8


class TestFinding:
    """Tests for Finding class."""

    def test_create_finding(self):
        """Test creating a finding."""
        finding = Finding(
            claim_text="This is a key finding",
            source_node_id="paper123",
            confidence=0.9
        )
        assert finding.claim_text == "This is a key finding"
        assert finding.source_node_id == "paper123"
        assert finding.confidence == 0.9

    def test_finding_to_node(self):
        """Test converting finding to claim node."""
        finding = Finding(
            claim_text="Important claim",
            source_node_id="paper123",
            confidence=0.95,
            metadata={"extra": "data"}
        )
        node = finding.to_node()

        assert node.type == NodeType.CLAIM
        assert node.content == "Important claim"
        assert node.metadata["confidence"] == 0.95
        assert node.metadata["source_node_id"] == "paper123"
        assert node.metadata["extra"] == "data"


class TestWorldModel:
    """Tests for WorldModel class."""

    def test_create_empty_world_model(self):
        """Test creating an empty world model."""
        world = WorldModel()
        assert len(world) == 0
        assert len(world.edges) == 0
        assert len(world.findings) == 0

    def test_add_node(self):
        """Test adding a node to the world model."""
        world = WorldModel()
        node = Node(type=NodeType.PAPER, content="Test Paper")

        added = world.add_node(node)
        assert added == node
        assert len(world) == 1
        assert world.get_node(node.id) == node

    def test_add_edge(self):
        """Test adding an edge to the world model."""
        world = WorldModel()

        # Create two nodes
        node1 = world.add_node(Node(type=NodeType.PAPER, content="Paper 1"))
        node2 = world.add_node(Node(type=NodeType.PAPER, content="Paper 2"))

        # Create edge
        edge = Edge(source_id=node1.id, target_id=node2.id, type=EdgeType.CITES)
        added = world.add_edge(edge)

        assert added == edge
        assert len(world.edges) == 1

    def test_add_edge_invalid_nodes(self):
        """Test adding edge with invalid node IDs."""
        world = WorldModel()
        edge = Edge(source_id="invalid1", target_id="invalid2", type=EdgeType.CITES)

        with pytest.raises(ValueError):
            world.add_edge(edge)

    def test_add_finding(self):
        """Test adding a finding to the world model."""
        world = WorldModel()

        # Create paper node
        paper = world.add_node(Node(type=NodeType.PAPER, content="Test Paper"))

        # Create finding
        finding = Finding(
            claim_text="Key finding",
            source_node_id=paper.id,
            confidence=0.8
        )
        added = world.add_finding(finding)

        assert added == finding
        assert len(world.findings) == 1

        # Should have created a claim node
        claim_nodes = world.get_nodes_by_type(NodeType.CLAIM)
        assert len(claim_nodes) == 1
        assert claim_nodes[0].content == "Key finding"

        # Should have created an edge from claim to paper
        edges = world.get_edges_from_node(claim_nodes[0].id)
        assert len(edges) == 1
        assert edges[0].target_id == paper.id
        assert edges[0].type == EdgeType.DERIVED_FROM

    def test_get_nodes_by_type(self):
        """Test filtering nodes by type."""
        world = WorldModel()

        world.add_node(Node(type=NodeType.PAPER, content="Paper 1"))
        world.add_node(Node(type=NodeType.PAPER, content="Paper 2"))
        world.add_node(Node(type=NodeType.CLAIM, content="Claim 1"))

        papers = world.get_nodes_by_type(NodeType.PAPER)
        claims = world.get_nodes_by_type(NodeType.CLAIM)

        assert len(papers) == 2
        assert len(claims) == 1

    def test_get_edges_from_node(self):
        """Test getting edges from a node."""
        world = WorldModel()

        node1 = world.add_node(Node(type=NodeType.PAPER, content="Paper 1"))
        node2 = world.add_node(Node(type=NodeType.PAPER, content="Paper 2"))
        node3 = world.add_node(Node(type=NodeType.PAPER, content="Paper 3"))

        world.add_edge(Edge(source_id=node1.id, target_id=node2.id, type=EdgeType.CITES))
        world.add_edge(Edge(source_id=node1.id, target_id=node3.id, type=EdgeType.CITES))

        edges = world.get_edges_from_node(node1.id)
        assert len(edges) == 2

    def test_get_edges_to_node(self):
        """Test getting edges to a node."""
        world = WorldModel()

        node1 = world.add_node(Node(type=NodeType.PAPER, content="Paper 1"))
        node2 = world.add_node(Node(type=NodeType.PAPER, content="Paper 2"))
        node3 = world.add_node(Node(type=NodeType.PAPER, content="Paper 3"))

        world.add_edge(Edge(source_id=node1.id, target_id=node3.id, type=EdgeType.CITES))
        world.add_edge(Edge(source_id=node2.id, target_id=node3.id, type=EdgeType.CITES))

        edges = world.get_edges_to_node(node3.id)
        assert len(edges) == 2

    def test_get_findings_for_paper(self):
        """Test getting findings for a specific paper."""
        world = WorldModel()

        paper = world.add_node(Node(type=NodeType.PAPER, content="Test Paper"))

        finding1 = Finding(claim_text="Finding 1", source_node_id=paper.id, confidence=0.8)
        finding2 = Finding(claim_text="Finding 2", source_node_id=paper.id, confidence=0.9)

        world.add_finding(finding1)
        world.add_finding(finding2)

        findings = world.get_findings_for_paper(paper.id)
        assert len(findings) == 2

    def test_create_paper_node(self):
        """Test creating a paper node with metadata."""
        world = WorldModel()

        paper = world.create_paper_node(
            title="Test Paper",
            authors=["Alice", "Bob"],
            year=2023,
            doi="10.1234/test",
            abstract="This is a test abstract"
        )

        assert paper.type == NodeType.PAPER
        assert paper.content == "Test Paper"
        assert paper.metadata["title"] == "Test Paper"
        assert paper.metadata["authors"] == ["Alice", "Bob"]
        assert paper.metadata["year"] == 2023
        assert paper.metadata["doi"] == "10.1234/test"
        assert paper.metadata["abstract"] == "This is a test abstract"

    def test_summary(self):
        """Test world model summary statistics."""
        world = WorldModel()

        # Add papers
        paper1 = world.create_paper_node(
            title="Paper 1", authors=["Alice"], year=2023, doi="10.1234/1"
        )
        paper2 = world.create_paper_node(
            title="Paper 2", authors=["Bob"], year=2023, doi="10.1234/2"
        )

        # Add findings
        world.add_finding(Finding(
            claim_text="Finding 1", source_node_id=paper1.id, confidence=0.8
        ))
        world.add_finding(Finding(
            claim_text="Finding 2", source_node_id=paper2.id, confidence=0.9
        ))

        summary = world.summary()
        assert summary["papers"] == 2
        assert summary["claims"] == 2
        assert summary["total_findings"] == 2
        assert summary["total_nodes"] == 4  # 2 papers + 2 claims
