"""
Tests for the WorldModel class.
"""

import tempfile
from pathlib import Path

import pytest

from src.world_model.graph import EdgeType, NodeType, WorldModel


class TestWorldModelBasics:
    """Test basic WorldModel functionality."""

    def test_create_world_model(self):
        """Test creating a new world model."""
        wm = WorldModel()
        assert wm.graph.number_of_nodes() == 0
        assert wm.graph.number_of_edges() == 0
        assert wm.created_at is not None
        assert wm.updated_at is not None

    def test_create_world_model_with_db(self, tmp_path):
        """Test creating a world model with database path."""
        db_path = tmp_path / "test.db"
        wm = WorldModel(db_path=db_path)
        assert wm.db_path == db_path
        assert db_path.exists()


class TestNodeOperations:
    """Test node creation and manipulation."""

    def test_add_node(self):
        """Test adding a generic node."""
        wm = WorldModel()
        node_id = wm.add_node(
            node_type=NodeType.FINDING,
            text="Test finding",
            confidence=0.9,
        )
        assert node_id is not None
        assert wm.graph.number_of_nodes() == 1

        node = wm.get_node(node_id)
        assert node is not None
        assert node["text"] == "Test finding"
        assert node["confidence"] == 0.9
        assert node["node_type"] == NodeType.FINDING.value

    def test_add_finding(self):
        """Test adding a finding node."""
        wm = WorldModel()
        node_id = wm.add_finding(
            text="Temperature increased by 2°C",
            code_link="analysis.py:42",
            confidence=0.95,
        )

        node = wm.get_node(node_id)
        assert node["node_type"] == NodeType.FINDING.value
        assert node["text"] == "Temperature increased by 2°C"
        assert node["provenance"] == "analysis.py:42"
        assert node["confidence"] == 0.95

    def test_add_hypothesis(self):
        """Test adding a hypothesis node."""
        wm = WorldModel()

        # Add a finding first
        finding_id = wm.add_finding(
            text="CO2 levels are rising",
            confidence=0.9,
        )

        # Add hypothesis derived from finding
        hyp_id = wm.add_hypothesis(
            text="Rising CO2 causes temperature increase",
            parent_finding=finding_id,
            confidence=0.7,
        )

        node = wm.get_node(hyp_id)
        assert node["node_type"] == NodeType.HYPOTHESIS.value
        assert node["confidence"] == 0.7

        # Check the edge was created
        neighbors = wm.get_neighbors(hyp_id, edge_type=EdgeType.DERIVES_FROM)
        assert len(neighbors) == 1
        assert neighbors[0]["node_id"] == finding_id

    def test_add_question(self):
        """Test adding a question node."""
        wm = WorldModel()
        node_id = wm.add_question(
            text="What causes climate change?",
            metadata={"priority": "high"},
        )

        node = wm.get_node(node_id)
        assert node["node_type"] == NodeType.QUESTION.value
        assert node["metadata"]["priority"] == "high"

    def test_add_dataset(self):
        """Test adding a dataset node."""
        wm = WorldModel()
        node_id = wm.add_dataset(
            text="Climate data 1980-2020",
            path="/data/climate.csv",
            metadata={"rows": 10000},
        )

        node = wm.get_node(node_id)
        assert node["node_type"] == NodeType.DATASET.value
        assert node["metadata"]["path"] == "/data/climate.csv"
        assert node["metadata"]["rows"] == 10000

    def test_add_paper(self):
        """Test adding a paper node."""
        wm = WorldModel()
        node_id = wm.add_paper(
            text="Climate change impacts on biodiversity",
            title="Climate Change and Biodiversity",
            authors=["Smith, J.", "Doe, A."],
            year=2023,
            doi="10.1234/example",
        )

        node = wm.get_node(node_id)
        assert node["node_type"] == NodeType.PAPER.value
        assert node["metadata"]["title"] == "Climate Change and Biodiversity"
        assert node["metadata"]["authors"] == ["Smith, J.", "Doe, A."]
        assert node["metadata"]["year"] == 2023
        assert node["metadata"]["doi"] == "10.1234/example"

    def test_confidence_validation(self):
        """Test that confidence values are validated."""
        wm = WorldModel()

        # Valid confidence
        wm.add_finding(text="Valid", confidence=0.5)

        # Invalid confidence - too low
        with pytest.raises(ValueError):
            wm.add_finding(text="Invalid", confidence=-0.1)

        # Invalid confidence - too high
        with pytest.raises(ValueError):
            wm.add_finding(text="Invalid", confidence=1.1)

    def test_get_nonexistent_node(self):
        """Test getting a node that doesn't exist."""
        wm = WorldModel()
        node = wm.get_node("nonexistent")
        assert node is None


class TestEdgeOperations:
    """Test edge creation and manipulation."""

    def test_add_edge(self):
        """Test adding an edge between nodes."""
        wm = WorldModel()

        # Create two findings
        finding1 = wm.add_finding(text="Finding 1")
        finding2 = wm.add_finding(text="Finding 2")

        # Add edge
        wm.add_edge(finding1, finding2, EdgeType.SUPPORTS)

        assert wm.graph.number_of_edges() == 1

        # Check edge data
        edge_data = wm.graph.get_edge_data(finding1, finding2)
        assert edge_data["edge_type"] == EdgeType.SUPPORTS.value

    def test_add_edge_with_metadata(self):
        """Test adding an edge with metadata."""
        wm = WorldModel()

        finding1 = wm.add_finding(text="Finding 1")
        finding2 = wm.add_finding(text="Finding 2")

        wm.add_edge(
            finding1,
            finding2,
            EdgeType.RELATES_TO,
            metadata={"strength": 0.8},
        )

        edge_data = wm.graph.get_edge_data(finding1, finding2)
        assert edge_data["metadata"]["strength"] == 0.8

    def test_add_edge_invalid_nodes(self):
        """Test adding edge with invalid nodes."""
        wm = WorldModel()
        finding1 = wm.add_finding(text="Finding 1")

        # Try to add edge to nonexistent node
        with pytest.raises(ValueError):
            wm.add_edge(finding1, "nonexistent", EdgeType.SUPPORTS)

        with pytest.raises(ValueError):
            wm.add_edge("nonexistent", finding1, EdgeType.SUPPORTS)

    def test_get_neighbors(self):
        """Test getting neighboring nodes."""
        wm = WorldModel()

        # Create a small graph
        finding = wm.add_finding(text="Central finding")
        support1 = wm.add_finding(text="Supporting finding 1")
        support2 = wm.add_finding(text="Supporting finding 2")
        refute = wm.add_finding(text="Refuting finding")

        wm.add_edge(support1, finding, EdgeType.SUPPORTS)
        wm.add_edge(support2, finding, EdgeType.SUPPORTS)
        wm.add_edge(refute, finding, EdgeType.REFUTES)

        # Get all incoming neighbors
        neighbors = wm.get_neighbors(finding, direction="in")
        assert len(neighbors) == 3

        # Get only supporting neighbors
        supporters = wm.get_neighbors(finding, edge_type=EdgeType.SUPPORTS, direction="in")
        assert len(supporters) == 2

        # Get only refuting neighbors
        refuters = wm.get_neighbors(finding, edge_type=EdgeType.REFUTES, direction="in")
        assert len(refuters) == 1


class TestRelevantContext:
    """Test getting relevant context from the world model."""

    def test_get_relevant_context_basic(self):
        """Test basic context retrieval."""
        wm = WorldModel()

        # Add some nodes
        wm.add_finding(text="Climate change is accelerating")
        wm.add_finding(text="Ocean temperatures are rising")
        wm.add_finding(text="Polar ice is melting")
        wm.add_finding(text="Unrelated finding about economics")

        # Search for climate-related content
        subgraph = wm.get_relevant_context("climate")
        assert subgraph.number_of_nodes() == 1

        # Search for ocean-related content
        subgraph = wm.get_relevant_context("ocean")
        assert subgraph.number_of_nodes() == 1

    def test_get_relevant_context_with_edges(self):
        """Test context retrieval includes neighbors."""
        wm = WorldModel()

        # Create a connected graph
        finding = wm.add_finding(text="Climate change finding")
        hypothesis = wm.add_hypothesis(text="Climate hypothesis", parent_finding=finding)
        question = wm.add_question(text="What about climate?")

        wm.add_edge(question, finding, EdgeType.RELATES_TO)

        # Search for climate
        subgraph = wm.get_relevant_context("climate")

        # Should include all three nodes (matched + neighbors)
        assert subgraph.number_of_nodes() == 3

    def test_get_relevant_context_node_type_filter(self):
        """Test filtering by node type."""
        wm = WorldModel()

        wm.add_finding(text="Climate finding")
        wm.add_hypothesis(text="Climate hypothesis")
        wm.add_question(text="Climate question")

        # Get only findings
        subgraph = wm.get_relevant_context("climate", node_types=[NodeType.FINDING])
        assert subgraph.number_of_nodes() >= 1

        # Check all matched nodes are findings
        for node_id, data in subgraph.nodes(data=True):
            if "climate" in data.get("text", "").lower():
                assert data["node_type"] == NodeType.FINDING.value

    def test_get_relevant_context_confidence_filter(self):
        """Test filtering by confidence."""
        wm = WorldModel()

        wm.add_finding(text="High confidence climate finding", confidence=0.9)
        wm.add_finding(text="Low confidence climate finding", confidence=0.3)

        # Get only high confidence findings
        subgraph = wm.get_relevant_context("climate", min_confidence=0.5)

        # Should only get the high confidence one
        matched = [
            node_id for node_id, data in subgraph.nodes(data=True)
            if "climate" in data.get("text", "").lower()
        ]
        assert len(matched) == 1

    def test_get_relevant_context_max_nodes(self):
        """Test limiting number of nodes."""
        wm = WorldModel()

        # Add many matching nodes
        for i in range(20):
            wm.add_finding(text=f"Climate finding {i}")

        # Limit to 5 nodes
        subgraph = wm.get_relevant_context("climate", max_nodes=5)

        # Should not exceed max_nodes (plus neighbors, but we have no edges)
        assert subgraph.number_of_nodes() <= 5


class TestPersistence:
    """Test saving and loading world models."""

    def test_save_and_load_empty(self, tmp_path):
        """Test saving and loading an empty world model."""
        db_path = tmp_path / "test.db"

        # Create and save
        wm1 = WorldModel(db_path=db_path)
        wm1.save()

        # Load
        wm2 = WorldModel.load(db_path)
        assert wm2.graph.number_of_nodes() == 0
        assert wm2.graph.number_of_edges() == 0

    def test_save_and_load_with_data(self, tmp_path):
        """Test saving and loading a world model with data."""
        db_path = tmp_path / "test.db"

        # Create world model with data
        wm1 = WorldModel(db_path=db_path)
        finding1 = wm1.add_finding(
            text="Important finding",
            code_link="test.py:1",
            confidence=0.95,
        )
        finding2 = wm1.add_finding(text="Another finding")
        wm1.add_edge(finding1, finding2, EdgeType.SUPPORTS)

        hyp = wm1.add_hypothesis(
            text="Test hypothesis",
            parent_finding=finding1,
            confidence=0.7,
        )

        # Save
        wm1.save()

        # Load
        wm2 = WorldModel.load(db_path)

        # Verify nodes
        assert wm2.graph.number_of_nodes() == 3
        assert wm2.graph.number_of_edges() == 2

        # Verify finding data
        node = wm2.get_node(finding1)
        assert node["text"] == "Important finding"
        assert node["provenance"] == "test.py:1"
        assert node["confidence"] == 0.95

        # Verify hypothesis data
        node = wm2.get_node(hyp)
        assert node["text"] == "Test hypothesis"
        assert node["confidence"] == 0.7

        # Verify edges
        edge_data = wm2.graph.get_edge_data(finding1, finding2)
        assert edge_data["edge_type"] == EdgeType.SUPPORTS.value

    def test_save_updates_existing_db(self, tmp_path):
        """Test that save overwrites existing database."""
        db_path = tmp_path / "test.db"

        # Create and save first version
        wm1 = WorldModel(db_path=db_path)
        wm1.add_finding(text="Original finding")
        wm1.save()

        # Create new version with different data
        wm2 = WorldModel(db_path=db_path)
        wm2.add_finding(text="New finding")
        wm2.save()

        # Load and verify
        wm3 = WorldModel.load(db_path)
        assert wm3.graph.number_of_nodes() == 1
        nodes = list(wm3.graph.nodes(data=True))
        assert nodes[0][1]["text"] == "New finding"

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            WorldModel.load(Path("/nonexistent/path.db"))

    def test_save_without_db_path(self):
        """Test saving without database path."""
        wm = WorldModel()
        with pytest.raises(ValueError):
            wm.save()


class TestStatistics:
    """Test statistics and summary methods."""

    def test_get_stats_empty(self):
        """Test statistics on empty world model."""
        wm = WorldModel()
        stats = wm.get_stats()

        assert stats["total_nodes"] == 0
        assert stats["total_edges"] == 0
        assert stats["node_types"] == {}
        assert stats["edge_types"] == {}
        assert "created_at" in stats
        assert "updated_at" in stats

    def test_get_stats_with_data(self):
        """Test statistics with data."""
        wm = WorldModel()

        # Add various nodes
        f1 = wm.add_finding(text="Finding 1")
        f2 = wm.add_finding(text="Finding 2")
        h1 = wm.add_hypothesis(text="Hypothesis 1")
        q1 = wm.add_question(text="Question 1")

        # Add edges
        wm.add_edge(f1, f2, EdgeType.SUPPORTS)
        wm.add_edge(h1, f1, EdgeType.DERIVES_FROM)

        stats = wm.get_stats()

        assert stats["total_nodes"] == 4
        assert stats["total_edges"] == 2
        assert stats["node_types"]["finding"] == 2
        assert stats["node_types"]["hypothesis"] == 1
        assert stats["node_types"]["question"] == 1
        assert stats["edge_types"]["supports"] == 1
        assert stats["edge_types"]["derives_from"] == 1

    def test_repr(self):
        """Test string representation."""
        wm = WorldModel()
        wm.add_finding(text="Test")
        wm.add_hypothesis(text="Test")

        repr_str = repr(wm)
        assert "WorldModel" in repr_str
        assert "nodes=2" in repr_str
