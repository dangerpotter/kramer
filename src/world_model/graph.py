"""
WorldModel - Graph-based knowledge store for scientific findings and hypotheses.

Uses NetworkX for the graph structure and SQLite for persistence.
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

import networkx as nx


class NodeType(str, Enum):
    """Types of nodes in the world model."""
    HYPOTHESIS = "hypothesis"
    FINDING = "finding"
    QUESTION = "question"
    DATASET = "dataset"
    PAPER = "paper"


class EdgeType(str, Enum):
    """Types of edges (relationships) in the world model."""
    SUPPORTS = "supports"
    REFUTES = "refutes"
    DERIVES_FROM = "derives_from"
    RELATES_TO = "relates_to"


class WorldModel:
    """
    Graph-based world model for storing and querying scientific knowledge.

    The world model is a directed graph where:
    - Nodes represent findings, hypotheses, questions, datasets, and papers
    - Edges represent relationships between nodes
    - Each node has metadata (text, confidence, provenance, etc.)

    The graph can be serialized to SQLite for persistence.
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize a new WorldModel.

        Args:
            db_path: Path to SQLite database for persistence. If None, creates in-memory.
        """
        self.graph = nx.DiGraph()
        self.db_path = db_path

        # Track creation time
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()

        # Async-safe locking for database writes
        self._db_lock = asyncio.Lock()

        # Initialize database if path provided
        if db_path:
            self._init_database()

    def _init_database(self) -> None:
        """Initialize SQLite database schema."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Create nodes table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                text TEXT NOT NULL,
                confidence REAL,
                provenance TEXT,
                metadata TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        # Create edges table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS edges (
                source_id TEXT NOT NULL,
                target_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                metadata TEXT,
                created_at TEXT NOT NULL,
                PRIMARY KEY (source_id, target_id, edge_type),
                FOREIGN KEY (source_id) REFERENCES nodes(node_id),
                FOREIGN KEY (target_id) REFERENCES nodes(node_id)
            )
        """)

        # Create metadata table for the world model itself
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS world_model_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        conn.commit()
        conn.close()

    def add_node(
        self,
        node_type: NodeType,
        text: str,
        confidence: Optional[float] = None,
        provenance: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """
        Add a node to the world model (synchronous).

        Note: NetworkX graph modifications are thread-safe for reading but not for writing.
        For async contexts with multiple concurrent writes, use add_node_async instead.

        Args:
            node_type: Type of the node
            text: Main text content of the node
            confidence: Confidence score (0.0 to 1.0)
            provenance: Source of the information (e.g., "analysis/script.py:42")
            metadata: Additional metadata as a dictionary
            node_id: Optional custom node ID (generates UUID if not provided)

        Returns:
            The node ID
        """
        if node_id is None:
            node_id = str(uuid4())

        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")

        now = datetime.utcnow().isoformat()

        self.graph.add_node(
            node_id,
            node_type=node_type.value,
            text=text,
            confidence=confidence,
            provenance=provenance,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

        self.updated_at = datetime.utcnow()
        return node_id

    async def add_node_async(
        self,
        node_type: NodeType,
        text: str,
        confidence: Optional[float] = None,
        provenance: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        node_id: Optional[str] = None,
    ) -> str:
        """
        Add a node to the world model (async version with locking).

        Args:
            node_type: Type of the node
            text: Main text content of the node
            confidence: Confidence score (0.0 to 1.0)
            provenance: Source of the information (e.g., "analysis/script.py:42")
            metadata: Additional metadata as a dictionary
            node_id: Optional custom node ID (generates UUID if not provided)

        Returns:
            The node ID
        """
        async with self._db_lock:
            return self.add_node(node_type, text, confidence, provenance, metadata, node_id)

    def add_finding(
        self,
        text: str,
        code_link: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a finding to the world model.

        Args:
            text: Description of the finding
            code_link: Link to code that generated the finding (e.g., "analysis.py:42")
            confidence: Confidence in the finding (0.0 to 1.0)
            metadata: Additional metadata

        Returns:
            The node ID of the finding
        """
        return self.add_node(
            node_type=NodeType.FINDING,
            text=text,
            confidence=confidence,
            provenance=code_link,
            metadata=metadata,
        )

    def add_hypothesis(
        self,
        text: str,
        parent_finding: Optional[str] = None,
        confidence: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a hypothesis to the world model.

        Args:
            text: Description of the hypothesis
            parent_finding: Node ID of the finding that led to this hypothesis
            confidence: Confidence in the hypothesis (0.0 to 1.0)
            metadata: Additional metadata

        Returns:
            The node ID of the hypothesis
        """
        node_id = self.add_node(
            node_type=NodeType.HYPOTHESIS,
            text=text,
            confidence=confidence,
            metadata=metadata,
        )

        # Link to parent finding if provided
        if parent_finding:
            self.add_edge(
                source=node_id,
                target=parent_finding,
                edge_type=EdgeType.DERIVES_FROM,
            )

        return node_id

    def add_question(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a question to the world model.

        Args:
            text: The question text
            metadata: Additional metadata

        Returns:
            The node ID of the question
        """
        return self.add_node(
            node_type=NodeType.QUESTION,
            text=text,
            metadata=metadata,
        )

    def add_dataset(
        self,
        text: str,
        path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a dataset reference to the world model.

        Args:
            text: Description of the dataset
            path: Path to the dataset file
            metadata: Additional metadata

        Returns:
            The node ID of the dataset
        """
        meta = metadata or {}
        if path:
            meta["path"] = path

        return self.add_node(
            node_type=NodeType.DATASET,
            text=text,
            metadata=meta,
        )

    def add_paper(
        self,
        text: str,
        title: Optional[str] = None,
        authors: Optional[List[str]] = None,
        year: Optional[int] = None,
        doi: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a paper reference to the world model.

        Args:
            text: Summary or key finding from the paper
            title: Paper title
            authors: List of author names
            year: Publication year
            doi: DOI identifier
            metadata: Additional metadata

        Returns:
            The node ID of the paper
        """
        meta = metadata or {}
        if title:
            meta["title"] = title
        if authors:
            meta["authors"] = authors
        if year:
            meta["year"] = year
        if doi:
            meta["doi"] = doi

        return self.add_node(
            node_type=NodeType.PAPER,
            text=text,
            metadata=meta,
        )

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an edge (relationship) between two nodes (synchronous).

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of relationship
            metadata: Additional metadata for the edge
        """
        if source not in self.graph:
            raise ValueError(f"Source node {source} not found in graph")
        if target not in self.graph:
            raise ValueError(f"Target node {target} not found in graph")

        self.graph.add_edge(
            source,
            target,
            edge_type=edge_type.value,
            metadata=metadata or {},
            created_at=datetime.utcnow().isoformat(),
        )

        self.updated_at = datetime.utcnow()

    async def add_edge_async(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add an edge (relationship) between two nodes (async version with locking).

        Args:
            source: Source node ID
            target: Target node ID
            edge_type: Type of relationship
            metadata: Additional metadata for the edge
        """
        async with self._db_lock:
            self.add_edge(source, target, edge_type, metadata)

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a node by ID.

        Args:
            node_id: The node ID

        Returns:
            Node data dictionary or None if not found
        """
        if node_id not in self.graph:
            return None

        return {"node_id": node_id, **self.graph.nodes[node_id]}

    def get_neighbors(
        self,
        node_id: str,
        edge_type: Optional[EdgeType] = None,
        direction: str = "both",
    ) -> List[Dict[str, Any]]:
        """
        Get neighboring nodes.

        Args:
            node_id: The node ID
            edge_type: Filter by edge type (optional)
            direction: "in", "out", or "both"

        Returns:
            List of neighbor node data
        """
        if node_id not in self.graph:
            return []

        neighbors = []

        if direction in ("out", "both"):
            for _, target, data in self.graph.out_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type.value:
                    neighbors.append(self.get_node(target))

        if direction in ("in", "both"):
            for source, _, data in self.graph.in_edges(node_id, data=True):
                if edge_type is None or data.get("edge_type") == edge_type.value:
                    neighbors.append(self.get_node(source))

        return [n for n in neighbors if n is not None]

    def get_relevant_context(
        self,
        query: str,
        max_nodes: int = 50,
        node_types: Optional[List[NodeType]] = None,
        min_confidence: Optional[float] = None,
    ) -> nx.DiGraph:
        """
        Get a relevant subgraph based on a query.

        This performs a simple text-based search through node text and returns
        a subgraph containing matching nodes and their immediate neighbors.

        Args:
            query: Search query string
            max_nodes: Maximum number of nodes to return
            node_types: Filter by node types (optional)
            min_confidence: Minimum confidence threshold (optional)

        Returns:
            A NetworkX DiGraph containing the relevant subgraph
        """
        query_lower = query.lower()
        matching_nodes = []

        for node_id, data in self.graph.nodes(data=True):
            # Filter by node type
            if node_types and data.get("node_type") not in [nt.value for nt in node_types]:
                continue

            # Filter by confidence
            if min_confidence is not None:
                node_conf = data.get("confidence")
                if node_conf is None or node_conf < min_confidence:
                    continue

            # Check if query matches text
            text = data.get("text", "").lower()
            if query_lower in text:
                matching_nodes.append(node_id)

        # Limit to max_nodes
        matching_nodes = matching_nodes[:max_nodes]

        # Build subgraph with neighbors
        subgraph_nodes = set(matching_nodes)
        for node_id in matching_nodes:
            # Add immediate neighbors
            subgraph_nodes.update(self.graph.predecessors(node_id))
            subgraph_nodes.update(self.graph.successors(node_id))

        return self.graph.subgraph(subgraph_nodes).copy()

    def save(self, db_path: Optional[Path] = None) -> None:
        """
        Save the world model to SQLite database (synchronous version).

        Args:
            db_path: Path to database file. Uses self.db_path if not provided.
        """
        path = db_path or self.db_path
        if path is None:
            raise ValueError("No database path provided")

        # Ensure database is initialized
        self.db_path = path
        self._init_database()

        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()

        # Clear existing data
        cursor.execute("DELETE FROM edges")
        cursor.execute("DELETE FROM nodes")

        # Insert nodes
        for node_id, data in self.graph.nodes(data=True):
            cursor.execute(
                """
                INSERT INTO nodes (
                    node_id, node_type, text, confidence, provenance,
                    metadata, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    node_id,
                    data.get("node_type"),
                    data.get("text"),
                    data.get("confidence"),
                    data.get("provenance"),
                    json.dumps(data.get("metadata", {})),
                    data.get("created_at"),
                    data.get("updated_at"),
                ),
            )

        # Insert edges
        for source, target, data in self.graph.edges(data=True):
            cursor.execute(
                """
                INSERT INTO edges (
                    source_id, target_id, edge_type, metadata, created_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    source,
                    target,
                    data.get("edge_type"),
                    json.dumps(data.get("metadata", {})),
                    data.get("created_at"),
                ),
            )

        # Save metadata
        cursor.execute(
            "INSERT OR REPLACE INTO world_model_metadata (key, value) VALUES (?, ?)",
            ("created_at", self.created_at.isoformat()),
        )
        cursor.execute(
            "INSERT OR REPLACE INTO world_model_metadata (key, value) VALUES (?, ?)",
            ("updated_at", self.updated_at.isoformat()),
        )

        conn.commit()
        conn.close()

    async def save_async(self, db_path: Optional[Path] = None) -> None:
        """
        Save the world model to SQLite database (async version with locking).

        Args:
            db_path: Path to database file. Uses self.db_path if not provided.
        """
        async with self._db_lock:
            # Run save in thread pool to avoid blocking
            await asyncio.to_thread(self.save, db_path)

    @classmethod
    def load(cls, db_path: Path) -> "WorldModel":
        """
        Load a world model from SQLite database.

        Args:
            db_path: Path to database file

        Returns:
            A WorldModel instance
        """
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")

        world_model = cls(db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Load nodes
        cursor.execute("SELECT * FROM nodes")
        for row in cursor.fetchall():
            (
                node_id, node_type, text, confidence, provenance,
                metadata_json, created_at, updated_at
            ) = row

            world_model.graph.add_node(
                node_id,
                node_type=node_type,
                text=text,
                confidence=confidence,
                provenance=provenance,
                metadata=json.loads(metadata_json),
                created_at=created_at,
                updated_at=updated_at,
            )

        # Load edges
        cursor.execute("SELECT * FROM edges")
        for row in cursor.fetchall():
            source_id, target_id, edge_type, metadata_json, created_at = row

            world_model.graph.add_edge(
                source_id,
                target_id,
                edge_type=edge_type,
                metadata=json.loads(metadata_json),
                created_at=created_at,
            )

        # Load metadata
        cursor.execute("SELECT value FROM world_model_metadata WHERE key = 'created_at'")
        row = cursor.fetchone()
        if row:
            world_model.created_at = datetime.fromisoformat(row[0])

        cursor.execute("SELECT value FROM world_model_metadata WHERE key = 'updated_at'")
        row = cursor.fetchone()
        if row:
            world_model.updated_at = datetime.fromisoformat(row[0])

        conn.close()

        return world_model

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the world model.

        Returns:
            Dictionary with statistics
        """
        node_types = {}
        for _, data in self.graph.nodes(data=True):
            nt = data.get("node_type", "unknown")
            node_types[nt] = node_types.get(nt, 0) + 1

        edge_types = {}
        for _, _, data in self.graph.edges(data=True):
            et = data.get("edge_type", "unknown")
            edge_types[et] = edge_types.get(et, 0) + 1

        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "edge_types": edge_types,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"WorldModel(nodes={stats['total_nodes']}, "
            f"edges={stats['total_edges']}, "
            f"types={stats['node_types']})"
        )
