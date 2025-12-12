"""
SQLAlchemy ORM models for PostgreSQL persistence.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from sqlalchemy import (
    Column, String, Text, Float, Integer, Boolean, DateTime,
    ForeignKey, JSON, Index, Enum as SQLEnum
)
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid
import enum

from app.core.database import Base


class DiscoveryStatus(str, enum.Enum):
    """Discovery status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TaskStatus(str, enum.Enum):
    """Task status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CycleStatus(str, enum.Enum):
    """Cycle status enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    BUDGET_EXCEEDED = "budget_exceeded"
    FAILED = "failed"


class Discovery(Base):
    """Discovery session model."""
    __tablename__ = "discoveries"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    objective = Column(Text, nullable=False)
    dataset_path = Column(String(500), nullable=True)
    model = Column(String(100), nullable=False)

    # Configuration
    max_cycles = Column(Integer, default=20)
    max_total_budget = Column(Float, default=100.0)
    max_parallel_tasks = Column(Integer, default=4)

    # Status
    status = Column(String(20), default=DiscoveryStatus.PENDING.value)
    current_cycle = Column(Integer, default=0)
    total_cost = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    cycles = relationship("Cycle", back_populates="discovery", cascade="all, delete-orphan")
    nodes = relationship("WorldModelNode", back_populates="discovery", cascade="all, delete-orphan")
    edges = relationship("WorldModelEdge", back_populates="discovery", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_discovery_status', 'status'),
        Index('idx_discovery_created', 'created_at'),
    )


class Cycle(Base):
    """Research cycle model."""
    __tablename__ = "cycles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    discovery_id = Column(String(36), ForeignKey("discoveries.id", ondelete="CASCADE"), nullable=False)
    cycle_number = Column(Integer, nullable=False)
    objective = Column(Text, nullable=False)

    # Status
    status = Column(String(20), default=CycleStatus.PENDING.value)
    budget_used = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    discovery = relationship("Discovery", back_populates="cycles")
    tasks = relationship("Task", back_populates="cycle", cascade="all, delete-orphan")

    __table_args__ = (
        Index('idx_cycle_discovery', 'discovery_id'),
        Index('idx_cycle_number', 'discovery_id', 'cycle_number'),
    )


class Task(Base):
    """Task model."""
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    cycle_id = Column(String(36), ForeignKey("cycles.id", ondelete="CASCADE"), nullable=False)
    task_type = Column(String(50), nullable=False)
    objective = Column(Text, nullable=False)

    # Status
    status = Column(String(20), default=TaskStatus.PENDING.value)

    # Context and result stored as JSON
    context = Column(JSON, default=dict)
    result = Column(JSON, nullable=True)
    error = Column(Text, nullable=True)
    cost = Column(Float, default=0.0)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Relationships
    cycle = relationship("Cycle", back_populates="tasks")

    __table_args__ = (
        Index('idx_task_cycle', 'cycle_id'),
        Index('idx_task_status', 'status'),
    )


class WorldModelNode(Base):
    """World model node (finding, hypothesis, paper, etc.)."""
    __tablename__ = "world_model_nodes"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    discovery_id = Column(String(36), ForeignKey("discoveries.id", ondelete="CASCADE"), nullable=False)

    node_type = Column(String(50), nullable=False)  # hypothesis, finding, paper, etc.
    text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=True)
    provenance = Column(String(500), nullable=True)
    extra_data = Column(JSON, default=dict)  # renamed from 'metadata' (reserved by SQLAlchemy)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    discovery = relationship("Discovery", back_populates="nodes")

    __table_args__ = (
        Index('idx_node_discovery', 'discovery_id'),
        Index('idx_node_type', 'discovery_id', 'node_type'),
    )


class WorldModelEdge(Base):
    """World model edge (relationship between nodes)."""
    __tablename__ = "world_model_edges"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    discovery_id = Column(String(36), ForeignKey("discoveries.id", ondelete="CASCADE"), nullable=False)

    source_id = Column(String(36), nullable=False)
    target_id = Column(String(36), nullable=False)
    edge_type = Column(String(50), nullable=False)  # supports, refutes, derives_from, etc.
    extra_data = Column(JSON, default=dict)  # renamed from 'metadata' (reserved by SQLAlchemy)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    discovery = relationship("Discovery", back_populates="edges")

    __table_args__ = (
        Index('idx_edge_discovery', 'discovery_id'),
        Index('idx_edge_source', 'source_id'),
        Index('idx_edge_target', 'target_id'),
    )
