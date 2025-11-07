"""
Pydantic models for discovery-related API endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DiscoveryStatus(str, Enum):
    """Discovery status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class DiscoveryConfig(BaseModel):
    """Configuration for starting a new discovery."""
    objective: str = Field(..., description="The research objective to pursue")
    dataset_path: Optional[str] = Field(None, description="Path to dataset file")
    max_cycles: int = Field(20, ge=1, le=100, description="Maximum number of cycles")
    max_total_budget: float = Field(100.0, ge=0, description="Maximum budget in USD")
    max_parallel_tasks: int = Field(4, ge=1, le=10, description="Max parallel tasks")
    enable_checkpointing: bool = Field(True, description="Enable automatic checkpointing")
    checkpoint_interval: int = Field(5, ge=1, description="Cycles between checkpoints")


class DiscoveryResponse(BaseModel):
    """Response when creating a new discovery."""
    discovery_id: str
    status: DiscoveryStatus
    message: Optional[str] = None


class TaskStatusInfo(BaseModel):
    """Status information for a task."""
    task_id: str
    task_type: str
    status: str
    objective: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class CycleInfo(BaseModel):
    """Information about a discovery cycle."""
    cycle_id: str
    cycle_number: int
    status: str
    tasks: List[TaskStatusInfo]
    budget_used: float
    findings_generated: int
    hypotheses_generated: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class DiscoveryDetail(BaseModel):
    """Detailed information about a discovery session."""
    discovery_id: str
    objective: str
    status: DiscoveryStatus
    config: DiscoveryConfig
    current_cycle: int
    total_cycles: int
    total_cost: float
    findings_count: int
    hypotheses_count: int
    papers_count: int
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class MetricsResponse(BaseModel):
    """Real-time metrics for a discovery."""
    discovery_id: str
    current_cycle: int
    total_cost: float
    cost_per_cycle: List[float]
    findings_per_cycle: List[int]
    hypotheses_per_cycle: List[int]
    tasks_completed: int
    tasks_pending: int
    tasks_running: int
    avg_task_duration: float
    estimated_time_remaining: Optional[float] = None


class DiscoveryList(BaseModel):
    """List of discoveries."""
    discoveries: List[DiscoveryDetail]
    total: int
