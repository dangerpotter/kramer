"""
Pydantic models for task-related data.
"""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel


class TaskInfo(BaseModel):
    """Information about a task."""
    task_id: str
    task_type: str
    status: str
    objective: str
    context: Dict[str, Any]
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parent_task_id: Optional[str] = None


class TaskResult(BaseModel):
    """Result of a completed task."""
    task_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    cost: float
    duration: float
    findings_generated: int
    hypotheses_generated: int
