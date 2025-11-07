"""
Event types and event handling for real-time updates.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict
from pydantic import BaseModel


class EventType(str, Enum):
    """Types of events emitted during discovery."""
    DISCOVERY_STARTED = "discovery_started"
    DISCOVERY_COMPLETED = "discovery_completed"
    DISCOVERY_FAILED = "discovery_failed"
    DISCOVERY_STOPPED = "discovery_stopped"

    CYCLE_STARTED = "cycle_started"
    CYCLE_COMPLETED = "cycle_completed"

    TASK_CREATED = "task_created"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"

    FINDING_ADDED = "finding_added"
    HYPOTHESIS_GENERATED = "hypothesis_generated"
    HYPOTHESIS_TESTED = "hypothesis_tested"

    PAPER_RETRIEVED = "paper_retrieved"

    BUDGET_WARNING = "budget_warning"
    CHECKPOINT_CREATED = "checkpoint_created"

    PROGRESS_UPDATE = "progress_update"
    LOG_MESSAGE = "log_message"


class Event(BaseModel):
    """Base event structure."""
    event_type: EventType
    discovery_id: str
    timestamp: datetime
    data: Dict[str, Any]

    class Config:
        use_enum_values = True


def create_event(
    event_type: EventType,
    discovery_id: str,
    data: Dict[str, Any]
) -> Event:
    """Create a new event."""
    return Event(
        event_type=event_type,
        discovery_id=discovery_id,
        timestamp=datetime.utcnow(),
        data=data,
    )
