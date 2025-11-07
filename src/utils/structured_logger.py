"""
Structured Logging System with Monitoring Support.

Provides:
- JSON structured logging
- Log aggregation to files
- Metrics collection
- Monitoring dashboard data export
- Performance tracking
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
import time
from collections import defaultdict
import threading


class LogLevel(Enum):
    """Log severity levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class EventType(Enum):
    """Types of events to log."""
    CYCLE_START = "cycle_start"
    CYCLE_END = "cycle_end"
    TASK_START = "task_start"
    TASK_END = "task_end"
    AGENT_CALL = "agent_call"
    API_CALL = "api_call"
    ERROR = "error"
    BUDGET_UPDATE = "budget_update"
    HYPOTHESIS_GENERATED = "hypothesis_generated"
    HYPOTHESIS_TESTED = "hypothesis_tested"
    FINDING_ADDED = "finding_added"
    SYNTHESIS = "synthesis"
    CHECKPOINT = "checkpoint"


@dataclass
class StructuredLogEntry:
    """Structured log entry with metadata."""
    timestamp: str
    level: str
    event_type: str
    message: str
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Performance metrics
    duration_ms: Optional[float] = None

    # Cost tracking
    cost: Optional[float] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None

    # Context
    cycle_number: Optional[int] = None
    task_id: Optional[str] = None
    hypothesis_id: Optional[str] = None

    # Error details
    error_type: Optional[str] = None
    stack_trace: Optional[str] = None


class StructuredLogger:
    """
    Structured logger with JSON output and metrics collection.

    Features:
    - JSON formatted logs
    - File and console outputs
    - Automatic metrics aggregation
    - Performance tracking
    - Cost tracking
    """

    def __init__(
        self,
        name: str,
        log_dir: str = "logs",
        console_output: bool = True,
        file_output: bool = True,
        min_level: LogLevel = LogLevel.INFO
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name (component identifier)
            log_dir: Directory for log files
            console_output: Enable console logging
            file_output: Enable file logging
            min_level: Minimum log level to capture
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.console_output = console_output
        self.file_output = file_output
        self.min_level = min_level

        # Create log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_log_file = self.log_dir / f"{name}_{timestamp}.jsonl"
        self.text_log_file = self.log_dir / f"{name}_{timestamp}.log"

        # Metrics storage
        self._metrics_lock = threading.Lock()
        self._metrics = {
            "event_counts": defaultdict(int),
            "total_cost": 0.0,
            "total_api_calls": 0,
            "total_errors": 0,
            "cycle_durations": [],
            "task_durations": [],
            "agent_call_durations": [],
        }

        # Standard Python logger for fallback
        self._std_logger = logging.getLogger(name)

    def log(
        self,
        level: LogLevel,
        event_type: EventType,
        message: str,
        **kwargs
    ):
        """
        Log a structured event.

        Args:
            level: Log severity level
            event_type: Type of event
            message: Human-readable message
            **kwargs: Additional metadata fields
        """
        # Check if should log based on level
        if not self._should_log(level):
            return

        # Create log entry
        entry = StructuredLogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            event_type=event_type.value,
            message=message,
            component=self.name,
            **kwargs
        )

        # Write to outputs
        self._write_json(entry)
        self._write_text(entry)

        # Update metrics
        self._update_metrics(entry)

    def debug(self, event_type: EventType, message: str, **kwargs):
        """Log debug event."""
        self.log(LogLevel.DEBUG, event_type, message, **kwargs)

    def info(self, event_type: EventType, message: str, **kwargs):
        """Log info event."""
        self.log(LogLevel.INFO, event_type, message, **kwargs)

    def warning(self, event_type: EventType, message: str, **kwargs):
        """Log warning event."""
        self.log(LogLevel.WARNING, event_type, message, **kwargs)

    def error(self, event_type: EventType, message: str, **kwargs):
        """Log error event."""
        self.log(LogLevel.ERROR, event_type, message, **kwargs)

    def critical(self, event_type: EventType, message: str, **kwargs):
        """Log critical event."""
        self.log(LogLevel.CRITICAL, event_type, message, **kwargs)

    def log_cycle_start(self, cycle_number: int, objective: str):
        """Log cycle start event."""
        self.info(
            EventType.CYCLE_START,
            f"Starting cycle {cycle_number}",
            cycle_number=cycle_number,
            metadata={"objective": objective}
        )

    def log_cycle_end(
        self,
        cycle_number: int,
        duration_ms: float,
        budget_used: float,
        tasks_completed: int
    ):
        """Log cycle completion event."""
        self.info(
            EventType.CYCLE_END,
            f"Completed cycle {cycle_number}",
            cycle_number=cycle_number,
            duration_ms=duration_ms,
            cost=budget_used,
            metadata={
                "tasks_completed": tasks_completed
            }
        )

    def log_task_start(self, task_id: str, task_type: str, cycle_number: int):
        """Log task start event."""
        self.info(
            EventType.TASK_START,
            f"Starting task {task_id}",
            task_id=task_id,
            cycle_number=cycle_number,
            metadata={"task_type": task_type}
        )

    def log_task_end(
        self,
        task_id: str,
        duration_ms: float,
        success: bool,
        cost: float = 0.0,
        error: Optional[str] = None
    ):
        """Log task completion event."""
        level = LogLevel.INFO if success else LogLevel.ERROR
        self.log(
            level,
            EventType.TASK_END,
            f"Task {task_id} {'completed' if success else 'failed'}",
            task_id=task_id,
            duration_ms=duration_ms,
            cost=cost,
            metadata={"success": success, "error": error}
        )

    def log_agent_call(
        self,
        agent_type: str,
        duration_ms: float,
        cost: float,
        tokens_input: int,
        tokens_output: int,
        success: bool
    ):
        """Log agent API call."""
        self.info(
            EventType.AGENT_CALL,
            f"Agent call: {agent_type}",
            duration_ms=duration_ms,
            cost=cost,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            metadata={"agent_type": agent_type, "success": success}
        )

    def log_api_call(
        self,
        endpoint: str,
        duration_ms: float,
        status_code: int,
        cost: float = 0.0
    ):
        """Log external API call."""
        self.info(
            EventType.API_CALL,
            f"API call: {endpoint}",
            duration_ms=duration_ms,
            cost=cost,
            metadata={"endpoint": endpoint, "status_code": status_code}
        )

    def log_error(
        self,
        message: str,
        error_type: str,
        stack_trace: Optional[str] = None,
        **kwargs
    ):
        """Log error with details."""
        self.error(
            EventType.ERROR,
            message,
            error_type=error_type,
            stack_trace=stack_trace,
            **kwargs
        )

    def log_budget_update(
        self,
        cycle_number: int,
        cycle_budget: float,
        total_budget: float,
        budget_limit: float
    ):
        """Log budget update."""
        self.info(
            EventType.BUDGET_UPDATE,
            f"Budget update for cycle {cycle_number}",
            cycle_number=cycle_number,
            cost=cycle_budget,
            metadata={
                "total_budget": total_budget,
                "budget_limit": budget_limit,
                "utilization_pct": (total_budget / budget_limit * 100) if budget_limit > 0 else 0
            }
        )

    def log_hypothesis_generated(
        self,
        hypothesis_id: str,
        hypothesis_text: str,
        cycle_number: int
    ):
        """Log hypothesis generation."""
        self.info(
            EventType.HYPOTHESIS_GENERATED,
            f"Generated hypothesis: {hypothesis_id}",
            hypothesis_id=hypothesis_id,
            cycle_number=cycle_number,
            metadata={"hypothesis_text": hypothesis_text}
        )

    def log_hypothesis_tested(
        self,
        hypothesis_id: str,
        outcome: str,
        confidence: float,
        cycle_number: int
    ):
        """Log hypothesis test result."""
        self.info(
            EventType.HYPOTHESIS_TESTED,
            f"Tested hypothesis {hypothesis_id}: {outcome}",
            hypothesis_id=hypothesis_id,
            cycle_number=cycle_number,
            metadata={"outcome": outcome, "confidence": confidence}
        )

    def log_finding_added(
        self,
        finding_text: str,
        confidence: float,
        cycle_number: int
    ):
        """Log new finding."""
        self.info(
            EventType.FINDING_ADDED,
            f"New finding (confidence: {confidence:.2f})",
            cycle_number=cycle_number,
            metadata={"finding_text": finding_text, "confidence": confidence}
        )

    def log_synthesis(
        self,
        cycle_number: int,
        report_path: str,
        findings_count: int,
        hypotheses_count: int
    ):
        """Log synthesis generation."""
        self.info(
            EventType.SYNTHESIS,
            f"Generated synthesis report",
            cycle_number=cycle_number,
            metadata={
                "report_path": report_path,
                "findings_count": findings_count,
                "hypotheses_count": hypotheses_count
            }
        )

    def log_checkpoint(
        self,
        checkpoint_path: str,
        cycle_number: int,
        total_budget: float
    ):
        """Log checkpoint creation."""
        self.info(
            EventType.CHECKPOINT,
            f"Created checkpoint at cycle {cycle_number}",
            cycle_number=cycle_number,
            metadata={
                "checkpoint_path": checkpoint_path,
                "total_budget": total_budget
            }
        )

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get aggregated metrics.

        Returns:
            Dictionary with metrics data
        """
        with self._metrics_lock:
            return {
                "event_counts": dict(self._metrics["event_counts"]),
                "total_cost": self._metrics["total_cost"],
                "total_api_calls": self._metrics["total_api_calls"],
                "total_errors": self._metrics["total_errors"],
                "avg_cycle_duration_ms": (
                    sum(self._metrics["cycle_durations"]) / len(self._metrics["cycle_durations"])
                    if self._metrics["cycle_durations"] else 0
                ),
                "avg_task_duration_ms": (
                    sum(self._metrics["task_durations"]) / len(self._metrics["task_durations"])
                    if self._metrics["task_durations"] else 0
                ),
                "avg_agent_call_duration_ms": (
                    sum(self._metrics["agent_call_durations"]) / len(self._metrics["agent_call_durations"])
                    if self._metrics["agent_call_durations"] else 0
                ),
            }

    def export_metrics(self, output_path: Optional[Path] = None) -> Path:
        """
        Export metrics to JSON file.

        Args:
            output_path: Optional custom output path

        Returns:
            Path to exported metrics file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.log_dir / f"metrics_{timestamp}.json"

        metrics = self.get_metrics()

        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        return output_path

    def _should_log(self, level: LogLevel) -> bool:
        """Check if message should be logged based on level."""
        level_order = {
            LogLevel.DEBUG: 0,
            LogLevel.INFO: 1,
            LogLevel.WARNING: 2,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 4
        }
        return level_order[level] >= level_order[self.min_level]

    def _write_json(self, entry: StructuredLogEntry):
        """Write JSON log entry to file."""
        if not self.file_output:
            return

        try:
            with open(self.json_log_file, 'a') as f:
                json.dump(asdict(entry), f)
                f.write('\n')
        except Exception as e:
            self._std_logger.error(f"Failed to write JSON log: {e}")

    def _write_text(self, entry: StructuredLogEntry):
        """Write human-readable log entry."""
        # Format text message
        text = f"{entry.timestamp} [{entry.level}] {entry.component} - {entry.message}"

        # Add important metadata
        if entry.duration_ms is not None:
            text += f" (duration: {entry.duration_ms:.1f}ms)"
        if entry.cost is not None:
            text += f" (cost: ${entry.cost:.4f})"

        # Console output
        if self.console_output:
            print(text, file=sys.stderr if entry.level in ["ERROR", "CRITICAL"] else sys.stdout)

        # File output
        if self.file_output:
            try:
                with open(self.text_log_file, 'a') as f:
                    f.write(text + '\n')
            except Exception as e:
                self._std_logger.error(f"Failed to write text log: {e}")

    def _update_metrics(self, entry: StructuredLogEntry):
        """Update aggregated metrics from log entry."""
        with self._metrics_lock:
            # Count events
            self._metrics["event_counts"][entry.event_type] += 1

            # Track costs
            if entry.cost is not None:
                self._metrics["total_cost"] += entry.cost

            # Track API calls
            if entry.event_type in [EventType.AGENT_CALL.value, EventType.API_CALL.value]:
                self._metrics["total_api_calls"] += 1

            # Track errors
            if entry.level == LogLevel.ERROR.value:
                self._metrics["total_errors"] += 1

            # Track durations
            if entry.duration_ms is not None:
                if entry.event_type == EventType.CYCLE_END.value:
                    self._metrics["cycle_durations"].append(entry.duration_ms)
                elif entry.event_type == EventType.TASK_END.value:
                    self._metrics["task_durations"].append(entry.duration_ms)
                elif entry.event_type == EventType.AGENT_CALL.value:
                    self._metrics["agent_call_durations"].append(entry.duration_ms)


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(
        self,
        logger: StructuredLogger,
        event_type: EventType,
        message: str,
        **kwargs
    ):
        """
        Initialize performance timer.

        Args:
            logger: StructuredLogger instance
            event_type: Type of event to log
            message: Message to log
            **kwargs: Additional metadata
        """
        self.logger = logger
        self.event_type = event_type
        self.message = message
        self.kwargs = kwargs
        self.start_time = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log."""
        duration_ms = (time.time() - self.start_time) * 1000

        if exc_type is not None:
            # Log error
            self.logger.error(
                self.event_type,
                f"{self.message} (failed)",
                duration_ms=duration_ms,
                error_type=exc_type.__name__,
                stack_trace=str(exc_val),
                **self.kwargs
            )
        else:
            # Log success
            self.logger.info(
                self.event_type,
                self.message,
                duration_ms=duration_ms,
                **self.kwargs
            )


# Global logger registry
_loggers: Dict[str, StructuredLogger] = {}


def get_logger(
    name: str,
    log_dir: str = "logs",
    **kwargs
) -> StructuredLogger:
    """
    Get or create a structured logger.

    Args:
        name: Logger name
        log_dir: Log directory
        **kwargs: Additional logger configuration

    Returns:
        StructuredLogger instance
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, log_dir, **kwargs)

    return _loggers[name]


def get_all_metrics() -> Dict[str, Any]:
    """
    Get metrics from all loggers.

    Returns:
        Dictionary mapping logger names to their metrics
    """
    return {
        name: logger.get_metrics()
        for name, logger in _loggers.items()
    }
