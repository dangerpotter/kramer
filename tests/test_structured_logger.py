"""
Tests for the StructuredLogger utility module.
"""

import json
import pytest
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch
import sys
import io

from src.utils.structured_logger import (
    StructuredLogger,
    StructuredLogEntry,
    LogLevel,
    EventType,
    PerformanceTimer,
    get_logger,
    get_all_metrics,
)


class TestLogLevelEnum:
    """Test LogLevel enumeration."""

    def test_log_levels_defined(self):
        """Test all log levels are defined."""
        assert LogLevel.DEBUG.value == "DEBUG"
        assert LogLevel.INFO.value == "INFO"
        assert LogLevel.WARNING.value == "WARNING"
        assert LogLevel.ERROR.value == "ERROR"
        assert LogLevel.CRITICAL.value == "CRITICAL"


class TestEventTypeEnum:
    """Test EventType enumeration."""

    def test_event_types_defined(self):
        """Test all event types are defined."""
        assert EventType.CYCLE_START.value == "cycle_start"
        assert EventType.CYCLE_END.value == "cycle_end"
        assert EventType.TASK_START.value == "task_start"
        assert EventType.TASK_END.value == "task_end"
        assert EventType.AGENT_CALL.value == "agent_call"
        assert EventType.API_CALL.value == "api_call"
        assert EventType.ERROR.value == "error"
        assert EventType.BUDGET_UPDATE.value == "budget_update"
        assert EventType.HYPOTHESIS_GENERATED.value == "hypothesis_generated"
        assert EventType.HYPOTHESIS_TESTED.value == "hypothesis_tested"
        assert EventType.FINDING_ADDED.value == "finding_added"
        assert EventType.SYNTHESIS.value == "synthesis"
        assert EventType.CHECKPOINT.value == "checkpoint"


class TestStructuredLogEntry:
    """Test StructuredLogEntry dataclass."""

    def test_create_basic_entry(self):
        """Test creating a basic log entry."""
        entry = StructuredLogEntry(
            timestamp="2024-01-01T00:00:00",
            level="INFO",
            event_type="test_event",
            message="Test message",
            component="test_component"
        )

        assert entry.timestamp == "2024-01-01T00:00:00"
        assert entry.level == "INFO"
        assert entry.event_type == "test_event"
        assert entry.message == "Test message"
        assert entry.component == "test_component"
        assert entry.metadata == {}

    def test_create_entry_with_metrics(self):
        """Test creating entry with performance metrics."""
        entry = StructuredLogEntry(
            timestamp="2024-01-01T00:00:00",
            level="INFO",
            event_type="api_call",
            message="API call completed",
            component="agent",
            duration_ms=150.5,
            cost=0.002,
            tokens_input=100,
            tokens_output=50
        )

        assert entry.duration_ms == 150.5
        assert entry.cost == 0.002
        assert entry.tokens_input == 100
        assert entry.tokens_output == 50

    def test_create_entry_with_context(self):
        """Test creating entry with context info."""
        entry = StructuredLogEntry(
            timestamp="2024-01-01T00:00:00",
            level="INFO",
            event_type="task_start",
            message="Task started",
            component="orchestrator",
            cycle_number=5,
            task_id="task-123",
            hypothesis_id="hyp-456"
        )

        assert entry.cycle_number == 5
        assert entry.task_id == "task-123"
        assert entry.hypothesis_id == "hyp-456"

    def test_create_entry_with_error(self):
        """Test creating entry with error details."""
        entry = StructuredLogEntry(
            timestamp="2024-01-01T00:00:00",
            level="ERROR",
            event_type="error",
            message="Something went wrong",
            component="executor",
            error_type="ValueError",
            stack_trace="Traceback..."
        )

        assert entry.error_type == "ValueError"
        assert entry.stack_trace == "Traceback..."


class TestStructuredLoggerBasics:
    """Test basic StructuredLogger functionality."""

    def test_create_logger(self, tmp_path):
        """Test creating a structured logger."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False,
            file_output=True
        )

        assert logger.name == "test_logger"
        assert logger.log_dir == tmp_path
        assert logger.console_output is False
        assert logger.file_output is True
        assert logger.min_level == LogLevel.INFO

    def test_create_logger_custom_level(self, tmp_path):
        """Test creating logger with custom min level."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            min_level=LogLevel.DEBUG
        )

        assert logger.min_level == LogLevel.DEBUG

    def test_log_files_created(self, tmp_path):
        """Test that log files are created."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False,
            file_output=True
        )

        # Log something to trigger file creation
        logger.info(EventType.CYCLE_START, "Test message")

        # Check log files exist
        assert logger.json_log_file.exists()
        assert logger.text_log_file.exists()


class TestLoggingMethods:
    """Test logging methods."""

    def test_log_basic(self, tmp_path):
        """Test basic logging."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False,
            file_output=True
        )

        logger.log(
            LogLevel.INFO,
            EventType.CYCLE_START,
            "Cycle started"
        )

        # Read JSON log
        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["level"] == "INFO"
        assert entry["event_type"] == "cycle_start"
        assert entry["message"] == "Cycle started"

    def test_debug_method(self, tmp_path):
        """Test debug method."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False,
            min_level=LogLevel.DEBUG
        )

        logger.debug(EventType.TASK_START, "Debug message")

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["level"] == "DEBUG"

    def test_info_method(self, tmp_path):
        """Test info method."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.info(EventType.TASK_START, "Info message")

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["level"] == "INFO"

    def test_warning_method(self, tmp_path):
        """Test warning method."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.warning(EventType.BUDGET_UPDATE, "Warning message")

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["level"] == "WARNING"

    def test_error_method(self, tmp_path):
        """Test error method."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.error(EventType.ERROR, "Error message")

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["level"] == "ERROR"

    def test_critical_method(self, tmp_path):
        """Test critical method."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.critical(EventType.ERROR, "Critical message")

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["level"] == "CRITICAL"

    def test_min_level_filtering(self, tmp_path):
        """Test that min_level filters out lower level logs."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False,
            min_level=LogLevel.WARNING
        )

        logger.debug(EventType.TASK_START, "Debug - should be filtered")
        logger.info(EventType.TASK_START, "Info - should be filtered")
        logger.warning(EventType.TASK_START, "Warning - should appear")
        logger.error(EventType.TASK_START, "Error - should appear")

        with open(logger.json_log_file) as f:
            lines = f.readlines()

        assert len(lines) == 2  # Only warning and error


class TestSpecializedLoggingMethods:
    """Test specialized logging methods."""

    def test_log_cycle_start(self, tmp_path):
        """Test logging cycle start."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_cycle_start(cycle_number=1, objective="Test objective")

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "cycle_start"
        assert entry["cycle_number"] == 1
        assert entry["metadata"]["objective"] == "Test objective"

    def test_log_cycle_end(self, tmp_path):
        """Test logging cycle end."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_cycle_end(
            cycle_number=1,
            duration_ms=5000.0,
            budget_used=0.05,
            tasks_completed=5
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "cycle_end"
        assert entry["cycle_number"] == 1
        assert entry["duration_ms"] == 5000.0
        assert entry["cost"] == 0.05
        assert entry["metadata"]["tasks_completed"] == 5

    def test_log_task_start(self, tmp_path):
        """Test logging task start."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_task_start(
            task_id="task-123",
            task_type="analyze_data",
            cycle_number=1
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "task_start"
        assert entry["task_id"] == "task-123"
        assert entry["metadata"]["task_type"] == "analyze_data"

    def test_log_task_end(self, tmp_path):
        """Test logging task end."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_task_end(
            task_id="task-123",
            duration_ms=1000.0,
            success=True,
            cost=0.01
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "task_end"
        assert entry["metadata"]["success"] is True

    def test_log_task_end_failure(self, tmp_path):
        """Test logging task failure."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_task_end(
            task_id="task-123",
            duration_ms=500.0,
            success=False,
            error="Task failed"
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["level"] == "ERROR"
        assert entry["metadata"]["success"] is False
        assert entry["metadata"]["error"] == "Task failed"

    def test_log_agent_call(self, tmp_path):
        """Test logging agent call."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_agent_call(
            agent_type="data_analysis",
            duration_ms=2000.0,
            cost=0.005,
            tokens_input=1000,
            tokens_output=500,
            success=True
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "agent_call"
        assert entry["duration_ms"] == 2000.0
        assert entry["cost"] == 0.005
        assert entry["tokens_input"] == 1000
        assert entry["tokens_output"] == 500

    def test_log_api_call(self, tmp_path):
        """Test logging API call."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_api_call(
            endpoint="semantic_scholar",
            duration_ms=500.0,
            status_code=200,
            cost=0.0
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "api_call"
        assert entry["metadata"]["endpoint"] == "semantic_scholar"
        assert entry["metadata"]["status_code"] == 200

    def test_log_error_detailed(self, tmp_path):
        """Test logging detailed error."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_error(
            message="Code execution failed",
            error_type="TimeoutError",
            stack_trace="File 'test.py', line 1\n..."
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "error"
        assert entry["error_type"] == "TimeoutError"
        assert "stack_trace" in entry

    def test_log_budget_update(self, tmp_path):
        """Test logging budget update."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_budget_update(
            cycle_number=1,
            cycle_budget=0.05,
            total_budget=0.15,
            budget_limit=1.0
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "budget_update"
        assert entry["metadata"]["utilization_pct"] == 15.0

    def test_log_hypothesis_generated(self, tmp_path):
        """Test logging hypothesis generation."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_hypothesis_generated(
            hypothesis_id="hyp-123",
            hypothesis_text="Test hypothesis",
            cycle_number=1
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "hypothesis_generated"
        assert entry["hypothesis_id"] == "hyp-123"

    def test_log_hypothesis_tested(self, tmp_path):
        """Test logging hypothesis test."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_hypothesis_tested(
            hypothesis_id="hyp-123",
            outcome="supported",
            confidence=0.85,
            cycle_number=1
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "hypothesis_tested"
        assert entry["metadata"]["outcome"] == "supported"
        assert entry["metadata"]["confidence"] == 0.85

    def test_log_finding_added(self, tmp_path):
        """Test logging finding addition."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_finding_added(
            finding_text="Important finding",
            confidence=0.9,
            cycle_number=1
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "finding_added"
        assert "0.90" in entry["message"]

    def test_log_synthesis(self, tmp_path):
        """Test logging synthesis."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_synthesis(
            cycle_number=1,
            report_path="/reports/report.md",
            findings_count=10,
            hypotheses_count=5
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "synthesis"
        assert entry["metadata"]["findings_count"] == 10
        assert entry["metadata"]["hypotheses_count"] == 5

    def test_log_checkpoint(self, tmp_path):
        """Test logging checkpoint."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.log_checkpoint(
            checkpoint_path="/checkpoints/checkpoint.json",
            cycle_number=1,
            total_budget=0.5
        )

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "checkpoint"
        assert entry["metadata"]["checkpoint_path"] == "/checkpoints/checkpoint.json"


class TestMetrics:
    """Test metrics collection and export."""

    def test_get_metrics_empty(self, tmp_path):
        """Test getting metrics from empty logger."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        metrics = logger.get_metrics()

        assert metrics["total_cost"] == 0.0
        assert metrics["total_api_calls"] == 0
        assert metrics["total_errors"] == 0
        assert metrics["avg_cycle_duration_ms"] == 0

    def test_metrics_updated_on_log(self, tmp_path):
        """Test that metrics are updated when logging."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        # Log some events
        logger.log_agent_call(
            agent_type="test",
            duration_ms=100.0,
            cost=0.01,
            tokens_input=100,
            tokens_output=50,
            success=True
        )

        logger.log_cycle_end(
            cycle_number=1,
            duration_ms=5000.0,
            budget_used=0.05,
            tasks_completed=5
        )

        metrics = logger.get_metrics()

        assert metrics["total_cost"] == pytest.approx(0.06)
        assert metrics["total_api_calls"] == 1
        assert metrics["avg_cycle_duration_ms"] == 5000.0
        assert metrics["avg_agent_call_duration_ms"] == 100.0

    def test_error_count_tracked(self, tmp_path):
        """Test that errors are counted."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        logger.error(EventType.ERROR, "Error 1")
        logger.error(EventType.ERROR, "Error 2")

        metrics = logger.get_metrics()

        assert metrics["total_errors"] == 2

    def test_export_metrics(self, tmp_path):
        """Test exporting metrics to file."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        # Log some data
        logger.log_agent_call(
            agent_type="test",
            duration_ms=100.0,
            cost=0.01,
            tokens_input=100,
            tokens_output=50,
            success=True
        )

        # Export
        output_path = logger.export_metrics()

        # Check file exists and contains expected data
        assert output_path.exists()

        with open(output_path) as f:
            exported = json.load(f)

        assert "total_cost" in exported
        assert exported["total_cost"] == pytest.approx(0.01)

    def test_event_counts(self, tmp_path):
        """Test event counts tracking."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        # Log different event types
        logger.info(EventType.CYCLE_START, "Start 1")
        logger.info(EventType.CYCLE_START, "Start 2")
        logger.info(EventType.TASK_START, "Task 1")

        metrics = logger.get_metrics()

        assert metrics["event_counts"]["cycle_start"] == 2
        assert metrics["event_counts"]["task_start"] == 1


class TestPerformanceTimer:
    """Test PerformanceTimer context manager."""

    def test_timer_basic(self, tmp_path):
        """Test basic timer usage."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        import time

        with PerformanceTimer(logger, EventType.TASK_END, "Task completed"):
            time.sleep(0.01)  # Sleep 10ms

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["event_type"] == "task_end"
        assert entry["duration_ms"] >= 10  # At least 10ms

    def test_timer_with_exception(self, tmp_path):
        """Test timer logs error on exception."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False
        )

        with pytest.raises(ValueError):
            with PerformanceTimer(logger, EventType.TASK_END, "Task"):
                raise ValueError("Test error")

        with open(logger.json_log_file) as f:
            entry = json.loads(f.readline())

        assert entry["level"] == "ERROR"
        assert entry["error_type"] == "ValueError"


class TestGlobalLoggerRegistry:
    """Test global logger registry functions."""

    def test_get_logger_creates_new(self, tmp_path):
        """Test that get_logger creates new logger."""
        # Clear any existing loggers first
        from src.utils.structured_logger import _loggers
        _loggers.clear()

        logger = get_logger("test_registry", log_dir=str(tmp_path))

        assert logger is not None
        assert logger.name == "test_registry"

    def test_get_logger_returns_existing(self, tmp_path):
        """Test that get_logger returns existing logger."""
        from src.utils.structured_logger import _loggers
        _loggers.clear()

        logger1 = get_logger("test_same", log_dir=str(tmp_path))
        logger2 = get_logger("test_same", log_dir=str(tmp_path))

        assert logger1 is logger2

    def test_get_all_metrics(self, tmp_path):
        """Test getting metrics from all loggers."""
        from src.utils.structured_logger import _loggers
        _loggers.clear()

        logger1 = get_logger("metrics_test_1", log_dir=str(tmp_path))
        logger2 = get_logger("metrics_test_2", log_dir=str(tmp_path))

        logger1.info(EventType.CYCLE_START, "Test 1", cost=0.01)
        logger2.info(EventType.CYCLE_START, "Test 2", cost=0.02)

        all_metrics = get_all_metrics()

        assert "metrics_test_1" in all_metrics
        assert "metrics_test_2" in all_metrics


class TestTextOutput:
    """Test text file output."""

    def test_text_output_format(self, tmp_path):
        """Test text output format."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=False,
            file_output=True
        )

        logger.info(
            EventType.TASK_END,
            "Task completed",
            duration_ms=100.0,
            cost=0.01
        )

        with open(logger.text_log_file) as f:
            text = f.read()

        assert "[INFO]" in text
        assert "Task completed" in text
        assert "duration: 100.0ms" in text
        assert "$0.0100" in text


class TestConsoleOutput:
    """Test console output."""

    def test_console_output_stdout(self, tmp_path, capsys):
        """Test console output goes to stdout for non-errors."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=True,
            file_output=False
        )

        logger.info(EventType.CYCLE_START, "Info message")

        captured = capsys.readouterr()
        assert "Info message" in captured.out

    def test_console_output_stderr_for_errors(self, tmp_path, capsys):
        """Test console output goes to stderr for errors."""
        logger = StructuredLogger(
            name="test_logger",
            log_dir=str(tmp_path),
            console_output=True,
            file_output=False
        )

        logger.error(EventType.ERROR, "Error message")

        captured = capsys.readouterr()
        assert "Error message" in captured.err
