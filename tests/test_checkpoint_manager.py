"""
Tests for the CheckpointManager module.
"""

import json
import pytest
import asyncio
from pathlib import Path
from datetime import datetime

from src.orchestrator.checkpoint_manager import (
    CheckpointManager,
    TaskCheckpoint,
    CycleCheckpoint,
    OrchestratorCheckpoint,
    CheckpointError,
    CheckpointNotFoundError,
    CheckpointCorruptedError,
)


# ==================== TaskCheckpoint Tests ====================


class TestTaskCheckpoint:
    """Test TaskCheckpoint dataclass."""

    def test_create_task_checkpoint(self):
        """Test creating a task checkpoint."""
        checkpoint = TaskCheckpoint(
            task_id="task-001",
            task_type="analyze_data",
            status="completed",
            objective="Analyze sales data",
            dependencies=["task-000"]
        )

        assert checkpoint.task_id == "task-001"
        assert checkpoint.task_type == "analyze_data"
        assert checkpoint.status == "completed"
        assert checkpoint.objective == "Analyze sales data"
        assert checkpoint.dependencies == ["task-000"]
        assert checkpoint.result is None
        assert checkpoint.cost == 0.0
        assert checkpoint.created_at is not None
        assert checkpoint.completed_at is None
        assert checkpoint.error is None

    def test_create_task_checkpoint_full(self):
        """Test creating a task checkpoint with all fields."""
        checkpoint = TaskCheckpoint(
            task_id="task-002",
            task_type="search_literature",
            status="completed",
            objective="Search papers",
            dependencies=[],
            result={"papers_found": 10},
            cost=0.05,
            completed_at=datetime.now().isoformat(),
            error=None
        )

        assert checkpoint.result == {"papers_found": 10}
        assert checkpoint.cost == 0.05
        assert checkpoint.completed_at is not None


# ==================== CycleCheckpoint Tests ====================


class TestCycleCheckpoint:
    """Test CycleCheckpoint dataclass."""

    def test_create_cycle_checkpoint(self):
        """Test creating a cycle checkpoint."""
        task = TaskCheckpoint(
            task_id="task-001",
            task_type="analyze_data",
            status="completed",
            objective="Test",
            dependencies=[]
        )

        checkpoint = CycleCheckpoint(
            cycle_number=1,
            status="completed",
            objective="Analyze climate data",
            tasks=[task]
        )

        assert checkpoint.cycle_number == 1
        assert checkpoint.status == "completed"
        assert checkpoint.objective == "Analyze climate data"
        assert len(checkpoint.tasks) == 1
        assert checkpoint.budget_used == 0.0
        assert checkpoint.start_time is not None
        assert checkpoint.end_time is None
        assert checkpoint.synthesis_generated is False


# ==================== OrchestratorCheckpoint Tests ====================


class TestOrchestratorCheckpoint:
    """Test OrchestratorCheckpoint dataclass."""

    def test_create_orchestrator_checkpoint_defaults(self):
        """Test creating orchestrator checkpoint with defaults."""
        checkpoint = OrchestratorCheckpoint(
            checkpoint_id="orch_001"
        )

        assert checkpoint.checkpoint_id == "orch_001"
        assert checkpoint.version == "1.0.0"
        assert checkpoint.created_at is not None
        assert checkpoint.research_objective == ""
        assert checkpoint.dataset_path is None
        assert checkpoint.total_cycles == 0
        assert checkpoint.total_budget_used == 0.0
        assert checkpoint.discovery_complete is False
        assert checkpoint.cycles == []
        assert checkpoint.current_cycle_number == 0
        assert checkpoint.world_model_db_path is None
        assert checkpoint.config == {}

    def test_create_orchestrator_checkpoint_full(self):
        """Test creating orchestrator checkpoint with all fields."""
        task = TaskCheckpoint(
            task_id="task-001",
            task_type="analyze_data",
            status="completed",
            objective="Test",
            dependencies=[]
        )

        cycle = CycleCheckpoint(
            cycle_number=1,
            status="completed",
            objective="Test cycle",
            tasks=[task],
            budget_used=0.5
        )

        checkpoint = OrchestratorCheckpoint(
            checkpoint_id="orch_002",
            research_objective="Analyze climate patterns",
            dataset_path="/data/climate.csv",
            total_cycles=1,
            total_budget_used=0.5,
            discovery_complete=False,
            cycles=[cycle],
            current_cycle_number=1,
            world_model_db_path="/data/world_model.db",
            config={"max_cycles": 10}
        )

        assert checkpoint.research_objective == "Analyze climate patterns"
        assert checkpoint.dataset_path == "/data/climate.csv"
        assert checkpoint.total_cycles == 1
        assert len(checkpoint.cycles) == 1
        assert checkpoint.config["max_cycles"] == 10


# ==================== CheckpointManager Tests ====================


class TestCheckpointManagerBasics:
    """Test basic CheckpointManager functionality."""

    def test_create_checkpoint_manager(self, tmp_path):
        """Test creating a checkpoint manager."""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path / "checkpoints")
        )

        assert manager.checkpoint_dir.exists()
        assert manager.auto_checkpoint is True
        assert manager.max_checkpoints == 10
        assert manager.checkpoint_interval == 1

    def test_create_checkpoint_manager_custom(self, tmp_path):
        """Test creating checkpoint manager with custom settings."""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path / "custom"),
            auto_checkpoint=False,
            max_checkpoints=5,
            checkpoint_interval=3
        )

        assert manager.auto_checkpoint is False
        assert manager.max_checkpoints == 5
        assert manager.checkpoint_interval == 3


class TestCheckpointCreation:
    """Test checkpoint creation methods."""

    def test_create_task_checkpoint(self, tmp_path):
        """Test creating task checkpoint via manager."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        task = manager.create_task_checkpoint(
            task_id="task-001",
            task_type="analyze_data",
            status="completed",
            objective="Test task",
            dependencies=["task-000"],
            result={"findings": 5},
            cost=0.02
        )

        assert isinstance(task, TaskCheckpoint)
        assert task.task_id == "task-001"
        assert task.result == {"findings": 5}
        assert task.cost == 0.02
        assert task.completed_at is not None

    def test_create_task_checkpoint_pending(self, tmp_path):
        """Test creating pending task checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        task = manager.create_task_checkpoint(
            task_id="task-001",
            task_type="analyze_data",
            status="pending",
            objective="Test",
            dependencies=[]
        )

        assert task.status == "pending"
        assert task.completed_at is None

    def test_create_cycle_checkpoint(self, tmp_path):
        """Test creating cycle checkpoint via manager."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        task = manager.create_task_checkpoint(
            task_id="task-001",
            task_type="analyze_data",
            status="completed",
            objective="Test",
            dependencies=[]
        )

        cycle = manager.create_cycle_checkpoint(
            cycle_number=1,
            status="completed",
            objective="Test cycle",
            tasks=[task],
            budget_used=0.5,
            synthesis_generated=True
        )

        assert isinstance(cycle, CycleCheckpoint)
        assert cycle.cycle_number == 1
        assert cycle.budget_used == 0.5
        assert cycle.synthesis_generated is True
        assert cycle.end_time is not None

    def test_create_orchestrator_checkpoint(self, tmp_path):
        """Test creating orchestrator checkpoint via manager."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        checkpoint = manager.create_orchestrator_checkpoint(
            checkpoint_id="orch_001",
            research_objective="Test objective",
            dataset_path="/data/test.csv",
            total_cycles=2,
            total_budget_used=1.5,
            discovery_complete=False,
            cycles=[],
            current_cycle_number=2,
            world_model_db_path="/data/wm.db",
            config={"setting": "value"}
        )

        assert isinstance(checkpoint, OrchestratorCheckpoint)
        assert checkpoint.checkpoint_id == "orch_001"
        assert checkpoint.research_objective == "Test objective"
        assert checkpoint.total_budget_used == 1.5


class TestSaveAndLoad:
    """Test saving and loading checkpoints."""

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, tmp_path):
        """Test saving a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        checkpoint = OrchestratorCheckpoint(
            checkpoint_id="test_001",
            research_objective="Test",
            total_budget_used=0.5
        )

        path = await manager.save_checkpoint(checkpoint)

        assert path.exists()
        assert path.suffix == ".json"

    @pytest.mark.asyncio
    async def test_save_checkpoint_custom_name(self, tmp_path):
        """Test saving checkpoint with custom name."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        checkpoint = OrchestratorCheckpoint(checkpoint_id="test_002")

        path = await manager.save_checkpoint(
            checkpoint,
            checkpoint_name="custom_checkpoint.json"
        )

        assert path.name == "custom_checkpoint.json"
        assert path.exists()

    @pytest.mark.asyncio
    async def test_load_checkpoint(self, tmp_path):
        """Test loading a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        # Save a checkpoint
        original = OrchestratorCheckpoint(
            checkpoint_id="load_test",
            research_objective="Test loading",
            total_cycles=3,
            total_budget_used=2.5
        )

        path = await manager.save_checkpoint(original)

        # Load it back
        loaded = await manager.load_checkpoint(path)

        assert loaded is not None
        assert loaded.checkpoint_id == "load_test"
        assert loaded.research_objective == "Test loading"
        assert loaded.total_cycles == 3
        assert loaded.total_budget_used == 2.5

    @pytest.mark.asyncio
    async def test_load_checkpoint_with_cycles(self, tmp_path):
        """Test loading checkpoint with cycles and tasks."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        # Create complex checkpoint
        task = manager.create_task_checkpoint(
            task_id="task-001",
            task_type="analyze_data",
            status="completed",
            objective="Task objective",
            dependencies=[]
        )

        cycle = manager.create_cycle_checkpoint(
            cycle_number=1,
            status="completed",
            objective="Cycle objective",
            tasks=[task]
        )

        original = manager.create_orchestrator_checkpoint(
            checkpoint_id="complex_001",
            research_objective="Complex test",
            dataset_path="/data/test.csv",
            total_cycles=1,
            total_budget_used=0.5,
            discovery_complete=False,
            cycles=[cycle],
            current_cycle_number=1,
            world_model_db_path=None,
            config={}
        )

        # Save and load
        path = await manager.save_checkpoint(original)
        loaded = await manager.load_checkpoint(path)

        assert loaded is not None
        assert len(loaded.cycles) == 1
        assert loaded.cycles[0].cycle_number == 1
        assert len(loaded.cycles[0].tasks) == 1
        assert loaded.cycles[0].tasks[0].task_id == "task-001"

    @pytest.mark.asyncio
    async def test_load_nonexistent_checkpoint(self, tmp_path):
        """Test loading nonexistent checkpoint returns None."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        loaded = await manager.load_checkpoint(
            tmp_path / "nonexistent.json"
        )

        assert loaded is None

    @pytest.mark.asyncio
    async def test_load_latest_checkpoint(self, tmp_path):
        """Test loading the latest checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        # Save multiple checkpoints
        for i in range(3):
            checkpoint = OrchestratorCheckpoint(
                checkpoint_id=f"orch_{i}"
            )
            await manager.save_checkpoint(checkpoint)
            await asyncio.sleep(0.1)  # Ensure different timestamps

        # Load latest
        loaded = await manager.load_checkpoint()

        assert loaded is not None
        assert loaded.checkpoint_id == "orch_2"  # Last saved


class TestCheckpointManagement:
    """Test checkpoint management operations."""

    @pytest.mark.asyncio
    async def test_get_latest_checkpoint(self, tmp_path):
        """Test getting latest checkpoint path."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        # No checkpoints yet
        latest = await manager.get_latest_checkpoint()
        assert latest is None

        # Save some checkpoints
        for i in range(3):
            checkpoint = OrchestratorCheckpoint(checkpoint_id=f"orch_{i}")
            await manager.save_checkpoint(checkpoint)
            await asyncio.sleep(0.1)

        # Get latest
        latest = await manager.get_latest_checkpoint()
        assert latest is not None
        assert latest.exists()

    @pytest.mark.asyncio
    async def test_list_checkpoints(self, tmp_path):
        """Test listing all checkpoints."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        # Save multiple checkpoints with unique names to avoid overwriting
        for i in range(3):
            checkpoint = OrchestratorCheckpoint(checkpoint_id=f"orch_{i}")
            await manager.save_checkpoint(checkpoint, f"checkpoint_{i}.json")

        # List checkpoints
        checkpoints = await manager.list_checkpoints()

        assert len(checkpoints) == 3
        assert all("name" in c for c in checkpoints)
        assert all("path" in c for c in checkpoints)
        assert all("size_bytes" in c for c in checkpoints)
        assert all("modified_at" in c for c in checkpoints)

    @pytest.mark.asyncio
    async def test_delete_checkpoint(self, tmp_path):
        """Test deleting a checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        # Save checkpoint
        checkpoint = OrchestratorCheckpoint(checkpoint_id="to_delete")
        path = await manager.save_checkpoint(checkpoint)

        # Verify it exists
        assert path.exists()

        # Delete it
        result = await manager.delete_checkpoint(path)

        assert result is True
        assert not path.exists()

    @pytest.mark.asyncio
    async def test_delete_nonexistent_checkpoint(self, tmp_path):
        """Test deleting nonexistent checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        result = await manager.delete_checkpoint(
            tmp_path / "nonexistent.json"
        )

        assert result is False


class TestAutoCheckpoint:
    """Test auto checkpoint functionality."""

    @pytest.mark.asyncio
    async def test_should_checkpoint_enabled(self, tmp_path):
        """Test should_checkpoint when enabled."""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            auto_checkpoint=True,
            checkpoint_interval=2
        )

        assert await manager.should_checkpoint(0) is True  # 0 % 2 == 0
        assert await manager.should_checkpoint(1) is False
        assert await manager.should_checkpoint(2) is True
        assert await manager.should_checkpoint(3) is False
        assert await manager.should_checkpoint(4) is True

    @pytest.mark.asyncio
    async def test_should_checkpoint_disabled(self, tmp_path):
        """Test should_checkpoint when disabled."""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            auto_checkpoint=False
        )

        assert await manager.should_checkpoint(0) is False
        assert await manager.should_checkpoint(1) is False
        assert await manager.should_checkpoint(2) is False


class TestCheckpointCleanup:
    """Test checkpoint cleanup functionality."""

    @pytest.mark.asyncio
    async def test_cleanup_old_checkpoints(self, tmp_path):
        """Test that old checkpoints are cleaned up."""
        manager = CheckpointManager(
            checkpoint_dir=str(tmp_path),
            max_checkpoints=3
        )

        # Save more than max_checkpoints
        for i in range(5):
            checkpoint = OrchestratorCheckpoint(checkpoint_id=f"orch_{i}")
            await manager.save_checkpoint(checkpoint)
            await asyncio.sleep(0.1)

        # List remaining checkpoints
        checkpoints = await manager.list_checkpoints()

        # Should have at most max_checkpoints
        assert len(checkpoints) <= 3


class TestExceptions:
    """Test checkpoint exceptions."""

    def test_checkpoint_error(self):
        """Test CheckpointError exception."""
        error = CheckpointError("Test error")
        assert str(error) == "Test error"

    def test_checkpoint_not_found_error(self):
        """Test CheckpointNotFoundError exception."""
        error = CheckpointNotFoundError("Checkpoint not found")
        assert isinstance(error, CheckpointError)

    def test_checkpoint_corrupted_error(self):
        """Test CheckpointCorruptedError exception."""
        error = CheckpointCorruptedError("Checkpoint corrupted")
        assert isinstance(error, CheckpointError)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_saves(self, tmp_path):
        """Test concurrent checkpoint saves."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        # Save multiple checkpoints concurrently
        async def save_checkpoint(i):
            checkpoint = OrchestratorCheckpoint(checkpoint_id=f"concurrent_{i}")
            return await manager.save_checkpoint(checkpoint)

        tasks = [save_checkpoint(i) for i in range(5)]
        paths = await asyncio.gather(*tasks)

        # All should succeed
        assert len(paths) == 5
        assert all(p.exists() for p in paths)

    @pytest.mark.asyncio
    async def test_empty_checkpoint(self, tmp_path):
        """Test saving and loading minimal checkpoint."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        checkpoint = OrchestratorCheckpoint(checkpoint_id="minimal")

        path = await manager.save_checkpoint(checkpoint)
        loaded = await manager.load_checkpoint(path)

        assert loaded is not None
        assert loaded.checkpoint_id == "minimal"
        assert loaded.cycles == []

    @pytest.mark.asyncio
    async def test_checkpoint_file_content(self, tmp_path):
        """Test that checkpoint file contains valid JSON."""
        manager = CheckpointManager(checkpoint_dir=str(tmp_path))

        checkpoint = OrchestratorCheckpoint(
            checkpoint_id="json_test",
            research_objective="Test JSON"
        )

        path = await manager.save_checkpoint(checkpoint)

        # Read file directly
        with open(path) as f:
            data = json.load(f)

        assert data["checkpoint_id"] == "json_test"
        assert data["research_objective"] == "Test JSON"
        assert "version" in data
        assert "created_at" in data
