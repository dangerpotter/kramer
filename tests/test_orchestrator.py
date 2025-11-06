"""
Tests for the Orchestrator class.
"""

import pytest

from src.orchestrator.cycle_manager import (
    Cycle,
    Orchestrator,
    Task,
    TaskStatus,
    TaskType,
)
from src.world_model.graph import NodeType, WorldModel


class TestTaskDataClass:
    """Test the Task dataclass."""

    def test_task_creation(self):
        """Test creating a task."""
        task = Task(
            task_id="test-123",
            task_type=TaskType.ANALYZE_DATA,
            status=TaskStatus.PENDING,
            objective="Analyze climate data",
        )

        assert task.task_id == "test-123"
        assert task.task_type == TaskType.ANALYZE_DATA
        assert task.status == TaskStatus.PENDING
        assert task.objective == "Analyze climate data"
        assert task.result is None
        assert task.error is None

    def test_task_to_dict(self):
        """Test converting task to dictionary."""
        task = Task(
            task_id="test-123",
            task_type=TaskType.ANALYZE_DATA,
            status=TaskStatus.PENDING,
            objective="Test",
            context={"key": "value"},
        )

        task_dict = task.to_dict()
        assert task_dict["task_id"] == "test-123"
        assert task_dict["task_type"] == TaskType.ANALYZE_DATA.value
        assert task_dict["status"] == TaskStatus.PENDING.value
        assert task_dict["context"]["key"] == "value"


class TestCycleDataClass:
    """Test the Cycle dataclass."""

    def test_cycle_creation(self):
        """Test creating a cycle."""
        cycle = Cycle(
            cycle_id="cycle-123",
            objective="Test objective",
            status=TaskStatus.PENDING,
            max_tasks=10,
        )

        assert cycle.cycle_id == "cycle-123"
        assert cycle.objective == "Test objective"
        assert cycle.status == TaskStatus.PENDING
        assert cycle.max_tasks == 10
        assert len(cycle.tasks) == 0

    def test_cycle_to_dict(self):
        """Test converting cycle to dictionary."""
        cycle = Cycle(
            cycle_id="cycle-123",
            objective="Test",
            status=TaskStatus.PENDING,
        )

        cycle_dict = cycle.to_dict()
        assert cycle_dict["cycle_id"] == "cycle-123"
        assert cycle_dict["objective"] == "Test"
        assert cycle_dict["status"] == TaskStatus.PENDING.value
        assert cycle_dict["tasks"] == []


class TestOrchestratorBasics:
    """Test basic Orchestrator functionality."""

    def test_create_orchestrator(self):
        """Test creating an orchestrator."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        assert orch.world_model == wm
        assert orch.max_concurrent_tasks == 3
        assert orch.default_budget == 100.0
        assert len(orch.cycles) == 0
        assert len(orch.active_tasks) == 0

    def test_create_orchestrator_custom_params(self):
        """Test creating orchestrator with custom parameters."""
        wm = WorldModel()
        orch = Orchestrator(
            wm,
            max_concurrent_tasks=5,
            default_budget=200.0,
        )

        assert orch.max_concurrent_tasks == 5
        assert orch.default_budget == 200.0


class TestCycleManagement:
    """Test cycle creation and management."""

    def test_create_cycle(self):
        """Test creating a new cycle."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = orch.create_cycle(
            objective="Test objective",
            max_tasks=5,
        )

        assert cycle.cycle_id is not None
        assert cycle.objective == "Test objective"
        assert cycle.max_tasks == 5
        assert cycle.status == TaskStatus.PENDING
        assert len(cycle.tasks) == 0

        # Check it was added to orchestrator
        assert cycle.cycle_id in orch.cycles
        assert orch.cycles[cycle.cycle_id] == cycle

    def test_get_cycle(self):
        """Test getting a cycle by ID."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = orch.create_cycle("Test")
        retrieved = orch.get_cycle(cycle.cycle_id)

        assert retrieved == cycle

    def test_get_nonexistent_cycle(self):
        """Test getting a cycle that doesn't exist."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        retrieved = orch.get_cycle("nonexistent")
        assert retrieved is None

    def test_get_active_cycles(self):
        """Test getting active cycles."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        # Create several cycles
        cycle1 = orch.create_cycle("Test 1")
        cycle2 = orch.create_cycle("Test 2")
        cycle3 = orch.create_cycle("Test 3")

        # Set different statuses
        cycle1.status = TaskStatus.RUNNING
        cycle2.status = TaskStatus.COMPLETED
        cycle3.status = TaskStatus.RUNNING

        active = orch.get_active_cycles()
        assert len(active) == 2
        assert cycle1 in active
        assert cycle3 in active
        assert cycle2 not in active


class TestTaskManagement:
    """Test task creation and management."""

    def test_create_task(self):
        """Test creating a task within a cycle."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = orch.create_cycle("Test cycle")
        task = orch.create_task(
            cycle_id=cycle.cycle_id,
            task_type=TaskType.ANALYZE_DATA,
            objective="Analyze data",
            context={"dataset": "test.csv"},
        )

        assert task.task_id is not None
        assert task.task_type == TaskType.ANALYZE_DATA
        assert task.objective == "Analyze data"
        assert task.context["dataset"] == "test.csv"
        assert task.status == TaskStatus.PENDING

        # Check it was added to cycle
        assert task in cycle.tasks

    def test_create_task_with_parent(self):
        """Test creating a task with a parent task."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = orch.create_cycle("Test")
        task1 = orch.create_task(
            cycle_id=cycle.cycle_id,
            task_type=TaskType.ANALYZE_DATA,
            objective="Parent task",
        )
        task2 = orch.create_task(
            cycle_id=cycle.cycle_id,
            task_type=TaskType.GENERATE_HYPOTHESIS,
            objective="Child task",
            parent_task_id=task1.task_id,
        )

        assert task2.parent_task_id == task1.task_id

    def test_create_task_invalid_cycle(self):
        """Test creating task with invalid cycle ID."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        with pytest.raises(ValueError):
            orch.create_task(
                cycle_id="nonexistent",
                task_type=TaskType.ANALYZE_DATA,
                objective="Test",
            )

    def test_create_task_max_limit(self):
        """Test that max_tasks limit is enforced."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = orch.create_cycle("Test", max_tasks=2)

        # Create two tasks (should work)
        orch.create_task(
            cycle_id=cycle.cycle_id,
            task_type=TaskType.ANALYZE_DATA,
            objective="Task 1",
        )
        orch.create_task(
            cycle_id=cycle.cycle_id,
            task_type=TaskType.ANALYZE_DATA,
            objective="Task 2",
        )

        # Try to create third task (should fail)
        with pytest.raises(ValueError, match="reached max tasks"):
            orch.create_task(
                cycle_id=cycle.cycle_id,
                task_type=TaskType.ANALYZE_DATA,
                objective="Task 3",
            )


class TestCycleExecution:
    """Test cycle execution."""

    @pytest.mark.asyncio
    async def test_spawn_cycle_basic(self):
        """Test spawning a basic cycle."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = await orch.spawn_cycle(
            objective="Analyze climate data",
            max_tasks=5,
        )

        assert cycle.cycle_id in orch.cycles
        assert cycle.status == TaskStatus.COMPLETED  # Placeholder completes immediately
        assert len(cycle.tasks) > 0  # Should have created initial tasks

    @pytest.mark.asyncio
    async def test_spawn_cycle_creates_tasks(self):
        """Test that spawn_cycle creates appropriate initial tasks."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        # Objective with 'data' should create analyze task
        cycle = await orch.spawn_cycle(
            objective="Analyze climate data",
        )

        # Check that at least one task was created
        assert len(cycle.tasks) > 0

        # Check that tasks have correct status
        for task in cycle.tasks:
            assert task.status == TaskStatus.COMPLETED  # Placeholder completes all

    @pytest.mark.asyncio
    async def test_spawn_cycle_literature_objective(self):
        """Test cycle with literature objective."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = await orch.spawn_cycle(
            objective="Search literature on climate change",
        )

        # Should have created at least one task
        assert len(cycle.tasks) > 0

    @pytest.mark.asyncio
    async def test_spawn_cycle_generic_objective(self):
        """Test cycle with generic objective."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = await orch.spawn_cycle(
            objective="Some random objective",
        )

        # Should create at least a synthesis task
        assert len(cycle.tasks) > 0


class TestHypothesesTracking:
    """Test tracking of active hypotheses."""

    def test_get_active_hypotheses_empty(self):
        """Test getting active hypotheses from empty world model."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        hypotheses = orch.get_active_hypotheses()
        assert len(hypotheses) == 0

    def test_get_active_hypotheses(self):
        """Test getting active hypotheses."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        # Add some hypotheses
        hyp1 = wm.add_hypothesis(text="Hypothesis 1")
        hyp2 = wm.add_hypothesis(text="Hypothesis 2")

        hypotheses = orch.get_active_hypotheses()
        assert len(hypotheses) == 2

        # Check IDs
        hyp_ids = [h["node_id"] for h in hypotheses]
        assert hyp1 in hyp_ids
        assert hyp2 in hyp_ids

    def test_get_active_hypotheses_filters_refuted(self):
        """Test that refuted hypotheses are filtered out."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        # Add hypotheses
        hyp1 = wm.add_hypothesis(text="Active hypothesis")
        hyp2 = wm.add_hypothesis(text="Refuted hypothesis")

        # Add a finding that refutes hyp2
        from src.world_model.graph import EdgeType
        finding = wm.add_finding(text="Refuting finding")
        wm.add_edge(finding, hyp2, EdgeType.REFUTES)

        # Get active hypotheses
        hypotheses = orch.get_active_hypotheses()

        # Should only get hyp1
        assert len(hypotheses) == 1
        assert hypotheses[0]["node_id"] == hyp1


class TestStatistics:
    """Test orchestrator statistics."""

    def test_get_stats_empty(self):
        """Test statistics on empty orchestrator."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        stats = orch.get_stats()

        assert stats["total_cycles"] == 0
        assert stats["active_cycles"] == 0
        assert stats["total_tasks"] == 0
        assert stats["completed_tasks"] == 0
        assert stats["active_tasks"] == 0
        assert stats["total_budget_used"] == 0.0

    @pytest.mark.asyncio
    async def test_get_stats_with_data(self):
        """Test statistics with cycles and tasks."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        # Spawn a cycle
        cycle = await orch.spawn_cycle("Test objective")

        stats = orch.get_stats()

        assert stats["total_cycles"] == 1
        assert stats["total_tasks"] == len(cycle.tasks)
        assert stats["completed_tasks"] == len(cycle.tasks)  # Placeholder completes all

    def test_get_cycle_summary(self):
        """Test getting cycle summary."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = orch.create_cycle("Test", max_tasks=5)
        orch.create_task(
            cycle_id=cycle.cycle_id,
            task_type=TaskType.ANALYZE_DATA,
            objective="Task 1",
        )
        orch.create_task(
            cycle_id=cycle.cycle_id,
            task_type=TaskType.ANALYZE_DATA,
            objective="Task 2",
        )

        summary = orch.get_cycle_summary(cycle.cycle_id)

        assert summary is not None
        assert summary["cycle_id"] == cycle.cycle_id
        assert summary["objective"] == "Test"
        assert summary["total_tasks"] == 2
        assert summary["status"] == TaskStatus.PENDING.value

    def test_get_cycle_summary_nonexistent(self):
        """Test getting summary for nonexistent cycle."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        summary = orch.get_cycle_summary("nonexistent")
        assert summary is None

    def test_repr(self):
        """Test string representation."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        orch.create_cycle("Test 1")
        orch.create_cycle("Test 2")

        repr_str = repr(orch)
        assert "Orchestrator" in repr_str
        assert "cycles=2" in repr_str


class TestTaskPlanning:
    """Test task planning logic."""

    def test_plan_initial_tasks_data_objective(self):
        """Test planning tasks for data analysis objective."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = orch.create_cycle("Analyze climate dataset")
        tasks = orch._plan_initial_tasks(cycle)

        # Should include analyze data task
        task_types = [t[0] for t in tasks]
        assert TaskType.ANALYZE_DATA in task_types

    def test_plan_initial_tasks_literature_objective(self):
        """Test planning tasks for literature objective."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = orch.create_cycle("Search papers on climate change")
        tasks = orch._plan_initial_tasks(cycle)

        # Should include literature search
        task_types = [t[0] for t in tasks]
        assert TaskType.SEARCH_LITERATURE in task_types

    def test_plan_initial_tasks_always_generates_hypotheses(self):
        """Test that hypothesis generation is always planned."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = orch.create_cycle("Test objective")
        tasks = orch._plan_initial_tasks(cycle)

        # Should always include hypothesis generation
        task_types = [t[0] for t in tasks]
        assert TaskType.GENERATE_HYPOTHESIS in task_types

    def test_plan_initial_tasks_generic(self):
        """Test planning for generic objective."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        cycle = orch.create_cycle("Random objective xyz")
        tasks = orch._plan_initial_tasks(cycle)

        # Should create at least one task
        assert len(tasks) > 0


class TestIntegration:
    """Integration tests combining WorldModel and Orchestrator."""

    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test a complete workflow."""
        # Create world model with some initial data
        wm = WorldModel()
        finding = wm.add_finding(
            text="CO2 levels have increased by 40%",
            confidence=0.95,
        )

        # Create orchestrator
        orch = Orchestrator(wm)

        # Spawn cycle
        cycle = await orch.spawn_cycle(
            objective="Analyze climate change trends",
            max_tasks=10,
        )

        # Verify cycle completed
        assert cycle.status == TaskStatus.COMPLETED
        assert len(cycle.tasks) > 0

        # Verify all tasks completed
        for task in cycle.tasks:
            assert task.status == TaskStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_multiple_cycles(self):
        """Test running multiple cycles."""
        wm = WorldModel()
        orch = Orchestrator(wm)

        # Spawn multiple cycles
        cycle1 = await orch.spawn_cycle("Objective 1")
        cycle2 = await orch.spawn_cycle("Objective 2")

        # Both should complete
        assert cycle1.status == TaskStatus.COMPLETED
        assert cycle2.status == TaskStatus.COMPLETED

        # Stats should reflect both cycles
        stats = orch.get_stats()
        assert stats["total_cycles"] == 2
