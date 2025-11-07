#!/usr/bin/env python3
"""
Test script for parallel agent execution.

This script verifies that the parallel execution implementation works correctly.
"""
import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.orchestrator.cycle_manager import (
    Orchestrator,
    Task,
    TaskStatus,
    TaskType,
    TaskDependencyGraph,
)
from src.world_model.graph import WorldModel


async def test_dependency_graph():
    """Test TaskDependencyGraph functionality."""
    print("Testing TaskDependencyGraph...")

    dep_graph = TaskDependencyGraph()

    # Create test tasks
    task1 = Task(
        task_id="task-1",
        task_type=TaskType.ANALYZE_DATA,
        status=TaskStatus.PENDING,
        objective="Task 1",
    )
    task2 = Task(
        task_id="task-2",
        task_type=TaskType.SEARCH_LITERATURE,
        status=TaskStatus.PENDING,
        objective="Task 2",
    )
    task3 = Task(
        task_id="task-3",
        task_type=TaskType.GENERATE_HYPOTHESIS,
        status=TaskStatus.PENDING,
        objective="Task 3",
    )

    # Add tasks with dependencies
    dep_graph.add_task(task1)  # No dependencies
    dep_graph.add_task(task2, depends_on=["task-1"])  # Depends on task1
    dep_graph.add_task(task3, depends_on=["task-1", "task-2"])  # Depends on both

    # Test get_ready_tasks
    ready = dep_graph.get_ready_tasks()
    assert len(ready) == 1, "Should have 1 ready task initially"
    assert ready[0].task_id == "task-1", "Task 1 should be ready"

    # Mark task1 complete
    dep_graph.mark_complete("task-1")

    # Now task2 should be ready
    ready = dep_graph.get_ready_tasks()
    assert len(ready) == 1, "Should have 1 ready task after completing task1"
    assert ready[0].task_id == "task-2", "Task 2 should be ready"

    # Mark task2 complete
    dep_graph.mark_complete("task-2")

    # Now task3 should be ready
    ready = dep_graph.get_ready_tasks()
    assert len(ready) == 1, "Should have 1 ready task after completing task2"
    assert ready[0].task_id == "task-3", "Task 3 should be ready"

    # Test pending status
    assert dep_graph.has_pending_tasks(), "Should have pending tasks"

    # Mark task3 complete
    dep_graph.mark_complete("task-3")

    # Should be no pending tasks
    assert not dep_graph.has_pending_tasks(), "Should have no pending tasks"

    print("✓ TaskDependencyGraph tests passed!")


async def test_orchestrator_parallel():
    """Test that orchestrator can handle parallel execution."""
    print("\nTesting Orchestrator parallel execution...")

    # Create world model
    world_model = WorldModel()

    # Create orchestrator with small concurrency limit
    orchestrator = Orchestrator(
        world_model=world_model,
        max_concurrent_tasks=2,
        default_budget=100.0,
    )

    # Verify semaphore was created
    assert hasattr(orchestrator, 'semaphore'), "Orchestrator should have semaphore"
    assert orchestrator.semaphore._value == 2, "Semaphore should be initialized with max_concurrent_tasks"

    print("✓ Orchestrator parallel execution setup passed!")


async def test_world_model_async():
    """Test async-safe WorldModel methods."""
    print("\nTesting WorldModel async methods...")

    world_model = WorldModel()

    # Test async node addition
    from src.world_model.graph import NodeType
    node_id = await world_model.add_node_async(
        node_type=NodeType.FINDING,
        text="Test finding",
        confidence=0.9,
    )

    assert node_id is not None, "Should return node ID"
    assert node_id in world_model.graph, "Node should be in graph"

    # Test async edge addition
    from src.world_model.graph import EdgeType
    node_id2 = await world_model.add_node_async(
        node_type=NodeType.HYPOTHESIS,
        text="Test hypothesis",
    )

    await world_model.add_edge_async(
        source=node_id2,
        target=node_id,
        edge_type=EdgeType.DERIVES_FROM,
    )

    # Verify edge exists
    assert world_model.graph.has_edge(node_id2, node_id), "Edge should exist"

    print("✓ WorldModel async methods passed!")


async def test_spawn_cycle():
    """Test spawning a cycle with parallel execution."""
    print("\nTesting spawn_cycle with parallel execution...")

    world_model = WorldModel()
    orchestrator = Orchestrator(
        world_model=world_model,
        max_concurrent_tasks=3,
    )

    # Spawn a simple cycle
    cycle = await orchestrator.spawn_cycle(
        objective="Test objective with data and literature",
        max_tasks=5,
    )

    assert cycle is not None, "Should return cycle"
    assert cycle.status == TaskStatus.COMPLETED, "Cycle should complete"
    assert len(cycle.tasks) > 0, "Should have created tasks"

    # All tasks should be completed
    for task in cycle.tasks:
        assert task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED], \
            f"Task {task.task_id} should be completed or failed"

    print(f"✓ spawn_cycle test passed! Created {len(cycle.tasks)} tasks")


async def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Parallel Agent Execution Implementation")
    print("=" * 60)

    try:
        await test_dependency_graph()
        await test_orchestrator_parallel()
        await test_world_model_async()
        await test_spawn_cycle()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        return 0

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
