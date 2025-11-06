"""
Orchestrator - Manages discovery cycles and task spawning.

The orchestrator coordinates multiple discovery cycles, tracks progress,
and manages the budget for exploration.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

from src.world_model.graph import NodeType, WorldModel


class TaskStatus(str, Enum):
    """Status of a task in the orchestrator."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    """Types of tasks that can be spawned."""
    ANALYZE_DATA = "analyze_data"
    SEARCH_LITERATURE = "search_literature"
    GENERATE_HYPOTHESIS = "generate_hypothesis"
    TEST_HYPOTHESIS = "test_hypothesis"
    SYNTHESIZE_FINDINGS = "synthesize_findings"


@dataclass
class Task:
    """Represents a single task in a discovery cycle."""
    task_id: str
    task_type: TaskType
    status: TaskStatus
    objective: str
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    parent_task_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "objective": self.objective,
            "context": self.context,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "parent_task_id": self.parent_task_id,
        }


@dataclass
class Cycle:
    """Represents a discovery cycle."""
    cycle_id: str
    objective: str
    status: TaskStatus
    tasks: List[Task] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    max_tasks: int = 10
    budget_used: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert cycle to dictionary."""
        return {
            "cycle_id": self.cycle_id,
            "objective": self.objective,
            "status": self.status.value,
            "tasks": [task.to_dict() for task in self.tasks],
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "max_tasks": self.max_tasks,
            "budget_used": self.budget_used,
        }


class Orchestrator:
    """
    Orchestrates discovery cycles and manages task execution.

    The orchestrator is responsible for:
    - Spawning discovery cycles
    - Managing task queues
    - Tracking progress and budget
    - Coordinating with the world model
    """

    def __init__(
        self,
        world_model: WorldModel,
        max_concurrent_tasks: int = 3,
        default_budget: float = 100.0,
    ):
        """
        Initialize the orchestrator.

        Args:
            world_model: The world model to use for knowledge storage
            max_concurrent_tasks: Maximum number of tasks to run concurrently
            default_budget: Default budget in dollars for cycles
        """
        self.world_model = world_model
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_budget = default_budget

        # Track cycles and tasks
        self.cycles: Dict[str, Cycle] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.completed_trajectories: List[List[str]] = []

        # Budget tracking
        self.total_budget_used: float = 0.0

    def create_cycle(
        self,
        objective: str,
        max_tasks: int = 10,
        budget: Optional[float] = None,
    ) -> Cycle:
        """
        Create a new discovery cycle.

        Args:
            objective: The high-level objective for this cycle
            max_tasks: Maximum number of tasks for this cycle
            budget: Budget in dollars for this cycle

        Returns:
            A new Cycle object
        """
        cycle = Cycle(
            cycle_id=str(uuid4()),
            objective=objective,
            status=TaskStatus.PENDING,
            max_tasks=max_tasks,
        )

        self.cycles[cycle.cycle_id] = cycle
        return cycle

    def create_task(
        self,
        cycle_id: str,
        task_type: TaskType,
        objective: str,
        context: Optional[Dict[str, Any]] = None,
        parent_task_id: Optional[str] = None,
    ) -> Task:
        """
        Create a new task within a cycle.

        Args:
            cycle_id: The cycle this task belongs to
            task_type: Type of task to create
            objective: Specific objective for this task
            context: Additional context for the task
            parent_task_id: Parent task ID if this is a subtask

        Returns:
            A new Task object
        """
        if cycle_id not in self.cycles:
            raise ValueError(f"Cycle {cycle_id} not found")

        cycle = self.cycles[cycle_id]

        # Check if we've reached max tasks
        if len(cycle.tasks) >= cycle.max_tasks:
            raise ValueError(f"Cycle {cycle_id} has reached max tasks ({cycle.max_tasks})")

        task = Task(
            task_id=str(uuid4()),
            task_type=task_type,
            status=TaskStatus.PENDING,
            objective=objective,
            context=context or {},
            parent_task_id=parent_task_id,
        )

        cycle.tasks.append(task)
        return task

    async def spawn_cycle(
        self,
        objective: str,
        max_tasks: int = 10,
        budget: Optional[float] = None,
    ) -> Cycle:
        """
        Spawn a new discovery cycle and begin execution.

        This is the main entry point for starting autonomous discovery.

        Args:
            objective: The high-level objective for this cycle
            max_tasks: Maximum number of tasks for this cycle
            budget: Budget in dollars for this cycle

        Returns:
            The cycle that was spawned
        """
        # Create the cycle
        cycle = self.create_cycle(objective, max_tasks, budget)

        # Mark cycle as started
        cycle.status = TaskStatus.RUNNING
        cycle.started_at = datetime.utcnow()

        # Create initial tasks based on objective
        # For now, we'll create a simple set of initial tasks
        initial_tasks = self._plan_initial_tasks(cycle)

        for task_type, task_objective, context in initial_tasks:
            self.create_task(
                cycle_id=cycle.cycle_id,
                task_type=task_type,
                objective=task_objective,
                context=context,
            )

        # Execute the cycle (placeholder for now)
        await self._execute_cycle(cycle)

        return cycle

    def _plan_initial_tasks(self, cycle: Cycle) -> List[tuple]:
        """
        Plan initial tasks for a cycle based on its objective.

        This is a simplified version that creates a basic task structure.
        In a full implementation, this would use an LLM to plan tasks.

        Args:
            cycle: The cycle to plan tasks for

        Returns:
            List of tuples (task_type, objective, context)
        """
        objective = cycle.objective.lower()

        tasks = []

        # If objective mentions data or analysis
        if any(word in objective for word in ["data", "analyze", "dataset"]):
            tasks.append((
                TaskType.ANALYZE_DATA,
                f"Analyze data related to: {cycle.objective}",
                {"requires_dataset": True},
            ))

        # If objective mentions literature or papers
        if any(word in objective for word in ["literature", "papers", "research"]):
            tasks.append((
                TaskType.SEARCH_LITERATURE,
                f"Search literature for: {cycle.objective}",
                {"max_papers": 10},
            ))

        # Always generate hypotheses
        tasks.append((
            TaskType.GENERATE_HYPOTHESIS,
            f"Generate hypotheses for: {cycle.objective}",
            {},
        ))

        # If no specific tasks, default to synthesis
        if not tasks:
            tasks.append((
                TaskType.SYNTHESIZE_FINDINGS,
                f"Synthesize current knowledge about: {cycle.objective}",
                {},
            ))

        return tasks

    async def _execute_cycle(self, cycle: Cycle) -> None:
        """
        Execute all tasks in a cycle.

        This is a placeholder implementation that marks tasks as completed.
        In a full implementation, this would actually execute the agents.

        Args:
            cycle: The cycle to execute
        """
        # Create a task queue
        pending_tasks = [t for t in cycle.tasks if t.status == TaskStatus.PENDING]

        # Execute tasks (simplified version)
        for task in pending_tasks:
            await self._execute_task(task)

        # Mark cycle as completed
        cycle.status = TaskStatus.COMPLETED
        cycle.completed_at = datetime.utcnow()

    async def _execute_task(self, task: Task) -> None:
        """
        Execute a single task.

        This is a placeholder that simulates task execution.
        In a full implementation, this would call the appropriate agent.

        Args:
            task: The task to execute
        """
        # Mark task as running
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()

        # Add to active tasks
        self.active_tasks[task.task_id] = task

        # Simulate some work
        await asyncio.sleep(0.1)

        # For now, just mark as completed with dummy result
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        task.result = {
            "status": "success",
            "message": f"Task {task.task_type.value} completed (placeholder)",
            "findings": [],
        }

        # Remove from active tasks
        del self.active_tasks[task.task_id]

    def get_cycle(self, cycle_id: str) -> Optional[Cycle]:
        """Get a cycle by ID."""
        return self.cycles.get(cycle_id)

    def get_active_cycles(self) -> List[Cycle]:
        """Get all active (running) cycles."""
        return [
            cycle for cycle in self.cycles.values()
            if cycle.status == TaskStatus.RUNNING
        ]

    def get_active_hypotheses(self) -> List[Dict[str, Any]]:
        """
        Get all active hypotheses from the world model.

        Returns:
            List of hypothesis nodes that haven't been refuted
        """
        hypotheses = []

        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") == NodeType.HYPOTHESIS.value:
                # Check if it's been refuted
                is_refuted = False
                for _, _, edge_data in self.world_model.graph.in_edges(node_id, data=True):
                    if edge_data.get("edge_type") == "refutes":
                        is_refuted = True
                        break

                if not is_refuted:
                    hypotheses.append({
                        "node_id": node_id,
                        **data,
                    })

        return hypotheses

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the orchestrator.

        Returns:
            Dictionary with orchestrator statistics
        """
        total_tasks = sum(len(cycle.tasks) for cycle in self.cycles.values())
        completed_tasks = sum(
            len([t for t in cycle.tasks if t.status == TaskStatus.COMPLETED])
            for cycle in self.cycles.values()
        )

        return {
            "total_cycles": len(self.cycles),
            "active_cycles": len(self.get_active_cycles()),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "active_tasks": len(self.active_tasks),
            "completed_trajectories": len(self.completed_trajectories),
            "total_budget_used": self.total_budget_used,
            "active_hypotheses": len(self.get_active_hypotheses()),
        }

    def get_cycle_summary(self, cycle_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a summary of a cycle's execution.

        Args:
            cycle_id: The cycle ID

        Returns:
            Dictionary with cycle summary or None if not found
        """
        cycle = self.get_cycle(cycle_id)
        if not cycle:
            return None

        task_stats = {}
        for task in cycle.tasks:
            status = task.status.value
            task_stats[status] = task_stats.get(status, 0) + 1

        return {
            "cycle_id": cycle.cycle_id,
            "objective": cycle.objective,
            "status": cycle.status.value,
            "total_tasks": len(cycle.tasks),
            "task_status": task_stats,
            "budget_used": cycle.budget_used,
            "duration_seconds": (
                (cycle.completed_at - cycle.started_at).total_seconds()
                if cycle.completed_at and cycle.started_at
                else None
            ),
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"Orchestrator(cycles={stats['total_cycles']}, "
            f"active={stats['active_cycles']}, "
            f"tasks={stats['total_tasks']})"
        )
