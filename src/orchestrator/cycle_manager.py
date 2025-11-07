"""
Orchestrator - Manages discovery cycles and task spawning.

The orchestrator coordinates multiple discovery cycles, tracks progress,
and manages the budget for exploration.
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import anthropic

from src.world_model.graph import NodeType, WorldModel


class TaskStatus(str, Enum):
    """Status of a task in the orchestrator."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BUDGET_EXCEEDED = "budget_exceeded"


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


class TaskDependencyGraph:
    """
    Tracks task dependencies and determines execution order.

    Manages task dependencies and provides methods to determine which tasks
    are ready to execute based on completion status of their dependencies.
    """

    def __init__(self):
        """Initialize the dependency graph."""
        self.dependencies: Dict[str, Set[str]] = {}  # task_id -> set of dependency task_ids
        self.completed: Set[str] = set()  # Set of completed task IDs
        self.all_tasks: Dict[str, Task] = {}  # task_id -> Task object

    def add_task(self, task: Task, depends_on: Optional[List[str]] = None) -> None:
        """
        Add a task to the dependency graph.

        Args:
            task: The task to add
            depends_on: List of task IDs that this task depends on
        """
        self.all_tasks[task.task_id] = task
        self.dependencies[task.task_id] = set(depends_on or [])

    def get_ready_tasks(self) -> List[Task]:
        """
        Get all tasks that are ready to execute.

        A task is ready if all its dependencies have been completed.

        Returns:
            List of Task objects that are ready to execute
        """
        ready = []
        for task_id, deps in self.dependencies.items():
            # Skip if already completed
            if task_id in self.completed:
                continue

            # Check if all dependencies are completed
            if deps.issubset(self.completed):
                ready.append(self.all_tasks[task_id])

        return ready

    def mark_complete(self, task_id: str) -> None:
        """
        Mark a task as completed.

        Args:
            task_id: The ID of the completed task
        """
        self.completed.add(task_id)

    def has_pending_tasks(self) -> bool:
        """
        Check if there are any pending tasks.

        Returns:
            True if there are tasks that haven't been completed
        """
        return len(self.completed) < len(self.all_tasks)

    def get_pending_count(self) -> int:
        """
        Get the number of pending tasks.

        Returns:
            Number of tasks not yet completed
        """
        return len(self.all_tasks) - len(self.completed)


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
        max_cycle_budget: float = 10.0,
        max_total_budget: float = 100.0,
    ):
        """
        Initialize the orchestrator.

        Args:
            world_model: The world model to use for knowledge storage
            max_concurrent_tasks: Maximum number of tasks to run concurrently
            default_budget: Default budget in dollars for cycles
            max_cycle_budget: Maximum budget per cycle in dollars
            max_total_budget: Maximum total budget for all cycles in dollars
        """
        self.world_model = world_model
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_budget = default_budget
        self.max_cycle_budget = max_cycle_budget
        self.max_total_budget = max_total_budget

        # Track cycles and tasks
        self.cycles: Dict[str, Cycle] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.completed_trajectories: List[List[str]] = []

        # Budget tracking
        self.total_budget_used: float = 0.0

        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

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
        Plan initial tasks for a cycle using Claude API for intelligent planning.

        Args:
            cycle: The cycle to plan tasks for

        Returns:
            List of tuples (task_type, objective, context)
        """
        # Get world model context
        world_model_summary = self._get_world_model_summary()

        # Create prompt for Claude
        prompt = f"""Given this research objective and current world model state, create a task decomposition plan.

Research Objective: {cycle.objective}

Current World Model State:
- Total nodes: {world_model_summary['total_nodes']}
- Hypotheses: {world_model_summary['hypothesis_count']}
- Findings: {world_model_summary['finding_count']}
- Papers: {world_model_summary['paper_count']}

Recent Hypotheses:
{self._format_recent_items(world_model_summary['recent_hypotheses'])}

Recent Findings:
{self._format_recent_items(world_model_summary['recent_findings'])}

Available task types:
- ANALYZE_DATA: Analyze a dataset to find patterns and insights
- SEARCH_LITERATURE: Search scientific literature for relevant papers
- GENERATE_HYPOTHESIS: Generate new hypotheses based on current knowledge
- TEST_HYPOTHESIS: Test a specific hypothesis with data or literature
- SYNTHESIZE_FINDINGS: Synthesize current knowledge into coherent insights

Please create a task decomposition plan. For each task, specify:
1. task_type (one of the types above)
2. objective (specific goal for this task)
3. context (relevant parameters as JSON object)

Return your response as a JSON array of tasks in this format:
[
  {{
    "task_type": "SEARCH_LITERATURE",
    "objective": "Search for papers on X",
    "context": {{"max_papers": 10}}
  }},
  ...
]

Create 2-4 tasks that would best advance this research objective."""

        try:
            # Call Claude API
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                # Fallback to simple planning if no API key
                return self._fallback_task_planning(cycle)

            client = anthropic.Anthropic(api_key=api_key)

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ],
            )

            # Extract response text
            response_text = response.content[0].text

            # Parse JSON response
            # Handle potential markdown code blocks
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            task_plan = json.loads(response_text)

            # Convert to internal format and validate
            tasks = []
            for task_dict in task_plan:
                task_type_str = task_dict.get("task_type", "").upper()

                # Validate task type
                try:
                    task_type = TaskType[task_type_str]
                except KeyError:
                    # Skip invalid task types
                    continue

                objective = task_dict.get("objective", "")
                context = task_dict.get("context", {})

                tasks.append((task_type, objective, context))

            # Ensure we have at least one task
            if not tasks:
                return self._fallback_task_planning(cycle)

            return tasks

        except Exception as e:
            # Fallback to simple planning on error
            print(f"Error in Claude-based planning: {e}. Using fallback planning.")
            return self._fallback_task_planning(cycle)

    def _fallback_task_planning(self, cycle: Cycle) -> List[tuple]:
        """
        Fallback task planning using simple keyword matching.

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

    def _get_world_model_summary(self) -> Dict[str, Any]:
        """Get a summary of the current world model state."""
        summary = {
            "total_nodes": self.world_model.graph.number_of_nodes(),
            "hypothesis_count": 0,
            "finding_count": 0,
            "paper_count": 0,
            "recent_hypotheses": [],
            "recent_findings": [],
        }

        # Count node types and collect recent items
        for node_id, data in self.world_model.graph.nodes(data=True):
            node_type = data.get("node_type")

            if node_type == "hypothesis":
                summary["hypothesis_count"] += 1
                if len(summary["recent_hypotheses"]) < 5:
                    summary["recent_hypotheses"].append({
                        "text": data.get("text", ""),
                        "confidence": data.get("confidence", 0.0),
                    })

            elif node_type == "finding":
                summary["finding_count"] += 1
                if len(summary["recent_findings"]) < 5:
                    summary["recent_findings"].append({
                        "text": data.get("text", ""),
                        "confidence": data.get("confidence", 0.0),
                    })

            elif node_type == "paper":
                summary["paper_count"] += 1

        return summary

    def _format_recent_items(self, items: List[Dict[str, Any]]) -> str:
        """Format recent items for display in prompt."""
        if not items:
            return "None"

        formatted = []
        for i, item in enumerate(items[:5], 1):
            text = item.get("text", "")[:100]  # Limit length
            confidence = item.get("confidence", 0.0)
            formatted.append(f"{i}. {text} (confidence: {confidence:.2f})")

        return "\n".join(formatted)

    async def run_cycle(
        self,
        objective: str,
        max_cycles: int = 5,
        max_runtime: float = 43200.0,  # 12 hours in seconds
        max_tasks_per_cycle: int = 10,
        budget: Optional[float] = None,
    ) -> List[Cycle]:
        """
        Run autonomous discovery cycles with recursive spawning.

        This is the main entry point for autonomous discovery. It will:
        1. Spawn an initial cycle
        2. Execute all tasks in the cycle
        3. Check if new cycles should be spawned based on findings
        4. Recursively spawn and execute new cycles
        5. Stop when max_cycles, max_runtime, or budget is exhausted

        Args:
            objective: Initial research objective
            max_cycles: Maximum number of cycles to run
            max_runtime: Maximum runtime in seconds
            max_tasks_per_cycle: Maximum tasks per cycle
            budget: Total budget for all cycles

        Returns:
            List of all cycles that were executed
        """
        start_time = datetime.utcnow()
        all_cycles = []
        cycles_run = 0

        # Set budget if provided
        if budget:
            self.default_budget = budget

        # Spawn and run initial cycle
        current_cycle = await self.spawn_cycle(
            objective=objective,
            max_tasks=max_tasks_per_cycle,
            budget=budget,
        )
        all_cycles.append(current_cycle)
        cycles_run += 1

        # Discovery loop: spawn follow-up cycles as needed
        while cycles_run < max_cycles:
            # Check runtime limit
            elapsed = (datetime.utcnow() - start_time).total_seconds()
            if elapsed > max_runtime:
                print(f"Max runtime ({max_runtime}s) exceeded. Stopping discovery loop.")
                break

            # Check if we should spawn a new cycle
            if not self._should_spawn_new_cycle(current_cycle):
                print("No conditions met for spawning new cycle. Discovery loop complete.")
                break

            # Check budget
            budget_remaining = self.max_total_budget - self.total_budget_used
            if budget_remaining < self.max_cycle_budget:
                print(f"Insufficient budget remaining (${budget_remaining:.2f}). Stopping discovery loop.")
                break

            # Spawn follow-up cycle
            print(f"Spawning follow-up cycle {cycles_run + 1}/{max_cycles}...")
            follow_up_cycle = self._spawn_follow_up_cycle(current_cycle)

            # Plan tasks for the follow-up cycle
            initial_tasks = self._plan_initial_tasks(follow_up_cycle)
            for task_type, task_objective, context in initial_tasks:
                self.create_task(
                    cycle_id=follow_up_cycle.cycle_id,
                    task_type=task_type,
                    objective=task_objective,
                    context=context,
                )

            # Execute the follow-up cycle
            follow_up_cycle.status = TaskStatus.RUNNING
            follow_up_cycle.started_at = datetime.utcnow()
            await self._execute_cycle(follow_up_cycle)

            all_cycles.append(follow_up_cycle)
            current_cycle = follow_up_cycle
            cycles_run += 1

        # Print summary
        total_elapsed = (datetime.utcnow() - start_time).total_seconds()
        print(f"\nDiscovery loop completed:")
        print(f"  Cycles run: {cycles_run}")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"  Total budget used: ${self.total_budget_used:.2f}")

        # Print detailed budget report
        self.print_budget_report()

        return all_cycles

    async def _execute_cycle(self, cycle: Cycle) -> None:
        """
        Execute all tasks in a cycle using parallel execution.

        Args:
            cycle: The cycle to execute
        """
        # Get pending tasks
        pending_tasks = [t for t in cycle.tasks if t.status == TaskStatus.PENDING]

        # Execute tasks in parallel
        await self._execute_tasks_parallel(pending_tasks, cycle)

        # Check if budget was exceeded during execution
        if cycle.budget_used > self.max_cycle_budget:
            cycle.status = TaskStatus.BUDGET_EXCEEDED
            print(f"⚠️  Cycle budget exceeded: ${cycle.budget_used:.2f} > ${self.max_cycle_budget:.2f}")
        else:
            # Mark cycle as completed
            cycle.status = TaskStatus.COMPLETED

        cycle.completed_at = datetime.utcnow()

    async def _execute_tasks_parallel(self, tasks: List[Task], cycle: Cycle) -> List[Dict[str, Any]]:
        """
        Execute tasks in parallel with dependency tracking.

        Args:
            tasks: List of tasks to execute
            cycle: The cycle these tasks belong to

        Returns:
            List of task results
        """
        # Build dependency graph
        # For now, assume no dependencies between tasks (all can run in parallel)
        # In the future, we can extract dependencies from task context
        dep_graph = TaskDependencyGraph()
        for task in tasks:
            # Extract dependencies from task context if they exist
            depends_on = task.context.get("depends_on", [])
            dep_graph.add_task(task, depends_on)

        results = []

        # Execute tasks in waves based on dependencies
        while dep_graph.has_pending_tasks():
            # Get tasks that are ready to execute
            ready_tasks = dep_graph.get_ready_tasks()

            if not ready_tasks:
                # No tasks ready (possible circular dependency or error)
                print("Warning: No tasks ready but pending tasks remain. Breaking.")
                break

            # Create coroutines for ready tasks
            coroutines = [
                self._execute_task_with_semaphore(task, cycle)
                for task in ready_tasks
            ]

            # Execute tasks concurrently
            task_results = await asyncio.gather(*coroutines, return_exceptions=True)

            # Process results and mark completed
            for task, result in zip(ready_tasks, task_results):
                if isinstance(result, Exception):
                    # Handle exception
                    print(f"Task {task.task_id} failed with exception: {result}")
                    task.status = TaskStatus.FAILED
                    task.error = str(result)
                else:
                    results.append(result)

                # Mark task as complete in dependency graph
                dep_graph.mark_complete(task.task_id)

        return results

    async def _execute_task_with_semaphore(self, task: Task, cycle: Cycle) -> Dict[str, Any]:
        """
        Execute a task with semaphore-based concurrency control.

        Args:
            task: The task to execute
            cycle: The cycle this task belongs to

        Returns:
            Task result dictionary
        """
        async with self.semaphore:
            return await self._execute_task(task, cycle)

    async def _execute_task(self, task: Task, cycle: Cycle) -> Dict[str, Any]:
        """
        Execute a single task using the appropriate agent.

        Args:
            task: The task to execute
            cycle: The cycle this task belongs to

        Returns:
            Task result dictionary
        """
        from src.orchestrator.agent_coordinator import AgentCoordinator, TaskResult

        # Mark task as running
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()

        # Add to active tasks
        self.active_tasks[task.task_id] = task

        result: Optional[TaskResult] = None

        try:
            # Initialize agent coordinator
            coordinator = AgentCoordinator()

            # Execute based on task type
            match task.task_type:
                case TaskType.ANALYZE_DATA:
                    result = await coordinator.execute_data_analysis(task, self.world_model)

                case TaskType.SEARCH_LITERATURE:
                    result = await coordinator.execute_literature_search(task, self.world_model)

                case TaskType.GENERATE_HYPOTHESIS:
                    result = await coordinator.execute_hypothesis_generation(task, self.world_model)

                case TaskType.TEST_HYPOTHESIS:
                    result = await coordinator.execute_hypothesis_test(task, self.world_model)

                case TaskType.SYNTHESIZE_FINDINGS:
                    # TODO: Implement synthesis agent
                    result = TaskResult(
                        success=True,
                        task_id=task.task_id,
                        task_type=task.task_type.value,
                        findings=[],
                        cost=0.0,
                        metadata={"status": "not_implemented"},
                    )

                case _:
                    result = TaskResult(
                        success=False,
                        task_id=task.task_id,
                        task_type=task.task_type.value,
                        findings=[],
                        cost=0.0,
                        metadata={},
                        error=f"Unknown task type: {task.task_type}",
                    )

            # Update task based on result
            if result.success:
                task.status = TaskStatus.COMPLETED
                task.result = result.to_dict()
            else:
                task.status = TaskStatus.FAILED
                task.error = result.error
                task.result = result.to_dict()

            # Update budget
            cycle.budget_used += result.cost
            self.total_budget_used += result.cost

        except Exception as e:
            # Handle unexpected errors
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.result = {
                "success": False,
                "error": str(e),
            }

        finally:
            # Mark completion time and remove from active tasks
            task.completed_at = datetime.utcnow()
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

        return task.result or {}

    def _should_spawn_new_cycle(self, current_cycle: Cycle) -> bool:
        """
        Determine if a new discovery cycle should be spawned.

        Checks:
        - New hypotheses added since cycle start
        - Novel findings exceed threshold
        - Budget remaining > minimum cycle budget

        Args:
            current_cycle: The cycle that just completed

        Returns:
            True if a new cycle should be spawned
        """
        # Check if we have budget remaining
        budget_remaining = self.max_total_budget - self.total_budget_used
        if budget_remaining < self.max_cycle_budget:
            print(f"Insufficient budget for new cycle: ${budget_remaining:.2f} remaining")
            return False

        # Count new hypotheses added during this cycle
        new_hypotheses = 0
        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") == "hypothesis":
                node_created = data.get("created_at")
                if node_created and current_cycle.started_at:
                    # Parse datetime if it's a string
                    if isinstance(node_created, str):
                        try:
                            node_created = datetime.fromisoformat(node_created)
                        except (ValueError, AttributeError):
                            continue

                    if node_created >= current_cycle.started_at:
                        new_hypotheses += 1

        # Check for novel findings with high confidence
        novel_findings_count = 0
        novelty_threshold = 0.7

        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") == "finding":
                confidence = data.get("confidence", 0.0)
                novelty = data.get("novelty", 0.0)

                node_created = data.get("created_at")
                if node_created and current_cycle.started_at:
                    # Parse datetime if it's a string
                    if isinstance(node_created, str):
                        try:
                            node_created = datetime.fromisoformat(node_created)
                        except (ValueError, AttributeError):
                            continue

                    if node_created >= current_cycle.started_at and novelty > novelty_threshold:
                        novel_findings_count += 1

        # Spawn new cycle if:
        # - We have new hypotheses to explore, OR
        # - We have novel findings that warrant further investigation
        return new_hypotheses > 0 or novel_findings_count > 0

    def _spawn_follow_up_cycle(self, parent_cycle: Cycle) -> Cycle:
        """
        Spawn a follow-up discovery cycle based on parent cycle results.

        Args:
            parent_cycle: The parent cycle that triggered this spawn

        Returns:
            New Cycle object linked to parent
        """
        # Query world model for recent high-confidence findings
        recent_findings = []
        recent_hypotheses = []

        for node_id, data in self.world_model.graph.nodes(data=True):
            node_created = data.get("created_at")
            if node_created and parent_cycle.started_at:
                # Parse datetime if it's a string
                if isinstance(node_created, str):
                    try:
                        node_created = datetime.fromisoformat(node_created)
                    except (ValueError, AttributeError):
                        continue

                if node_created >= parent_cycle.started_at:
                    if data.get("node_type") == "finding":
                        confidence = data.get("confidence", 0.0)
                        if confidence > 0.6:
                            recent_findings.append({
                                "id": node_id,
                                "text": data.get("text", ""),
                                "confidence": confidence,
                            })
                    elif data.get("node_type") == "hypothesis":
                        recent_hypotheses.append({
                            "id": node_id,
                            "text": data.get("text", ""),
                            "confidence": data.get("confidence", 0.0),
                        })

        # Generate new objective based on findings
        if recent_hypotheses:
            # Focus on testing new hypotheses
            hypothesis_text = recent_hypotheses[0]["text"]
            new_objective = f"Investigate and test hypothesis: {hypothesis_text}"
        elif recent_findings:
            # Build on recent findings
            finding_text = recent_findings[0]["text"]
            new_objective = f"Explore implications of finding: {finding_text}"
        else:
            # Broaden the search
            new_objective = f"Continue investigation from: {parent_cycle.objective}"

        # Create new cycle
        new_cycle = self.create_cycle(
            objective=new_objective,
            max_tasks=10,
        )

        # Link to parent via metadata
        # Store parent cycle ID in the cycle object
        # (We could extend Cycle dataclass to have parent_cycle_id field)
        # For now, we'll just track it in our records
        new_cycle.status = TaskStatus.PENDING

        return new_cycle

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

    def get_budget_report(self) -> Dict[str, Any]:
        """
        Get a detailed budget report.

        Returns:
            Dictionary with budget information:
                - total_spent: Total budget used across all cycles
                - total_budget: Maximum total budget
                - remaining: Budget remaining
                - cycles: List of cycle budget information
        """
        cycles_budget_info = []
        for cycle in self.cycles.values():
            cycles_budget_info.append({
                "cycle_id": cycle.cycle_id,
                "objective": cycle.objective[:50] + "..." if len(cycle.objective) > 50 else cycle.objective,
                "spent": cycle.budget_used,
                "status": cycle.status.value,
                "max_budget": self.max_cycle_budget,
                "exceeded": cycle.budget_used > self.max_cycle_budget,
            })

        return {
            "total_spent": self.total_budget_used,
            "total_budget": self.max_total_budget,
            "remaining": self.max_total_budget - self.total_budget_used,
            "max_cycle_budget": self.max_cycle_budget,
            "cycles": cycles_budget_info,
        }

    def print_budget_report(self) -> None:
        """Print a formatted budget report to console."""
        report = self.get_budget_report()

        print("\n" + "="*60)
        print("BUDGET REPORT")
        print("="*60)
        print(f"Total Budget:     ${report['total_budget']:.2f}")
        print(f"Total Spent:      ${report['total_spent']:.2f}")
        print(f"Remaining:        ${report['remaining']:.2f}")
        print(f"Utilization:      {(report['total_spent'] / report['total_budget'] * 100):.1f}%")
        print(f"\nCycle Budget:     ${report['max_cycle_budget']:.2f} per cycle")
        print(f"Total Cycles:     {len(report['cycles'])}")
        print("\nCycle Breakdown:")
        print("-"*60)

        for i, cycle_info in enumerate(report['cycles'], 1):
            status_marker = "✓" if cycle_info['status'] == "completed" else "⚠️" if cycle_info['exceeded'] else "•"
            print(f"{status_marker} Cycle {i}: ${cycle_info['spent']:.2f} - {cycle_info['status']}")
            print(f"   {cycle_info['objective']}")

        print("="*60 + "\n")

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"Orchestrator(cycles={stats['total_cycles']}, "
            f"active={stats['active_cycles']}, "
            f"tasks={stats['total_tasks']})"
        )
