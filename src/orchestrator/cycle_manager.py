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
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from uuid import uuid4

import anthropic

from src.world_model.graph import NodeType, WorldModel
from src.reporting.cycle_report_generator import CycleReportGenerator, CycleReportContent
from src.reporting.report_generator import ReportGenerator

# Budget reservation for final report generation
FINAL_REPORT_BUDGET_RESERVE = 0.25  # Reserve $0.25 for final report


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


class CycleType(str, Enum):
    """Types of discovery cycles."""
    EXPLORATION = "exploration"     # Normal exploration cycle
    SYNTHESIS = "synthesis"         # Meta-analysis and synthesis cycle
    PIVOT = "pivot"                 # Redirecting to unexplored aspects
    # TODO: Future cycle types to consider:
    # VALIDATION = "validation"     # Focused on validating existing hypotheses
    # DEEP_DIVE = "deep_dive"       # Deep exploration of a specific topic
    # CONTRADICTION = "contradiction"  # Resolving conflicting findings


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
    cycle_type: CycleType = CycleType.EXPLORATION  # Type of cycle
    parent_cycle_id: Optional[str] = None  # For tracking cycle lineage
    metadata: Dict[str, Any] = field(default_factory=dict)  # Synthesis context and other metadata

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
            "cycle_type": self.cycle_type.value,
            "parent_cycle_id": self.parent_cycle_id,
            "metadata": self.metadata,
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
        auto_synthesize: bool = True,
        synthesis_interval: Optional[int] = None,
        synthesis_threshold: float = 0.8,
        output_dir: str = "outputs",
    ):
        """
        Initialize the orchestrator.

        Args:
            world_model: The world model to use for knowledge storage
            max_concurrent_tasks: Maximum number of tasks to run concurrently
            default_budget: Default budget in dollars for cycles
            max_cycle_budget: Maximum budget per cycle in dollars
            max_total_budget: Maximum total budget for all cycles in dollars
            auto_synthesize: Whether to automatically trigger synthesis
            synthesis_interval: Synthesize every N cycles, or None for auto
            synthesis_threshold: Confidence threshold for completion detection
            output_dir: Directory for report outputs
        """
        self.world_model = world_model
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_budget = default_budget
        self.max_cycle_budget = max_cycle_budget
        self.max_total_budget = max_total_budget
        self.auto_synthesize = auto_synthesize
        self.synthesis_interval = synthesis_interval
        self.synthesis_threshold = synthesis_threshold
        self.output_dir = output_dir

        # Track cycles and tasks
        self.cycles: Dict[str, Cycle] = {}
        self.active_tasks: Dict[str, Task] = {}
        self.completed_trajectories: List[List[str]] = []

        # Budget tracking
        self.total_budget_used: float = 0.0

        # Synthesis tracking
        self.synthesis_results: List[Dict[str, Any]] = []

        # Completion tracking (for spawn decisions)
        self.last_completion_score: Optional[float] = None

        # Concurrency control
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

        # Discovery tracking (set when running)
        self.discovery_id: Optional[str] = None
        self.persistence_service = None

        # Cycle report tracking
        self.cycle_reports: List[CycleReportContent] = []

    def set_discovery_context(
        self,
        discovery_id: str,
        persistence_service: Any = None,
    ) -> None:
        """
        Set the discovery context for cycle report persistence.

        Args:
            discovery_id: The discovery ID for this run
            persistence_service: The persistence service for saving reports
        """
        self.discovery_id = discovery_id
        self.persistence_service = persistence_service

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

        # Ensure context includes cycle objective for hypothesis generation
        task_context = context or {}
        if task_type == TaskType.GENERATE_HYPOTHESIS and "objective" not in task_context:
            task_context["objective"] = cycle.objective

        task = Task(
            task_id=str(uuid4()),
            task_type=task_type,
            status=TaskStatus.PENDING,
            objective=objective,
            context=task_context,
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

        # Get previous cycle context
        previous_cycle_context = self._get_previous_cycle_summaries()

        # Use specialized planning for synthesis cycles
        if cycle.cycle_type == CycleType.SYNTHESIS:
            return self._plan_synthesis_tasks(cycle, world_model_summary, previous_cycle_context)

        # Create prompt for Claude
        prompt = f"""Given this research objective and current world model state, create a task decomposition plan.

Research Objective: {cycle.objective}

{previous_cycle_context}

Current World Model State:
- Total nodes: {world_model_summary['total_nodes']}
- Hypotheses: {world_model_summary['hypothesis_count']}
- Findings: {world_model_summary['finding_count']}
- Papers: {world_model_summary['paper_count']}

Recent Hypotheses (with their IDs for TEST_HYPOTHESIS tasks):
{self._format_recent_items(world_model_summary['recent_hypotheses'], include_id=True)}

Recent Findings:
{self._format_recent_items(world_model_summary['recent_findings'])}

Available task types:
- ANALYZE_DATA: Analyze a dataset to find patterns and insights
- SEARCH_LITERATURE: Search scientific literature for relevant papers
- GENERATE_HYPOTHESIS: Generate new hypotheses based on current knowledge
- TEST_HYPOTHESIS: Test a specific hypothesis with data or literature (REQUIRES hypothesis_id from the list above)
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
  {{
    "task_type": "TEST_HYPOTHESIS",
    "objective": "Test hypothesis about X",
    "context": {{"hypothesis_id": "actual-uuid-from-list-above"}}
  }}
]

IMPORTANT: For TEST_HYPOTHESIS tasks, you MUST use an actual hypothesis ID (UUID) from the "Recent Hypotheses" list above, not the hypothesis text or a number.

Create 2-4 tasks that would best advance this research objective."""

        try:
            # Call Claude API
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                # Fallback to simple planning if no API key
                return self._fallback_task_planning(cycle)

            client = anthropic.Anthropic(api_key=api_key)

            response = client.messages.create(
                model=os.getenv("CLAUDE_MODEL"),
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

                objective = str(task_dict.get("objective", ""))
                context = task_dict.get("context", {}) if isinstance(task_dict.get("context"), dict) else {}

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

    def _plan_synthesis_tasks(
        self,
        cycle: Cycle,
        world_model_summary: Dict[str, Any],
        previous_cycle_context: str
    ) -> List[tuple]:
        """
        Plan tasks for a synthesis cycle using Claude for intelligent planning.

        Synthesis cycles focus on:
        1. Consolidating findings into coherent insights
        2. Identifying and reconciling contradictions
        3. Analyzing coverage gaps vs original objective
        4. Generating new hypotheses based on synthesis
        5. Suggesting pivot directions for future exploration

        Args:
            cycle: The synthesis cycle to plan for
            world_model_summary: Current world model state
            previous_cycle_context: Summary of previous cycles

        Returns:
            List of tuples (task_type, objective, context)
        """
        # Get synthesis context from cycle metadata
        contradictions = getattr(cycle, 'metadata', {}).get('contradictions', [])
        coverage_analysis = getattr(cycle, 'metadata', {}).get('coverage_analysis', {})
        original_objective = getattr(cycle, 'metadata', {}).get('original_objective', '')

        # If no metadata, compute fresh
        if not contradictions and not coverage_analysis:
            contradictions = self._detect_contradictory_findings()
            coverage_analysis = self._analyze_objective_coverage()
            original_objective = self._get_original_objective()

        # Try Claude-based planning first
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                tasks = self._claude_plan_synthesis_tasks(
                    cycle, world_model_summary, contradictions,
                    coverage_analysis, original_objective, previous_cycle_context
                )
                if tasks:
                    return tasks
        except Exception as e:
            print(f"Warning: Claude synthesis task planning failed: {e}")

        # Fallback to rule-based planning
        return self._fallback_synthesis_task_planning(
            cycle, world_model_summary, contradictions, coverage_analysis
        )

    def _claude_plan_synthesis_tasks(
        self,
        cycle: Cycle,
        world_model_summary: Dict[str, Any],
        contradictions: List[Dict[str, Any]],
        coverage_analysis: Dict[str, Any],
        original_objective: Optional[str],
        previous_cycle_context: str,
    ) -> Optional[List[tuple]]:
        """
        Use Claude to intelligently plan synthesis tasks.

        Args:
            cycle: The synthesis cycle
            world_model_summary: Current world model state
            contradictions: Detected contradictions
            coverage_analysis: Coverage analysis results
            original_objective: The original research objective
            previous_cycle_context: Summary of previous cycles

        Returns:
            List of task tuples, or None if planning fails
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        # Format contradictions
        contradictions_text = "None detected"
        if contradictions:
            contradictions_text = "\n".join([
                f"- Finding A: \"{c['finding1_text'][:80]}...\"\n  Finding B: \"{c['finding2_text'][:80]}...\""
                for c in contradictions[:5]
            ])

        prompt = f"""Plan tasks for a synthesis cycle in this research discovery process.

SYNTHESIS CYCLE OBJECTIVE:
{cycle.objective}

ORIGINAL RESEARCH OBJECTIVE:
{original_objective or 'Not specified'}

CURRENT KNOWLEDGE STATE:
- Total findings: {world_model_summary['finding_count']}
- Total hypotheses: {world_model_summary['hypothesis_count']}
- Papers reviewed: {world_model_summary['paper_count']}

COVERAGE ANALYSIS:
- Overall coverage of objective: {coverage_analysis.get('overall_coverage', 0):.0%}
- Coverage gap: {coverage_analysis.get('coverage_gap', 0):.0%}
- Aspects covered: {coverage_analysis.get('aspects_covered', 0)}

CONTRADICTORY FINDINGS TO RECONCILE:
{contradictions_text}

RECENT HYPOTHESES (with IDs for TEST_HYPOTHESIS tasks):
{self._format_recent_items(world_model_summary['recent_hypotheses'], include_id=True)}

RECENT FINDINGS:
{self._format_recent_items(world_model_summary['recent_findings'])}

{previous_cycle_context}

AVAILABLE TASK TYPES:
- SYNTHESIZE_FINDINGS: Consolidate knowledge, identify patterns, draw conclusions
- GENERATE_HYPOTHESIS: Create new hypotheses based on synthesis insights
- TEST_HYPOTHESIS: Test a specific hypothesis (requires hypothesis_id from list above)
- SEARCH_LITERATURE: Search for papers on specific topics (for filling gaps)
- ANALYZE_DATA: Analyze data to resolve contradictions or fill gaps

PLANNING GUIDELINES:
1. If contradictions exist: Include tasks to reconcile them (search for evidence, test competing hypotheses)
2. If coverage gaps exist: Include tasks to explore under-covered aspects
3. Always include a primary synthesis task
4. Generate new hypotheses if synthesis reveals new patterns
5. Prioritize high-impact tasks that advance the original objective

Create 3-6 tasks that best accomplish this synthesis cycle's objectives.

Return as JSON array:
[
  {{"task_type": "TASK_TYPE", "objective": "specific objective", "context": {{}}}}
]

For TEST_HYPOTHESIS tasks, use actual hypothesis IDs from the list above."""

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=os.getenv("CLAUDE_MODEL"),
            max_tokens=2000,
            temperature=0.5,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = response.content[0].text

        # Parse JSON
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        task_plan = json.loads(response_text)

        # Convert to internal format
        tasks = []
        for task_dict in task_plan:
            task_type_str = task_dict.get("task_type", "").upper()
            try:
                task_type = TaskType[task_type_str]
            except KeyError:
                continue

            objective = str(task_dict.get("objective", ""))
            context = task_dict.get("context", {}) if isinstance(task_dict.get("context"), dict) else {}

            tasks.append((task_type, objective, context))

        return tasks if tasks else None

    def _fallback_synthesis_task_planning(
        self,
        cycle: Cycle,
        world_model_summary: Dict[str, Any],
        contradictions: List[Dict[str, Any]],
        coverage_analysis: Dict[str, Any],
    ) -> List[tuple]:
        """
        Rule-based fallback for synthesis task planning.

        Args:
            cycle: The synthesis cycle
            world_model_summary: Current world model state
            contradictions: Detected contradictions
            coverage_analysis: Coverage analysis results

        Returns:
            List of task tuples
        """
        tasks = []

        # Primary task: Synthesize all current findings
        tasks.append((
            TaskType.SYNTHESIZE_FINDINGS,
            f"Meta-analysis: Synthesize all findings and identify key insights for: {cycle.objective}",
            {
                "synthesis_type": "comprehensive",
                "include_confidence_analysis": True,
                "contradictions_to_address": len(contradictions),
                "coverage_gap": coverage_analysis.get("coverage_gap", 0),
            },
        ))

        # Add contradiction reconciliation tasks
        if contradictions:
            # Search for additional evidence to resolve top contradiction
            top_contradiction = contradictions[0]
            tasks.append((
                TaskType.SEARCH_LITERATURE,
                f"Search for evidence to resolve contradiction between findings about: "
                f"{top_contradiction['finding1_text'][:50]}... vs {top_contradiction['finding2_text'][:50]}...",
                {
                    "purpose": "contradiction_resolution",
                    "max_papers": 5,
                },
            ))

        # Add coverage gap tasks
        if coverage_analysis.get("coverage_gap", 0) > 0.4:
            original_objective = self._get_original_objective() or cycle.objective
            tasks.append((
                TaskType.SEARCH_LITERATURE,
                f"Search for papers covering under-explored aspects of: {original_objective[:80]}",
                {
                    "purpose": "coverage_gap",
                    "max_papers": 5,
                },
            ))

        # Generate hypotheses if we have enough findings
        if world_model_summary["finding_count"] > 3 and world_model_summary["hypothesis_count"] < 5:
            tasks.append((
                TaskType.GENERATE_HYPOTHESIS,
                "Generate hypotheses based on synthesized findings and identified patterns",
                {
                    "focus": "synthesis_derived",
                    "min_novelty": 0.5,
                    "consider_contradictions": len(contradictions) > 0,
                },
            ))

        # Test highest confidence untested hypothesis
        if world_model_summary["recent_hypotheses"]:
            sorted_hypotheses = sorted(
                world_model_summary["recent_hypotheses"],
                key=lambda h: h.get("confidence", 0),
                reverse=True
            )
            if sorted_hypotheses:
                top_hypothesis = sorted_hypotheses[0]
                tasks.append((
                    TaskType.TEST_HYPOTHESIS,
                    f"Test high-priority hypothesis: {top_hypothesis['text'][:80]}",
                    {"hypothesis_id": top_hypothesis["id"]},
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
                # Only include hypotheses that haven't been conclusively tested
                metadata = data.get("metadata", {})
                test_outcome = metadata.get("test_outcome", "")
                if test_outcome not in ["supported", "refuted"]:
                    if len(summary["recent_hypotheses"]) < 5:
                        summary["recent_hypotheses"].append({
                            "id": node_id,  # Include hypothesis ID for TEST_HYPOTHESIS tasks
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

    def _format_recent_items(self, items: List[Dict[str, Any]], include_id: bool = False) -> str:
        """Format recent items for display in prompt."""
        if not items:
            return "None"

        formatted = []
        for i, item in enumerate(items[:5], 1):
            text = item.get("text", "")[:100]  # Limit length
            confidence = item.get("confidence", 0.0)
            item_id = item.get("id", "")
            if include_id and item_id:
                formatted.append(f"- ID: {item_id}\n  Text: {text} (confidence: {confidence:.2f})")
            else:
                formatted.append(f"{i}. {text} (confidence: {confidence:.2f})")

        return "\n".join(formatted)

    def _get_previous_cycle_summaries(self, max_cycles: int = 3) -> str:
        """
        Get compact summaries from previous cycles for planning context.

        Args:
            max_cycles: Maximum number of previous cycle summaries to include

        Returns:
            Formatted string with previous cycle summaries, or empty string if none
        """
        if not self.cycle_reports:
            return ""

        # Get the most recent summaries (up to max_cycles)
        recent_reports = self.cycle_reports[-max_cycles:]

        if not recent_reports:
            return ""

        lines = [
            "Previous Cycle Summaries:",
            "(Use this context to inform task planning - avoid redundant work)"
        ]

        for report in recent_reports:
            lines.append("")
            lines.append(report.summary)

        return "\n".join(lines)

    async def run_cycle(
        self,
        objective: str,
        max_cycles: int = 5,
        max_runtime: float = 43200.0,  # 12 hours in seconds
        max_tasks_per_cycle: int = 10,
        budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Run autonomous discovery cycles with recursive spawning.

        This is the main entry point for autonomous discovery. It will:
        1. Spawn an initial cycle
        2. Execute all tasks in the cycle
        3. Check if new cycles should be spawned based on findings
        4. Recursively spawn and execute new cycles
        5. Stop when max_cycles, max_runtime, or budget is exhausted
        6. Automatically synthesize findings based on configuration

        Args:
            objective: Initial research objective
            max_cycles: Maximum number of cycles to run
            max_runtime: Maximum runtime in seconds
            max_tasks_per_cycle: Maximum tasks per cycle
            budget: Total budget for all cycles

        Returns:
            Dictionary containing:
                - cycles: List of all cycles that were executed
                - cycles_completed: Number of cycles completed
                - synthesis_reports: List of synthesis results
                - total_cost: Total budget used
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

        # Generate cycle report for initial cycle
        await self._generate_cycle_report(current_cycle, cycles_run)

        # Check for synthesis trigger after initial cycle
        if self.auto_synthesize and self._should_trigger_synthesis(cycles_run - 1):
            print(f"Triggering synthesis at cycle {cycles_run}")
            await self._execute_synthesis_task(cycles_run - 1)

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

            # Check budget - use minimum viable threshold instead of full cycle budget
            budget_remaining = self.max_total_budget - self.total_budget_used
            MIN_VIABLE_BUDGET = 0.10  # $0.10 minimum to attempt another cycle
            if budget_remaining < MIN_VIABLE_BUDGET:
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

            # Generate cycle report for follow-up cycle
            await self._generate_cycle_report(follow_up_cycle, cycles_run)

            # Check for synthesis trigger after this cycle
            if self.auto_synthesize and self._should_trigger_synthesis(cycles_run - 1):
                print(f"Triggering synthesis at cycle {cycles_run}")
                await self._execute_synthesis_task(cycles_run - 1)

                # Check if this was final synthesis (objective complete)
                if self._check_objective_completion():
                    print("Objective complete - ending discovery loop")
                    break

        # Final synthesis if we haven't done one yet
        if self.auto_synthesize and not self.synthesis_results:
            print(f"Triggering final synthesis at end of discovery loop")
            await self._execute_synthesis_task(cycles_run - 1)

        # Print summary
        total_elapsed = (datetime.utcnow() - start_time).total_seconds()
        print(f"\nDiscovery loop completed:")
        print(f"  Cycles run: {cycles_run}")
        print(f"  Total time: {total_elapsed:.1f}s")
        print(f"  Total budget used: ${self.total_budget_used:.2f}")

        if self.synthesis_results:
            print(f"  Synthesis reports generated: {len(self.synthesis_results)}")
            for i, result in enumerate(self.synthesis_results, 1):
                report_path = result.get("metadata", {}).get("report_path", "unknown")
                print(f"    {i}. {report_path}")

        # Print detailed budget report
        self.print_budget_report()

        # Generate final report
        final_report = await self._generate_final_report(objective)

        return {
            "cycles": all_cycles,
            "cycles_completed": cycles_run,
            "synthesis_reports": self.synthesis_results,
            "total_cost": self.total_budget_used,
            "final_report": final_report,
        }

    async def _generate_final_report(self, objective: str) -> Optional[Dict[str, Any]]:
        """
        Generate the final discovery report.

        Args:
            objective: The research objective

        Returns:
            Dictionary containing report metadata and path, or None if generation failed
        """
        try:
            print("\n📝 Generating final report...")

            generator = ReportGenerator(
                world_model=self.world_model,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                min_confidence=0.7,
            )

            output_path = Path(self.output_dir) / "final_report.md"

            result = generator.generate_report(
                output_path=output_path,
                generate_narratives=True,
                include_appendix=True,
            )

            # Track the cost
            report_cost = result.get("cost", 0.0)
            self.total_budget_used += report_cost

            print(f"✅ Final report generated: {output_path}")
            print(f"   Report cost: ${report_cost:.2f}")
            print(f"   Total cost (including report): ${self.total_budget_used:.2f}")

            # Save final report to database for UI access
            if self.persistence_service and self.discovery_id:
                try:
                    # Read the generated report content
                    with open(output_path, "r", encoding="utf-8") as f:
                        full_content = f.read()

                    # Extract summary from first non-heading paragraph
                    lines = full_content.split('\n')
                    summary = "Final discovery report with complete findings and conclusions"
                    for line in lines:
                        if line.strip() and not line.startswith('#'):
                            summary = line.strip()[:300]
                            break

                    # Save to database with cycle_id=None to indicate final report
                    await self.persistence_service.save_cycle_report(
                        discovery_id=self.discovery_id,
                        cycle_id=None,
                        summary=summary,
                        full_content=full_content,
                        tasks_completed=0,
                        findings_count=result.get("total_findings", 0),
                        hypotheses_count=0,
                        papers_count=0,
                        budget_used=self.total_budget_used,
                        generation_cost=report_cost,
                    )
                    print("   📊 Final report saved to database")
                except Exception as db_err:
                    print(f"   ⚠️  Failed to save final report to database: {db_err}")

            return {
                "report_path": str(output_path),
                "cost": report_cost,
                "discovery_count": result.get("discovery_count", 0),
                "total_findings": result.get("total_findings", 0),
            }

        except Exception as e:
            print(f"⚠️  Failed to generate final report: {e}")
            return None

    async def _execute_cycle(self, cycle: Cycle) -> None:
        """
        Execute all tasks in a cycle using parallel execution.

        Supports dynamic task scheduling - if tasks add new tasks during execution,
        those will be executed in subsequent waves.

        Args:
            cycle: The cycle to execute
        """
        max_waves = 10  # Prevent infinite loops
        wave_count = 0

        while wave_count < max_waves:
            # Get pending tasks
            pending_tasks = [t for t in cycle.tasks if t.status == TaskStatus.PENDING]

            if not pending_tasks:
                # No more pending tasks
                break

            print(f"Executing wave {wave_count + 1} with {len(pending_tasks)} tasks")

            # Execute tasks in parallel
            await self._execute_tasks_parallel(pending_tasks, cycle)

            wave_count += 1

            # Check if budget was exceeded during execution
            if cycle.budget_used > self.max_cycle_budget:
                cycle.status = TaskStatus.BUDGET_EXCEEDED
                print(f"⚠️  Cycle budget exceeded: ${cycle.budget_used:.2f} > ${self.max_cycle_budget:.2f}")
                break

        # Mark cycle as completed if not already failed or budget exceeded
        if cycle.status == TaskStatus.RUNNING:
            cycle.status = TaskStatus.COMPLETED

        cycle.completed_at = datetime.utcnow()

    async def _generate_cycle_report(
        self,
        cycle: Cycle,
        cycle_number: int,
    ) -> Optional[CycleReportContent]:
        """
        Generate and save a cycle report after cycle completion.

        Args:
            cycle: The completed cycle
            cycle_number: The cycle number (1-indexed)

        Returns:
            The generated cycle report content, or None if generation failed
        """
        try:
            generator = CycleReportGenerator(
                world_model=self.world_model,
                anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
                model=os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514"),
            )

            # Count completed tasks
            tasks_completed = sum(
                1 for t in cycle.tasks
                if t.status == TaskStatus.COMPLETED
            )

            # Generate the report
            report = await generator.generate_cycle_report(
                cycle_id=cycle.cycle_id,
                cycle_number=cycle_number,
                objective=cycle.objective,
                cycle_started_at=cycle.started_at or cycle.created_at,
                cycle_completed_at=cycle.completed_at,
                tasks_completed=tasks_completed,
                budget_used=cycle.budget_used,
            )

            # Track generation cost
            self.total_budget_used += report.generation_cost
            cycle.budget_used += report.generation_cost

            # Save to file
            output_path = Path(self.output_dir)
            generator.save_report_to_file(report, output_path, cycle_number)

            # Save to database if persistence service available
            if self.persistence_service and self.discovery_id:
                # Save cycle and report in the same transaction to avoid FK violations
                await self.persistence_service.save_cycle_with_report(
                    discovery_id=self.discovery_id,
                    cycle_id=cycle.cycle_id,
                    cycle_number=cycle_number,
                    objective=cycle.objective,
                    cycle_status=cycle.status.value if hasattr(cycle.status, 'value') else str(cycle.status),
                    budget_used=cycle.budget_used,
                    started_at=cycle.started_at,
                    completed_at=cycle.completed_at,
                    report_summary=report.summary,
                    report_full_content=report.full_content,
                    tasks_completed=report.tasks_completed,
                    findings_count=report.findings_count,
                    hypotheses_count=report.hypotheses_count,
                    papers_count=report.papers_count,
                    generation_cost=report.generation_cost,
                )

            # Store for LLM context
            self.cycle_reports.append(report)

            print(f"📊 Generated cycle {cycle_number} report (cost: ${report.generation_cost:.2f})")
            return report

        except Exception as e:
            print(f"⚠️  Failed to generate cycle report: {e}")
            return None

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

        # Log task start
        objective_str = str(task.objective) if task.objective else ""
        raw_info = task.context.get("hypothesis_id", objective_str[:50] if objective_str else "")
        task_info = str(raw_info)[:50] if raw_info else ""
        print(f"  [TASK] Starting {task.task_type.value}: {task_info}...")

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
                    from src.reporting.report_generator import ReportGenerator, ReportConfig

                    # Extract parameters
                    output_dir = task.context.get("output_dir", self.output_dir)
                    report_name = task.context.get("report_name", f"discovery_report_{task.task_id[:8]}")
                    min_confidence = task.context.get("min_confidence", 0.7)
                    generate_narratives = task.context.get("generate_narratives", True)
                    multi_report = task.context.get("multi_report", False)

                    # Create generator
                    generator = ReportGenerator(
                        world_model=self.world_model,
                        min_confidence=min_confidence,
                        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") if generate_narratives else None,
                    )

                    # Generate report(s)
                    if multi_report:
                        # Generate multiple reports with different focuses
                        report_configs = task.context.get("report_configs")  # Optional custom configs
                        report_result = generator.generate_multiple_reports(
                            output_dir=Path(output_dir),
                            report_configs=report_configs,
                            generate_narratives=generate_narratives,
                            include_appendix=True,
                        )

                        # Format results for multi-report
                        findings = []
                        for report in report_result.get("reports", []):
                            findings.append({
                                "type": "synthesis",
                                "report_name": report["name"],
                                "report_path": str(report["report"]),
                                "appendix_path": str(report.get("appendix", "")),
                                "discoveries_count": report.get("discoveries_count", 0),
                                "total_findings": report.get("total_findings", 0),
                            })

                        result = TaskResult(
                            success=True,
                            task_id=task.task_id,
                            task_type=task.task_type.value,
                            findings=findings,
                            cost=report_result.get("total_cost", 0.0),
                            metadata={
                                "output_dir": str(report_result["output_dir"]),
                                "reports_generated": len(findings),
                            },
                        )
                    else:
                        # Generate single report
                        output_path = Path(output_dir) / f"{report_name}.md"
                        report_result = generator.generate_report(
                            output_path=output_path,
                            generate_narratives=generate_narratives,
                            include_appendix=True,
                        )

                        # Return results
                        result = TaskResult(
                            success=True,
                            task_id=task.task_id,
                            task_type=task.task_type.value,
                            findings=[{
                                "type": "synthesis",
                                "report_path": str(report_result["report"]),
                                "appendix_path": str(report_result.get("appendix")),
                                "discoveries_count": report_result.get("discoveries_count", 0),
                            }],
                            cost=report_result.get("cost", 0.0),
                            metadata={"report_path": str(report_result["report"])},
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
                print(f"  [TASK] Completed {task.task_type.value}: ${result.cost:.4f}")
            else:
                task.status = TaskStatus.FAILED
                task.error = result.error
                task.result = result.to_dict()
                print(f"  [TASK] Failed {task.task_type.value}: {result.error}")

            # Update budget
            cycle.budget_used += result.cost
            self.total_budget_used += result.cost

            # Auto-schedule hypothesis tests if we just generated hypotheses
            if task.task_type == TaskType.GENERATE_HYPOTHESIS and result.success:
                await self._schedule_hypothesis_tests(task, result, cycle)

        except Exception as e:
            # Handle unexpected errors
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.result = {
                "success": False,
                "error": str(e),
            }
            print(f"  [TASK] Error in {task.task_type.value}: {str(e)}")

        finally:
            # Mark completion time and remove from active tasks
            task.completed_at = datetime.utcnow()
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]

        return task.result or {}

    async def _schedule_hypothesis_tests(
        self,
        hypothesis_task: Task,
        result,
        cycle: Cycle,
    ) -> None:
        """
        Automatically schedule hypothesis tests for newly generated hypotheses.

        Args:
            hypothesis_task: The completed hypothesis generation task
            result: TaskResult from hypothesis generation
            cycle: The current cycle
        """
        from src.orchestrator.agent_coordinator import TaskResult

        # Extract hypothesis IDs from result metadata
        hypothesis_ids = result.metadata.get("hypothesis_ids", [])

        if not hypothesis_ids:
            print("No new hypotheses generated, skipping test scheduling")
            return

        print(f"Auto-scheduling tests for {len(hypothesis_ids)} new hypotheses")

        # Get dataset path from original task context if available
        dataset_path = hypothesis_task.context.get("dataset_path")

        # Schedule a test task for each new hypothesis
        for hypothesis_id in hypothesis_ids:
            # Check if we've already hit max tasks for this cycle
            if len(cycle.tasks) >= cycle.max_tasks:
                print(f"Cycle max tasks ({cycle.max_tasks}) reached, skipping remaining hypothesis tests")
                break

            # Create test task
            test_task = Task(
                task_id=str(uuid4()),
                task_type=TaskType.TEST_HYPOTHESIS,
                status=TaskStatus.PENDING,
                objective=f"Test hypothesis {hypothesis_id}",
                context={
                    "hypothesis_id": hypothesis_id,
                    "dataset_path": dataset_path,
                    "test_approaches": ["both"],  # Use both data and literature testing
                },
                parent_task_id=hypothesis_task.task_id,
            )

            # Add task to cycle
            cycle.tasks.append(test_task)
            print(f"  → Scheduled TEST_HYPOTHESIS task for {hypothesis_id}")

        # Note: These tasks will be picked up in the next execution wave
        # since we're in the middle of executing the current wave

    def _count_untested_hypotheses(self) -> int:
        """Count hypotheses that haven't been tested yet."""
        count = 0
        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") == "hypothesis":
                metadata = data.get("metadata", {})
                if not metadata.get("tested", False):
                    count += 1
        return count

    def _check_recent_progress(self, num_cycles: int = 3) -> Dict[str, Any]:
        """
        Check if meaningful progress has been made in recent cycles.

        This prevents the system from spawning new cycles when it's stuck
        in an unproductive loop.

        Args:
            num_cycles: Number of recent cycles to check

        Returns:
            Dictionary with progress metrics and has_progress flag
        """
        # Get the most recent completed cycles
        completed_cycles = [
            c for c in self.cycles.values()
            if c.status in [TaskStatus.COMPLETED, TaskStatus.FAILED] and c.started_at
        ]
        sorted_cycles = sorted(
            completed_cycles,
            key=lambda c: c.started_at,
            reverse=True
        )[:num_cycles]

        if len(sorted_cycles) < num_cycles:
            # Not enough cycles to judge - allow spawning
            return {
                "has_progress": True,
                "cycles_checked": len(sorted_cycles),
                "total_findings": 0,
                "hypotheses_tested": 0,
                "novel_findings": 0,
                "reason": "Not enough cycles to evaluate"
            }

        # Count progress metrics across recent cycles
        total_findings = 0
        hypotheses_tested = 0
        novel_findings = 0
        novelty_threshold = 0.5  # Lower threshold for progress check

        # Get the start time of the oldest cycle we're checking
        oldest_start = sorted_cycles[-1].started_at

        for node_id, data in self.world_model.graph.nodes(data=True):
            node_created = data.get("created_at")
            if not node_created:
                continue

            # Parse datetime if string
            if isinstance(node_created, str):
                try:
                    node_created = datetime.fromisoformat(node_created)
                except (ValueError, AttributeError):
                    continue

            # Only count nodes created during the cycles we're checking
            if node_created < oldest_start:
                continue

            node_type = data.get("node_type")

            if node_type == "finding":
                total_findings += 1
                novelty = data.get("metadata", {}).get("novelty", 0.0)
                if novelty > novelty_threshold:
                    novel_findings += 1

            elif node_type == "hypothesis":
                metadata = data.get("metadata", {})
                if metadata.get("tested", False):
                    hypotheses_tested += 1

        # Determine if there's meaningful progress
        # Progress requires at least ONE of:
        # - 2+ findings created
        # - 1+ hypothesis tested
        # - 1+ novel finding
        has_progress = (
            total_findings >= 2 or
            hypotheses_tested >= 1 or
            novel_findings >= 1
        )

        return {
            "has_progress": has_progress,
            "cycles_checked": len(sorted_cycles),
            "total_findings": total_findings,
            "hypotheses_tested": hypotheses_tested,
            "novel_findings": novel_findings,
            "reason": "Sufficient progress" if has_progress else "Insufficient progress in recent cycles"
        }

    def _should_trigger_synthesis_cycle(self) -> Dict[str, Any]:
        """
        Check if a synthesis cycle should be triggered.

        Synthesis cycles are triggered:
        - Every N exploration cycles (default: 5)
        - When progress has stalled but we haven't hit budget limits
        - When there are contradictory findings that need reconciliation
        - When coverage of original objective is imbalanced
        - When Claude assesses that synthesis would be valuable

        Returns:
            Dictionary with should_trigger flag and reason
        """
        SYNTHESIS_INTERVAL = 5  # Trigger synthesis every 5 exploration cycles

        # Count completed exploration cycles since last synthesis
        completed_cycles = [
            c for c in self.cycles.values()
            if c.status == TaskStatus.COMPLETED
        ]
        sorted_cycles = sorted(
            completed_cycles,
            key=lambda c: c.created_at,
            reverse=True
        )

        # Find last synthesis cycle
        last_synthesis_idx = -1
        for i, cycle in enumerate(sorted_cycles):
            if cycle.cycle_type == CycleType.SYNTHESIS:
                last_synthesis_idx = i
                break

        # Count exploration cycles since last synthesis
        if last_synthesis_idx == -1:
            cycles_since_synthesis = len([c for c in sorted_cycles if c.cycle_type == CycleType.EXPLORATION])
        else:
            cycles_since_synthesis = len([
                c for c in sorted_cycles[:last_synthesis_idx]
                if c.cycle_type == CycleType.EXPLORATION
            ])

        # Basic interval check
        interval_trigger = cycles_since_synthesis >= SYNTHESIS_INTERVAL

        # Check for contradictions using embedding service
        contradictions = self._detect_contradictory_findings()
        has_contradictions = len(contradictions) > 0

        # Check coverage imbalance
        coverage_analysis = self._analyze_objective_coverage()
        has_coverage_gap = coverage_analysis.get("coverage_gap", 0) > 0.4

        # Use Claude to assess synthesis value (if enough cycles have passed)
        claude_recommends = False
        claude_reasoning = ""
        if cycles_since_synthesis >= 3 and not interval_trigger:
            claude_assessment = self._claude_assess_synthesis_value()
            claude_recommends = claude_assessment.get("should_synthesize", False)
            claude_reasoning = claude_assessment.get("reasoning", "")

        # Determine if we should trigger synthesis
        should_trigger = (
            interval_trigger or
            has_contradictions or
            has_coverage_gap or
            claude_recommends
        )

        # Build reason string
        reasons = []
        if interval_trigger:
            reasons.append(f"interval reached ({cycles_since_synthesis} cycles)")
        if has_contradictions:
            reasons.append(f"found {len(contradictions)} contradictory findings")
        if has_coverage_gap:
            reasons.append(f"coverage gap detected ({coverage_analysis.get('coverage_gap', 0):.1%})")
        if claude_recommends:
            reasons.append(f"Claude assessment: {claude_reasoning[:50]}")

        return {
            "should_trigger": should_trigger,
            "cycles_since_synthesis": cycles_since_synthesis,
            "synthesis_interval": SYNTHESIS_INTERVAL,
            "contradictions_found": len(contradictions),
            "coverage_gap": coverage_analysis.get("coverage_gap", 0),
            "claude_recommends": claude_recommends,
            "reason": " | ".join(reasons) if reasons else "Not due for synthesis"
        }

    def _detect_contradictory_findings(self) -> List[Dict[str, Any]]:
        """
        Detect potentially contradictory findings using embedding similarity.

        Returns:
            List of contradiction pairs with their details
        """
        try:
            from src.utils.embedding_service import get_embedding_service

            # Collect all findings
            findings = []
            for node_data in self.world_model.query_nodes(NodeType.FINDING):
                findings.append({
                    "id": node_data.get("node_id"),
                    "text": node_data.get("text", ""),
                    "confidence": node_data.get("confidence", 0.5),
                })

            if len(findings) < 2:
                return []

            embedding_service = get_embedding_service()
            contradictions_raw = embedding_service.find_contradictions(
                findings,
                similarity_threshold=0.7,
                min_confidence_diff=0.3
            )

            # Format for return
            contradictions = []
            for finding1, finding2, similarity in contradictions_raw:
                contradictions.append({
                    "finding1_id": finding1.get("id"),
                    "finding1_text": finding1.get("text", "")[:100],
                    "finding2_id": finding2.get("id"),
                    "finding2_text": finding2.get("text", "")[:100],
                    "similarity": round(similarity, 3),
                })

            return contradictions
        except Exception as e:
            print(f"Warning: Could not detect contradictions: {e}")
            return []

    def _analyze_objective_coverage(self) -> Dict[str, Any]:
        """
        Analyze how well current findings cover the original research objective.

        Returns:
            Dictionary with coverage metrics
        """
        try:
            from src.utils.embedding_service import get_embedding_service

            # Get original objective
            original_objective = self._get_original_objective()
            if not original_objective:
                return {"coverage_gap": 0.0, "overall_coverage": 1.0}

            # Collect all findings
            findings = []
            for node_data in self.world_model.query_nodes(NodeType.FINDING):
                text = node_data.get("text", "")
                if text:
                    findings.append(text)

            if not findings:
                return {"coverage_gap": 1.0, "overall_coverage": 0.0}

            embedding_service = get_embedding_service()
            coverage = embedding_service.compute_topic_coverage(
                original_objective,
                findings,
                num_aspects=5
            )

            return coverage
        except Exception as e:
            print(f"Warning: Could not analyze coverage: {e}")
            return {"coverage_gap": 0.0, "overall_coverage": 0.5}

    def _get_original_objective(self) -> Optional[str]:
        """Get the original research objective from the first cycle."""
        if not self.cycles:
            return None

        # Find the root cycle (no parent)
        for cycle in self.cycles.values():
            if cycle.parent_cycle_id is None:
                return cycle.objective

        # Fallback to first cycle
        first_cycle = list(self.cycles.values())[0]
        return first_cycle.objective

    def _claude_assess_synthesis_value(self) -> Dict[str, Any]:
        """
        Use Claude to assess whether a synthesis cycle would be valuable now.

        Returns:
            Dictionary with should_synthesize and reasoning
        """
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                return {"should_synthesize": False, "reasoning": "No API key"}

            # Get context
            summary = self._get_world_model_summary()
            recent_progress = self._check_recent_progress(num_cycles=3)

            prompt = f"""Assess whether a synthesis cycle would be valuable for this research discovery process.

Current State:
- Total findings: {summary['finding_count']}
- Total hypotheses: {summary['hypothesis_count']}
- Total papers reviewed: {summary['paper_count']}

Recent Progress (last 3 cycles):
- New findings: {recent_progress.get('total_findings', 0)}
- Hypotheses tested: {recent_progress.get('hypotheses_tested', 0)}
- Novel findings: {recent_progress.get('novel_findings', 0)}
- Has meaningful progress: {recent_progress.get('has_progress', False)}

Recent Findings:
{self._format_recent_items(summary['recent_findings'])}

Recent Hypotheses:
{self._format_recent_items(summary['recent_hypotheses'])}

A synthesis cycle consolidates findings, identifies gaps and contradictions, and suggests new directions.

Should we trigger a synthesis cycle now? Consider:
1. Are there enough findings to synthesize meaningfully?
2. Has progress slowed, suggesting we need to step back and reassess?
3. Are there apparent gaps or contradictions in the findings?
4. Would synthesis help identify new productive directions?

Respond with JSON:
{{"should_synthesize": true/false, "reasoning": "brief explanation (max 50 words)"}}"""

            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model=os.getenv("CLAUDE_MODEL"),
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text

            # Parse JSON
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            result = json.loads(response_text)
            return {
                "should_synthesize": result.get("should_synthesize", False),
                "reasoning": result.get("reasoning", "")
            }

        except Exception as e:
            print(f"Warning: Claude synthesis assessment failed: {e}")
            return {"should_synthesize": False, "reasoning": f"Error: {str(e)[:30]}"}

    def _create_synthesis_objective(self) -> str:
        """
        Create an intelligent objective for a synthesis cycle using Claude.

        Analyzes current state, identifies under-explored aspects, and
        generates a focused synthesis objective.

        Returns:
            Synthesis cycle objective string
        """
        # Get current state summary
        summary = self._get_world_model_summary()
        original_objective = self._get_original_objective()
        contradictions = self._detect_contradictory_findings()
        coverage = self._analyze_objective_coverage()

        # Try Claude-based objective generation
        try:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if api_key:
                objective = self._claude_generate_synthesis_objective(
                    summary, original_objective, contradictions, coverage
                )
                if objective:
                    return objective
        except Exception as e:
            print(f"Warning: Claude synthesis objective generation failed: {e}")

        # Fallback to rule-based objective generation
        return self._fallback_synthesis_objective(summary, contradictions, coverage)

    def _claude_generate_synthesis_objective(
        self,
        summary: Dict[str, Any],
        original_objective: Optional[str],
        contradictions: List[Dict[str, Any]],
        coverage: Dict[str, Any],
    ) -> Optional[str]:
        """
        Use Claude to generate an intelligent synthesis objective.

        Args:
            summary: World model summary
            original_objective: The original research objective
            contradictions: List of detected contradictions
            coverage: Coverage analysis results

        Returns:
            Synthesis objective string, or None if generation fails
        """
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            return None

        # Format contradictions for prompt
        contradictions_text = "None detected"
        if contradictions:
            contradictions_text = "\n".join([
                f"- \"{c['finding1_text'][:60]}...\" vs \"{c['finding2_text'][:60]}...\""
                for c in contradictions[:3]
            ])

        prompt = f"""Generate a focused synthesis objective for this research discovery process.

Original Research Objective:
{original_objective or 'Not specified'}

Current Knowledge State:
- Findings: {summary['finding_count']}
- Hypotheses: {summary['hypothesis_count']}
- Papers reviewed: {summary['paper_count']}

Recent Findings:
{self._format_recent_items(summary['recent_findings'])}

Recent Hypotheses:
{self._format_recent_items(summary['recent_hypotheses'])}

Coverage Analysis:
- Overall coverage: {coverage.get('overall_coverage', 0):.1%}
- Coverage gap: {coverage.get('coverage_gap', 0):.1%}
- Aspects covered: {coverage.get('aspects_covered', 0)}

Potentially Contradictory Findings:
{contradictions_text}

Generate a synthesis objective that:
1. If contradictions exist: Focus on reconciling them
2. If coverage gaps exist: Identify under-explored aspects to investigate
3. Otherwise: Consolidate findings into coherent insights and identify new directions

The objective should be specific, actionable, and directly address the most pressing need.
Keep it to 1-2 sentences, max 100 words.

Respond with just the objective text, no quotes or formatting."""

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=os.getenv("CLAUDE_MODEL"),
            max_tokens=200,
            temperature=0.5,
            messages=[{"role": "user", "content": prompt}],
        )

        objective = response.content[0].text.strip()

        # Clean up any quotes or formatting
        if objective.startswith('"') and objective.endswith('"'):
            objective = objective[1:-1]

        return objective if objective else None

    def _fallback_synthesis_objective(
        self,
        summary: Dict[str, Any],
        contradictions: List[Dict[str, Any]],
        coverage: Dict[str, Any],
    ) -> str:
        """
        Generate synthesis objective using rule-based approach.

        Args:
            summary: World model summary
            contradictions: List of detected contradictions
            coverage: Coverage analysis results

        Returns:
            Synthesis objective string
        """
        # Prioritize based on what needs attention
        if contradictions:
            return (
                f"Reconcile {len(contradictions)} contradictory findings and establish "
                f"which conclusions are best supported by evidence"
            )

        if coverage.get("coverage_gap", 0) > 0.4:
            return (
                f"Analyze coverage gaps in current research and identify under-explored "
                f"aspects of the original objective (current coverage: {coverage.get('overall_coverage', 0):.0%})"
            )

        # Default synthesis
        base_objective = "Synthesize and evaluate current knowledge"
        context_parts = []
        if summary["finding_count"] > 0:
            context_parts.append(f"{summary['finding_count']} findings")
        if summary["hypothesis_count"] > 0:
            context_parts.append(f"{summary['hypothesis_count']} hypotheses")

        if context_parts:
            base_objective = f"{base_objective}: consolidate {', '.join(context_parts)} into coherent insights"

        return base_objective

    def _should_spawn_new_cycle(self, current_cycle: Cycle) -> bool:
        """
        Determine if a new discovery cycle should be spawned.

        Checks:
        - New hypotheses added since cycle start
        - Novel findings exceed threshold
        - Budget remaining > minimum cycle budget
        - Progress gating: at least some progress in recent cycles

        Args:
            current_cycle: The cycle that just completed

        Returns:
            True if a new cycle should be spawned
        """
        # Check if we have budget remaining - reserve budget for final report
        budget_remaining = self.max_total_budget - self.total_budget_used - FINAL_REPORT_BUDGET_RESERVE
        MIN_VIABLE_BUDGET = 0.10  # $0.10 minimum to attempt another cycle
        if budget_remaining < MIN_VIABLE_BUDGET:
            print(f"Insufficient budget for new cycle: ${budget_remaining:.2f} remaining "
                  f"(reserving ${FINAL_REPORT_BUDGET_RESERVE:.2f} for final report)")
            return False

        # Progress gating: check if we've made progress in recent cycles
        # This prevents infinite loops of unproductive cycles
        recent_progress = self._check_recent_progress(num_cycles=3)
        if not recent_progress["has_progress"]:
            print(f"⚠️  Progress gating triggered: No meaningful progress in last {recent_progress['cycles_checked']} cycles")
            print(f"   - Findings in recent cycles: {recent_progress['total_findings']}")
            print(f"   - Hypotheses tested: {recent_progress['hypotheses_tested']}")
            print(f"   - Novel findings: {recent_progress['novel_findings']}")
            # TODO: Future enhancement - instead of stopping, trigger a synthesis/pivot cycle (Fix 10)
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

        # Check for untested hypotheses
        untested_hypotheses = self._count_untested_hypotheses()

        # Check completion score - continue if objective is incomplete
        completion_incomplete = (
            self.last_completion_score is not None and
            self.last_completion_score < 0.7
        )

        # Spawn new cycle if:
        # - We have new hypotheses to explore, OR
        # - We have novel findings that warrant further investigation, OR
        # - We have untested hypotheses, OR
        # - The objective completion is below threshold
        should_spawn = (
            new_hypotheses > 0 or
            novel_findings_count > 0 or
            untested_hypotheses > 0 or
            completion_incomplete
        )

        if should_spawn:
            reasons = []
            if new_hypotheses > 0:
                reasons.append(f"{new_hypotheses} new hypotheses")
            if novel_findings_count > 0:
                reasons.append(f"{novel_findings_count} novel findings")
            if untested_hypotheses > 0:
                reasons.append(f"{untested_hypotheses} untested hypotheses")
            if completion_incomplete:
                reasons.append(f"completion score {self.last_completion_score:.2f} < 0.7")
            print(f"Spawn conditions met: {', '.join(reasons)}")

        return should_spawn

    def _spawn_follow_up_cycle(self, parent_cycle: Cycle) -> Cycle:
        """
        Spawn a follow-up discovery cycle based on parent cycle results.

        Args:
            parent_cycle: The parent cycle that triggered this spawn

        Returns:
            New Cycle object linked to parent
        """
        # Check if we should trigger a synthesis cycle instead
        synthesis_check = self._should_trigger_synthesis_cycle()
        if synthesis_check["should_trigger"]:
            print(f"🔄 Triggering synthesis cycle: {synthesis_check['reason']}")
            return self._create_synthesis_cycle(parent_cycle)

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

        # Determine cycle type based on situation
        cycle_type = CycleType.EXPLORATION

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
            # Broaden the search - but prevent recursive nesting of "Continue investigation from:"
            # Strip any existing "Continue investigation from:" prefixes to get the base objective
            base_objective = parent_cycle.objective
            while base_objective.startswith("Continue investigation from: "):
                base_objective = base_objective[len("Continue investigation from: "):]
            # Also strip "Synthesize findings and explore new aspects of:" prefix
            while base_objective.startswith("Synthesize findings and explore new aspects of: "):
                base_objective = base_objective[len("Synthesize findings and explore new aspects of: "):]

            # Count how many times we've continued without progress
            nesting_count = parent_cycle.objective.count("Continue investigation from:")

            if nesting_count >= 2:
                # After 2 continuations without findings/hypotheses, force a pivot cycle
                new_objective = f"Synthesize findings and explore new aspects of: {base_objective}"
                cycle_type = CycleType.PIVOT
            else:
                new_objective = f"Continue investigation from: {base_objective}"

        # Create new cycle with type and parent linkage
        new_cycle = self.create_cycle(
            objective=new_objective,
            max_tasks=10,
        )

        # Set cycle type and parent linkage
        new_cycle.cycle_type = cycle_type
        new_cycle.parent_cycle_id = parent_cycle.cycle_id
        new_cycle.status = TaskStatus.PENDING

        return new_cycle

    def _create_synthesis_cycle(self, parent_cycle: Cycle) -> Cycle:
        """
        Create a synthesis cycle for meta-analysis and knowledge consolidation.

        Analyzes the current state to identify:
        - Gaps in coverage of the original objective
        - Contradictory findings that need reconciliation
        - Under-explored areas that might benefit from pivots

        Args:
            parent_cycle: The parent cycle that triggered this synthesis

        Returns:
            New Cycle object configured for synthesis
        """
        # Analyze current state
        contradictions = self._detect_contradictory_findings()
        coverage = self._analyze_objective_coverage()
        original_objective = self._get_original_objective()

        # Log synthesis analysis
        print(f"   📊 Synthesis Analysis:")
        if contradictions:
            print(f"      ⚠️  Found {len(contradictions)} potentially contradictory findings")
        if coverage.get("coverage_gap", 0) > 0.3:
            print(f"      📉 Coverage gap: {coverage.get('coverage_gap', 0):.0%} of objective under-explored")
        print(f"      📈 Current coverage: {coverage.get('overall_coverage', 0):.0%}")

        # Generate synthesis objective (uses Claude if available)
        synthesis_objective = self._create_synthesis_objective()

        # Determine appropriate number of tasks based on what needs to be done
        max_tasks = 5
        if contradictions:
            max_tasks += min(len(contradictions), 3)  # Add tasks for reconciliation
        if coverage.get("coverage_gap", 0) > 0.4:
            max_tasks += 2  # Add tasks for gap exploration

        # Create the synthesis cycle
        new_cycle = self.create_cycle(
            objective=synthesis_objective,
            max_tasks=min(max_tasks, 10),  # Cap at 10 tasks
        )

        # Set cycle type and parent linkage
        new_cycle.cycle_type = CycleType.SYNTHESIS
        new_cycle.parent_cycle_id = parent_cycle.cycle_id
        new_cycle.status = TaskStatus.PENDING

        # Store synthesis context in cycle metadata for task planning
        new_cycle.metadata = {
            "contradictions": contradictions,
            "coverage_analysis": coverage,
            "original_objective": original_objective,
        }

        print(f"   📊 Created synthesis cycle: {new_cycle.cycle_id[:8]}...")
        print(f"   📋 Objective: {synthesis_objective[:100]}...")

        return new_cycle

    def _get_recent_findings(self, num_cycles: int = 3) -> List[Dict[str, Any]]:
        """
        Get findings from the most recent N cycles.

        Args:
            num_cycles: Number of recent cycles to retrieve findings from

        Returns:
            List of findings from recent cycles
        """
        recent_findings = []

        # Get the most recent cycles
        sorted_cycles = sorted(
            self.cycles.values(),
            key=lambda c: c.started_at if c.started_at else datetime.min,
            reverse=True
        )

        recent_cycle_ids = [c.cycle_id for c in sorted_cycles[:num_cycles]]

        # Collect findings from these cycles
        for cycle in sorted_cycles[:num_cycles]:
            if not cycle.started_at:
                continue

            # Get findings created during this cycle
            for node_id, data in self.world_model.graph.nodes(data=True):
                if data.get("node_type") != NodeType.FINDING.value:
                    continue

                node_created = data.get("created_at")
                if node_created and cycle.started_at:
                    # Parse datetime if it's a string
                    if isinstance(node_created, str):
                        try:
                            node_created = datetime.fromisoformat(node_created)
                        except (ValueError, AttributeError):
                            continue

                    if node_created >= cycle.started_at:
                        recent_findings.append({
                            "node_id": node_id,
                            "text": data.get("text"),
                            "confidence": data.get("confidence", 0.0),
                            "cycle_id": cycle.cycle_id,
                        })

        return recent_findings

    def _compute_findings_rate(self) -> List[float]:
        """
        Compute the rate of new findings per cycle.

        Returns:
            List of findings counts per cycle (in chronological order)
        """
        findings_per_cycle = []

        # Sort cycles chronologically
        sorted_cycles = sorted(
            self.cycles.values(),
            key=lambda c: c.started_at if c.started_at else datetime.min
        )

        for cycle in sorted_cycles:
            if not cycle.started_at:
                continue

            # Count findings created during this cycle
            count = 0
            for node_id, data in self.world_model.graph.nodes(data=True):
                if data.get("node_type") != NodeType.FINDING.value:
                    continue

                node_created = data.get("created_at")
                if node_created and cycle.started_at:
                    # Parse datetime if it's a string
                    if isinstance(node_created, str):
                        try:
                            node_created = datetime.fromisoformat(node_created)
                        except (ValueError, AttributeError):
                            continue

                    # Check if created during this cycle
                    cycle_end = cycle.completed_at if cycle.completed_at else datetime.utcnow()
                    if cycle.started_at <= node_created <= cycle_end:
                        count += 1

            findings_per_cycle.append(float(count))

        return findings_per_cycle

    def _summarize_findings(self, max_findings: int = 20) -> str:
        """
        Summarize recent high-confidence findings for LLM assessment.

        Args:
            max_findings: Maximum number of findings to include

        Returns:
            Formatted string of findings
        """
        # Get all findings from world model
        all_findings = []
        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") == NodeType.FINDING.value:
                all_findings.append({
                    "text": data.get("text", ""),
                    "confidence": data.get("confidence", 0.0),
                    "created_at": data.get("created_at", ""),
                })

        # Sort by confidence and recency
        all_findings.sort(
            key=lambda f: (f["confidence"], f["created_at"]),
            reverse=True
        )

        # Take top findings
        top_findings = all_findings[:max_findings]

        # Format for LLM
        if not top_findings:
            return "No findings yet."

        formatted = []
        for i, finding in enumerate(top_findings, 1):
            formatted.append(
                f"{i}. {finding['text']} (confidence: {finding['confidence']:.2f})"
            )

        return "\n".join(formatted)

    def _assess_objective_with_llm(self) -> Dict[str, Any]:
        """
        Use Claude to assess objective completion.

        Returns:
            Dictionary with completion_score (0.0-1.0) and reasoning
        """
        # Get the objective from the first cycle (they're all related)
        objective = "Research objective not specified"
        if self.cycles:
            first_cycle = list(self.cycles.values())[0]
            objective = first_cycle.objective

        # Get findings summary
        findings_summary = self._summarize_findings()

        # Get world model stats
        stats = self.world_model.get_stats()

        # Construct prompt for Claude
        prompt = f"""Research Objective: {objective}

Current Findings:
{findings_summary}

World Model Statistics:
- Total nodes: {stats['total_nodes']}
- Total findings: {stats['node_types'].get('finding', 0)}
- Total hypotheses: {stats['node_types'].get('hypothesis', 0)}
- Total papers: {stats['node_types'].get('paper', 0)}

On a scale of 0.0 to 1.0, how well have we addressed this research objective?

Consider:
- Coverage of key aspects of the objective
- Quality and confidence of evidence
- Novel insights discovered
- Remaining questions or gaps

Respond with JSON in this exact format:
{{"completion_score": 0.0, "reasoning": "explanation here"}}"""

        try:
            # Call Claude API
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print("⚠️  No ANTHROPIC_API_KEY found, skipping LLM assessment")
                return {
                    "completion_score": 0.0,
                    "reasoning": "API key not available",
                    "error": "No API key"
                }

            client = anthropic.Anthropic(api_key=api_key)

            response = client.messages.create(
                model=os.getenv("CLAUDE_MODEL"),
                max_tokens=1000,
                temperature=0.3,  # Lower temperature for more consistent scoring
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

            result = json.loads(response_text)

            # Validate result
            if "completion_score" not in result:
                return {
                    "completion_score": 0.0,
                    "reasoning": "Invalid response format",
                    "error": "Missing completion_score"
                }

            # Ensure score is in valid range
            score = float(result["completion_score"])
            score = max(0.0, min(1.0, score))

            return {
                "completion_score": score,
                "reasoning": result.get("reasoning", "No reasoning provided")
            }

        except Exception as e:
            print(f"⚠️  Error in LLM assessment: {e}")
            return {
                "completion_score": 0.0,
                "reasoning": f"Error during assessment: {str(e)}",
                "error": str(e)
            }

    def _compute_completion_heuristics(self) -> Dict[str, Any]:
        """
        Calculate objective completion metrics using heuristics.

        Returns:
            Dictionary with various completion metrics
        """
        # Get all findings and hypotheses
        all_findings = []
        all_hypotheses = []
        high_confidence_findings = []

        for node_id, data in self.world_model.graph.nodes(data=True):
            node_type = data.get("node_type")
            confidence = data.get("confidence", 0.0)

            if node_type == NodeType.FINDING.value:
                all_findings.append({
                    "id": node_id,
                    "confidence": confidence,
                    "novelty": data.get("metadata", {}).get("novelty", 0.5),
                })
                if confidence > 0.7:
                    high_confidence_findings.append(node_id)

            elif node_type == NodeType.HYPOTHESIS.value:
                # Check if hypothesis is tested (has incoming support/refute edges)
                is_tested = False
                for _, _, edge_data in self.world_model.graph.in_edges(node_id, data=True):
                    edge_type = edge_data.get("edge_type")
                    if edge_type in ["supports", "refutes"]:
                        is_tested = True
                        break

                all_hypotheses.append({
                    "id": node_id,
                    "is_tested": is_tested,
                })

        # Calculate metrics
        total_hypotheses = len(all_hypotheses)
        tested_hypotheses = sum(1 for h in all_hypotheses if h["is_tested"])

        hypothesis_validation_rate = (
            tested_hypotheses / total_hypotheses if total_hypotheses > 0 else 0.0
        )

        # Calculate novelty score (average across all findings)
        novelty_scores = [f["novelty"] for f in all_findings]
        novelty_score = sum(novelty_scores) / len(novelty_scores) if novelty_scores else 0.0

        # Calculate diminishing returns
        findings_rate = self._compute_findings_rate()
        if len(findings_rate) >= 2:
            recent_rate = findings_rate[-1]
            historical_avg = sum(findings_rate[:-1]) / len(findings_rate[:-1])
            diminishing_returns = recent_rate < 0.5 * historical_avg if historical_avg > 0 else False
        else:
            diminishing_returns = False

        # Estimate world model coverage
        # This is a rough estimate based on number of nodes and relationships
        total_nodes = self.world_model.graph.number_of_nodes()
        total_edges = self.world_model.graph.number_of_edges()

        # Heuristic: well-connected graph suggests good coverage
        # Average degree (connections per node)
        avg_degree = (2 * total_edges / total_nodes) if total_nodes > 0 else 0
        # Normalize to 0-1 scale (assume 5+ connections per node is excellent coverage)
        world_model_coverage = min(1.0, avg_degree / 5.0)

        return {
            "findings_count": len(high_confidence_findings),
            "total_findings": len(all_findings),
            "hypothesis_validation_rate": hypothesis_validation_rate,
            "total_hypotheses": total_hypotheses,
            "tested_hypotheses": tested_hypotheses,
            "novelty_score": novelty_score,
            "diminishing_returns": diminishing_returns,
            "world_model_coverage": world_model_coverage,
            "total_nodes": total_nodes,
            "findings_rate_history": findings_rate,
        }

    def _check_objective_completion(self) -> bool:
        """
        Check if research objective is complete using combined LLM and heuristic approach.

        Uses three methods:
        1. LLM-based assessment for semantic understanding
        2. Heuristic-based metrics for objective measures
        3. Combined decision logic

        Returns:
            True if objective appears to be complete
        """
        print("\n" + "="*60)
        print("OBJECTIVE COMPLETION CHECK")
        print("="*60)

        # Method 1: Compute heuristics
        print("\n📊 Computing heuristics...")
        heuristics = self._compute_completion_heuristics()

        print(f"  • High-confidence findings: {heuristics['findings_count']}")
        print(f"  • Total findings: {heuristics['total_findings']}")
        print(f"  • Hypothesis validation rate: {heuristics['hypothesis_validation_rate']:.1%}")
        print(f"  • Tested hypotheses: {heuristics['tested_hypotheses']}/{heuristics['total_hypotheses']}")
        print(f"  • Novelty score: {heuristics['novelty_score']:.2f}")
        print(f"  • World model coverage: {heuristics['world_model_coverage']:.2f}")
        print(f"  • Diminishing returns: {'Yes' if heuristics['diminishing_returns'] else 'No'}")

        # Method 2: LLM assessment
        print("\n🤖 Assessing with LLM...")
        llm_assessment = self._assess_objective_with_llm()

        if "error" not in llm_assessment:
            print(f"  • Completion score: {llm_assessment['completion_score']:.2f}")
            print(f"  • Reasoning: {llm_assessment['reasoning'][:100]}...")
        else:
            print(f"  ⚠️  LLM assessment failed: {llm_assessment.get('error')}")

        # Method 3: Combined decision logic
        print("\n⚖️  Combined assessment...")

        llm_score = llm_assessment.get("completion_score", 0.0)
        has_llm = "error" not in llm_assessment

        # Store for use in spawn decisions
        self.last_completion_score = llm_score if has_llm else None

        # Decision criteria
        completion_criteria = []

        # Criterion 1: LLM confidence + heuristics support
        if has_llm and llm_score > 0.8 and heuristics["findings_count"] >= 5:
            completion_criteria.append("LLM high confidence + sufficient findings")

        # Criterion 2: Strong heuristics indicate completion
        if (heuristics["hypothesis_validation_rate"] > 0.9 and
            heuristics["diminishing_returns"] and
            heuristics["findings_count"] >= 5):
            completion_criteria.append("High validation rate + diminishing returns")

        # Criterion 3: Comprehensive coverage with quality findings
        if (heuristics["world_model_coverage"] > 0.7 and
            heuristics["findings_count"] >= 8 and
            heuristics["novelty_score"] > 0.6):
            completion_criteria.append("Comprehensive coverage + quality findings")

        # Criterion 4: LLM moderate confidence + very strong heuristics
        if (has_llm and llm_score > 0.7 and
            heuristics["hypothesis_validation_rate"] > 0.85 and
            heuristics["findings_count"] >= 7):
            completion_criteria.append("LLM moderate confidence + strong heuristics")

        # Make decision
        is_complete = len(completion_criteria) > 0

        print(f"\n{'✅' if is_complete else '❌'} Decision: {'OBJECTIVE COMPLETE' if is_complete else 'Continue research'}")
        if completion_criteria:
            print("  Criteria met:")
            for criterion in completion_criteria:
                print(f"    • {criterion}")
        else:
            print("  No completion criteria met yet")

        print("="*60 + "\n")

        return is_complete

    def _should_trigger_synthesis(self, current_cycle_num: int) -> bool:
        """
        Determine if synthesis should run.

        Args:
            current_cycle_num: Current cycle number (0-indexed)

        Returns:
            True if synthesis should be triggered
        """
        # Interval-based: Every N cycles
        if self.synthesis_interval and (current_cycle_num + 1) % self.synthesis_interval == 0:
            return True

        # Completion-based: Check if objective met
        if self._check_objective_completion():
            return True

        # Budget-based: 90% of budget used
        if self.total_budget_used >= 0.9 * self.max_total_budget:
            return True

        return False

    async def _execute_synthesis_task(self, current_cycle_num: int) -> Optional[Dict[str, Any]]:
        """
        Execute a synthesis task to generate a report.

        Args:
            current_cycle_num: Current cycle number

        Returns:
            Synthesis result dictionary or None if failed
        """
        from src.orchestrator.agent_coordinator import TaskResult

        # Create synthesis task
        synthesis_task = Task(
            task_id=str(uuid4()),
            task_type=TaskType.SYNTHESIZE_FINDINGS,
            status=TaskStatus.PENDING,
            objective=f"Synthesize discoveries from cycle {current_cycle_num + 1}",
            context={
                "output_dir": self.output_dir,
                "report_name": f"discovery_report_cycle_{current_cycle_num + 1}",
                "min_confidence": 0.7,
                "generate_narratives": True,
            },
        )

        # Execute synthesis task
        # We need to simulate a cycle context for the task execution
        # Create a temporary cycle to hold this task
        temp_cycle = Cycle(
            cycle_id=f"synthesis_{current_cycle_num}",
            objective="Synthesis",
            status=TaskStatus.RUNNING,
            tasks=[synthesis_task],
        )
        temp_cycle.started_at = datetime.utcnow()

        # Execute the task
        result_dict = await self._execute_task(synthesis_task, temp_cycle)

        # Store result
        if result_dict.get("success"):
            self.synthesis_results.append(result_dict)
            print(f"✓ Synthesis complete: {result_dict.get('metadata', {}).get('report_path')}")
        else:
            print(f"⚠️  Synthesis failed: {result_dict.get('error')}")

        return result_dict

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

    def _identify_knowledge_gaps(self) -> Dict[str, Any]:
        """
        Analyze world model for knowledge gaps.

        Returns:
            Dictionary containing:
                - unexplored_data: List of datasets or areas with minimal analysis
                - weak_evidence: List of claims with low validation strength
                - novel_findings_count: Count of novel findings that need follow-up
        """
        unexplored_data = []
        weak_evidence = []
        novel_findings = []

        # Track which datasets have been analyzed
        analyzed_datasets = set()
        all_datasets = set()

        # Identify hypotheses with weak evidence
        for node_id, data in self.world_model.graph.nodes(data=True):
            node_type = data.get("node_type")

            if node_type == NodeType.HYPOTHESIS.value:
                # Count supporting and refuting evidence
                support_count = 0
                refute_count = 0

                for _, _, edge_data in self.world_model.graph.in_edges(node_id, data=True):
                    edge_type = edge_data.get("edge_type")
                    if edge_type == "supports":
                        support_count += 1
                    elif edge_type == "refutes":
                        refute_count += 1

                total_evidence = support_count + refute_count
                confidence = data.get("confidence", 0.0)

                # Consider it weak evidence if:
                # - Has few supporting pieces of evidence (< 2)
                # - Low confidence (< 0.6)
                # - Not refuted (still active)
                if total_evidence < 2 and confidence < 0.6 and refute_count == 0:
                    validation_strength = total_evidence / 3.0  # Normalize to 0-1
                    weak_evidence.append({
                        "hypothesis_id": node_id,
                        "text": data.get("text", ""),
                        "confidence": confidence,
                        "validation_strength": validation_strength,
                        "evidence_count": total_evidence,
                        "novelty": data.get("metadata", {}).get("novelty", 0.5),
                        "objective_alignment": self._compute_objective_alignment(data.get("text", "")),
                    })

            elif node_type == NodeType.FINDING.value:
                # Track novel findings
                novelty = data.get("metadata", {}).get("novelty", 0.5)
                confidence = data.get("confidence", 0.0)

                if novelty > 0.7 and confidence > 0.6:
                    novel_findings.append({
                        "finding_id": node_id,
                        "text": data.get("text", ""),
                        "novelty": novelty,
                        "confidence": confidence,
                        "objective_alignment": self._compute_objective_alignment(data.get("text", "")),
                    })

            elif node_type == NodeType.DATASET.value:
                # Track all datasets
                all_datasets.add(node_id)

                # Check if dataset has been analyzed (has outgoing edges to findings)
                has_findings = False
                for _, target, edge_data in self.world_model.graph.out_edges(node_id, data=True):
                    target_data = self.world_model.graph.nodes[target]
                    if target_data.get("node_type") == NodeType.FINDING.value:
                        has_findings = True
                        analyzed_datasets.add(node_id)
                        break

                if not has_findings:
                    # Unexplored dataset
                    unexplored_data.append({
                        "dataset_id": node_id,
                        "text": data.get("text", ""),
                        "novelty": 0.8,  # High novelty for unexplored data
                        "validation_strength": 0.0,  # No validation yet
                        "objective_alignment": self._compute_objective_alignment(data.get("text", "")),
                    })

        # Identify areas with no recent exploration (based on papers without follow-up)
        for node_id, data in self.world_model.graph.nodes(data=True):
            if data.get("node_type") == NodeType.PAPER.value:
                # Check if paper has been used to generate hypotheses or findings
                has_follow_up = False
                for _, target, _ in self.world_model.graph.out_edges(node_id, data=True):
                    target_data = self.world_model.graph.nodes[target]
                    target_type = target_data.get("node_type")
                    if target_type in [NodeType.HYPOTHESIS.value, NodeType.FINDING.value]:
                        has_follow_up = True
                        break

                if not has_follow_up:
                    # Paper without follow-up represents unexplored research area
                    unexplored_data.append({
                        "paper_id": node_id,
                        "text": data.get("text", "")[:200],  # Truncate for readability
                        "novelty": 0.7,
                        "validation_strength": 0.0,
                        "objective_alignment": self._compute_objective_alignment(data.get("text", "")),
                    })

        return {
            "unexplored_data": unexplored_data,
            "weak_evidence": weak_evidence,
            "novel_findings_count": len(novel_findings),
            "novel_findings": novel_findings,
        }

    def _compute_objective_alignment(self, text: str) -> float:
        """
        Compute how well a piece of text aligns with the current research objective.

        Uses simple keyword matching for now. Could be enhanced with embeddings.

        Args:
            text: Text to evaluate

        Returns:
            Alignment score (0.0 to 1.0)
        """
        if not self.cycles:
            return 0.5  # Default if no objective

        # Get objective from most recent cycle
        recent_cycles = sorted(
            self.cycles.values(),
            key=lambda c: c.created_at,
            reverse=True
        )
        objective = recent_cycles[0].objective if recent_cycles else ""

        if not objective:
            return 0.5

        # Simple keyword-based alignment
        # Extract important words from objective (> 4 chars, lowercase)
        objective_words = set(
            word.lower()
            for word in objective.split()
            if len(word) > 4 and word.isalnum()
        )

        text_words = set(
            word.lower()
            for word in text.split()
            if len(word) > 4 and word.isalnum()
        )

        if not objective_words:
            return 0.5

        # Calculate Jaccard similarity
        intersection = len(objective_words & text_words)
        union = len(objective_words | text_words)

        if union == 0:
            return 0.0

        alignment = intersection / len(objective_words)  # Normalize by objective words
        return min(1.0, alignment)  # Cap at 1.0

    def _score_task_priority(self, gap: Dict[str, Any], type: str) -> float:
        """
        Score task priority based on world model state.

        Args:
            gap: Gap information dictionary
            type: Task type ("data_analysis", "literature", "hypothesis_gen")

        Returns:
            Priority score (0.0 to 1.0+)
        """
        score = 0.0

        # Novelty seeking: higher score for unexplored areas
        score += gap.get("novelty", 0.5) * 0.4

        # Validation: higher score for unvalidated claims
        score += (1.0 - gap.get("validation_strength", 0.5)) * 0.3

        # Relevance: higher score for objective-aligned tasks
        score += gap.get("objective_alignment", 0.5) * 0.3

        return score

    def _create_analysis_task(self, gap: Dict[str, Any], cycle_id: str) -> Optional[Task]:
        """
        Create a data analysis task for an unexplored gap.

        Args:
            gap: Gap information from _identify_knowledge_gaps
            cycle_id: Cycle ID to attach task to

        Returns:
            New Task object, or None if no dataset is available
        """
        if "dataset_id" in gap and "dataset_path" in gap:
            # Only create ANALYZE_DATA tasks when we have an actual dataset
            objective = f"Analyze unexplored dataset: {gap['text']}"
            context = {
                "dataset_id": gap["dataset_id"],
                "dataset_path": gap["dataset_path"],
                "requires_dataset": True,
            }
        elif "finding_id" in gap:
            # Follow-up on findings doesn't require a dataset - use SYNTHESIZE_FINDINGS instead
            objective = f"Follow up on novel finding: {gap['text'][:100]}"
            return Task(
                task_id=str(uuid4()),
                task_type=TaskType.SYNTHESIZE_FINDINGS,
                status=TaskStatus.PENDING,
                objective=objective,
                context={
                    "finding_id": gap["finding_id"],
                    "focus": "implications",
                },
            )
        else:
            # No dataset available - skip creating this task
            # ANALYZE_DATA tasks require a dataset_path to be useful
            return None

        return Task(
            task_id=str(uuid4()),
            task_type=TaskType.ANALYZE_DATA,
            status=TaskStatus.PENDING,
            objective=objective,
            context=context,
        )

    def _create_literature_task(self, gap: Dict[str, Any], cycle_id: str) -> Task:
        """
        Create a literature search task for weak evidence.

        Args:
            gap: Gap information from _identify_knowledge_gaps
            cycle_id: Cycle ID to attach task to

        Returns:
            New Task object
        """
        if "hypothesis_id" in gap:
            objective = f"Search literature to validate: {gap['text'][:100]}"
            context = {
                "hypothesis_id": gap["hypothesis_id"],
                "max_papers": 10,
                "focus": "validation",
            }
        else:
            objective = f"Search literature for: {gap['text'][:100]}"
            context = {"max_papers": 10}

        return Task(
            task_id=str(uuid4()),
            task_type=TaskType.SEARCH_LITERATURE,
            status=TaskStatus.PENDING,
            objective=objective,
            context=context,
        )

    def _create_hypothesis_task(self, cycle_id: str, novel_findings: List[Dict[str, Any]]) -> Task:
        """
        Create a hypothesis generation task.

        Args:
            cycle_id: Cycle ID to attach task to
            novel_findings: List of novel findings to base hypotheses on

        Returns:
            New Task object
        """
        if novel_findings:
            finding_texts = [f["text"][:50] for f in novel_findings[:3]]
            objective = f"Generate hypotheses from {len(novel_findings)} novel findings"
            context = {
                "finding_ids": [f["finding_id"] for f in novel_findings],
                "max_hypotheses": 5,
            }
        else:
            objective = "Generate new hypotheses from current knowledge"
            context = {"max_hypotheses": 3}

        return Task(
            task_id=str(uuid4()),
            task_type=TaskType.GENERATE_HYPOTHESIS,
            status=TaskStatus.PENDING,
            objective=objective,
            context=context,
        )

    def _propose_next_tasks(
        self,
        cycle_id: str,
        max_parallel_tasks: int = 5
    ) -> List[Task]:
        """
        Propose tasks with intelligent prioritization.

        This method analyzes the world model to identify knowledge gaps and
        creates a prioritized list of tasks to address them.

        Args:
            cycle_id: The cycle ID for task assignment
            max_parallel_tasks: Maximum number of tasks to propose

        Returns:
            List of Task objects, sorted by priority
        """
        # Analyze world model for gaps
        gaps = self._identify_knowledge_gaps()

        # Score potential tasks
        task_candidates = []

        # Data analysis tasks for unexplored datasets
        for gap in gaps["unexplored_data"]:
            task = self._create_analysis_task(gap, cycle_id)
            if task is not None:  # Skip if no dataset available
                score = self._score_task_priority(gap, type="data_analysis")
                task_candidates.append((score, task))

        # Literature search for under-supported claims
        for gap in gaps["weak_evidence"]:
            score = self._score_task_priority(gap, type="literature")
            task = self._create_literature_task(gap, cycle_id)
            task_candidates.append((score, task))

        # Hypothesis generation for novel findings
        if gaps["novel_findings_count"] > 3:
            score = self._score_task_priority(
                {
                    "novelty": 0.8,
                    "validation_strength": 0.3,
                    "objective_alignment": 0.7,
                },
                type="hypothesis_gen"
            )
            task = self._create_hypothesis_task(cycle_id, gaps["novel_findings"])
            task_candidates.append((score, task))

        # Sort by priority and return top N
        task_candidates.sort(reverse=True, key=lambda x: x[0])

        # Extract tasks (without scores) and limit to max_parallel_tasks
        return [task for score, task in task_candidates[:max_parallel_tasks]]

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"Orchestrator(cycles={stats['total_cycles']}, "
            f"active={stats['active_cycles']}, "
            f"tasks={stats['total_tasks']})"
        )
