import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.llm_client import get_llm_client
from src.utils.cost_tracker import CostTracker


@dataclass
class PlanningOutcome:
    """Record of a cycle's planning decisions and results."""
    cycle_id: str
    cycle_number: int
    objective: str
    tasks_planned: List[Dict[str, Any]]  # list of {task_type, objective} planned
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    findings_produced: int = 0
    hypotheses_generated: int = 0
    hypotheses_tested: int = 0
    avg_finding_confidence: float = 0.0
    avg_finding_novelty: float = 0.0
    cycle_cost: float = 0.0
    cycle_duration_seconds: float = 0.0
    reflection: str = ""  # LLM reflection on what worked/didn't
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "cycle_number": self.cycle_number,
            "objective": self.objective,
            "tasks_planned": self.tasks_planned,
            "tasks_succeeded": self.tasks_succeeded,
            "tasks_failed": self.tasks_failed,
            "findings_produced": self.findings_produced,
            "hypotheses_generated": self.hypotheses_generated,
            "hypotheses_tested": self.hypotheses_tested,
            "avg_finding_confidence": self.avg_finding_confidence,
            "avg_finding_novelty": self.avg_finding_novelty,
            "cycle_cost": self.cycle_cost,
            "cycle_duration_seconds": self.cycle_duration_seconds,
            "reflection": self.reflection,
            "timestamp": self.timestamp,
        }


class PlanningMemory:
    """
    Stores planning outcomes and provides context for future planning.

    After each cycle:
    1. Record what was planned vs what happened
    2. LLM reflects on effectiveness
    3. Store the outcome

    Before each cycle:
    1. Provide recent outcomes as planning context
    2. Include reflections so the planner learns from history
    """

    def __init__(self, max_history: int = 10):
        self.outcomes: List[PlanningOutcome] = []
        self.max_history = max_history
        self.llm_client = get_llm_client()
        self.total_cost: float = 0.0

    async def record_outcome(
        self,
        cycle,  # Cycle object from cycle_manager
        cycle_number: int,
        world_model,  # WorldModel
    ) -> PlanningOutcome:
        """
        Record a cycle's planning outcome and generate a reflection.

        Args:
            cycle: Completed Cycle object
            cycle_number: 1-indexed cycle number
            world_model: WorldModel to count findings from this cycle

        Returns:
            PlanningOutcome with reflection
        """
        from src.orchestrator.cycle_manager import TaskStatus, TaskType

        # Compute metrics from the cycle
        tasks_planned = []
        tasks_succeeded = 0
        tasks_failed = 0

        for task in cycle.tasks:
            tasks_planned.append({
                "task_type": task.task_type.value,
                "objective": task.objective[:100],
            })
            if task.status == TaskStatus.COMPLETED:
                tasks_succeeded += 1
            elif task.status == TaskStatus.FAILED:
                tasks_failed += 1

        # Count findings created during this cycle
        findings_produced = 0
        finding_confidences = []
        finding_novelties = []
        hypotheses_generated = 0
        hypotheses_tested = 0

        if cycle.started_at:
            for node_id, data in world_model.graph.nodes(data=True):
                created_at = data.get("created_at")
                if not created_at:
                    continue
                if isinstance(created_at, str):
                    try:
                        from datetime import datetime
                        created_at = datetime.fromisoformat(created_at)
                    except (ValueError, AttributeError):
                        continue

                if created_at >= cycle.started_at:
                    node_type = data.get("node_type")
                    if node_type == "finding":
                        findings_produced += 1
                        finding_confidences.append(data.get("confidence", 0.0))
                        finding_novelties.append(data.get("metadata", {}).get("novelty", 0.5))
                    elif node_type == "hypothesis":
                        meta = data.get("metadata", {})
                        if meta.get("test_outcome"):
                            hypotheses_tested += 1
                        else:
                            hypotheses_generated += 1

        avg_confidence = sum(finding_confidences) / len(finding_confidences) if finding_confidences else 0.0
        avg_novelty = sum(finding_novelties) / len(finding_novelties) if finding_novelties else 0.0

        duration = 0.0
        if cycle.started_at and cycle.completed_at:
            duration = (cycle.completed_at - cycle.started_at).total_seconds()

        outcome = PlanningOutcome(
            cycle_id=cycle.cycle_id,
            cycle_number=cycle_number,
            objective=cycle.objective,
            tasks_planned=tasks_planned,
            tasks_succeeded=tasks_succeeded,
            tasks_failed=tasks_failed,
            findings_produced=findings_produced,
            hypotheses_generated=hypotheses_generated,
            hypotheses_tested=hypotheses_tested,
            avg_finding_confidence=round(avg_confidence, 3),
            avg_finding_novelty=round(avg_novelty, 3),
            cycle_cost=cycle.budget_used,
            cycle_duration_seconds=duration,
        )

        # Generate LLM reflection
        outcome.reflection = await self._generate_reflection(outcome)

        # Store outcome (cap history)
        self.outcomes.append(outcome)
        if len(self.outcomes) > self.max_history:
            self.outcomes = self.outcomes[-self.max_history:]

        return outcome

    async def _generate_reflection(self, outcome: PlanningOutcome) -> str:
        """
        LLM reflects on what worked and what should change.

        Returns a short reflection string (2-4 sentences).
        """
        tasks_summary = "\n".join([
            f"  - {t['task_type']}: {t['objective'][:60]}"
            for t in outcome.tasks_planned
        ])

        prompt = f"""Briefly reflect on this research cycle's effectiveness (2-4 sentences).

Cycle objective: {outcome.objective}
Tasks planned:
{tasks_summary}

Results:
- Tasks succeeded: {outcome.tasks_succeeded}/{len(outcome.tasks_planned)}
- Tasks failed: {outcome.tasks_failed}
- Findings produced: {outcome.findings_produced}
- Avg finding confidence: {outcome.avg_finding_confidence:.2f}
- Avg finding novelty: {outcome.avg_finding_novelty:.2f}
- Hypotheses generated: {outcome.hypotheses_generated}
- Hypotheses tested: {outcome.hypotheses_tested}
- Cost: ${outcome.cycle_cost:.2f}

What worked well? What should be done differently next cycle? Be specific and actionable.
Respond with ONLY the reflection text, no formatting."""

        try:
            model = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
            response = self.llm_client.create_message(
                model=model,
                max_tokens=300,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}],
            )

            self.total_cost += CostTracker.calculate_cost(
                model, response.usage.input_tokens, response.usage.output_tokens
            )

            text = ""
            for block in response.content:
                if block.type == "text":
                    text += block.text

            return text.strip()

        except Exception as e:
            print(f"Warning: Could not generate planning reflection: {e}")
            return ""

    def get_planning_context(self, max_outcomes: int = 3) -> str:
        """
        Format recent planning outcomes as context for the next cycle's planner.

        Args:
            max_outcomes: Number of recent outcomes to include

        Returns:
            Formatted string to append to planning prompt, or empty string if no history
        """
        recent = self.outcomes[-max_outcomes:]

        if not recent:
            return ""

        lines = [
            "\n## Planning History (learn from previous cycles)",
            "Use these outcomes to improve your task planning. Avoid repeating mistakes.\n"
        ]

        for outcome in recent:
            lines.append(f"### Cycle {outcome.cycle_number}: {outcome.objective[:60]}")
            lines.append(f"Tasks: {outcome.tasks_succeeded} succeeded, {outcome.tasks_failed} failed of {len(outcome.tasks_planned)} planned")
            lines.append(f"Output: {outcome.findings_produced} findings (avg confidence: {outcome.avg_finding_confidence:.2f}, avg novelty: {outcome.avg_finding_novelty:.2f})")
            lines.append(f"Cost: ${outcome.cycle_cost:.2f}")
            if outcome.reflection:
                lines.append(f"Reflection: {outcome.reflection}")
            lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "outcomes": [o.to_dict() for o in self.outcomes],
            "total_cost": self.total_cost,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanningMemory":
        memory = cls()
        memory.total_cost = data.get("total_cost", 0.0)
        for o_data in data.get("outcomes", []):
            outcome = PlanningOutcome(**{k: v for k, v in o_data.items() if k != "timestamp"})
            outcome.timestamp = o_data.get("timestamp", "")
            memory.outcomes.append(outcome)
        return memory
