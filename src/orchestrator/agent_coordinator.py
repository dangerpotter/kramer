"""
AgentCoordinator - Provides unified interface for executing agents.

This module coordinates the execution of different specialized agents
(data analysis, literature search, hypothesis generation, etc.) and
returns structured results.
"""

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.kramer.data_analysis_agent import AgentConfig, DataAnalysisAgent
from src.kramer.hypothesis_agent import HypothesisAgent
from src.orchestrator.cycle_manager import Task
from src.world_model.graph import WorldModel

# Import LiteratureAgent - try multiple paths as it may be in different locations
try:
    from kramer.literature_agent import LiteratureAgent
except ImportError:
    try:
        from kramer.agents.literature import LiteratureAgent
    except ImportError:
        # If neither import works, we'll create a simple fallback
        LiteratureAgent = None


@dataclass
class TaskResult:
    """Result from executing a task through an agent."""

    success: bool
    task_id: str
    task_type: str
    findings: list[Dict[str, Any]]
    cost: float  # API cost in dollars
    metadata: Dict[str, Any]
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "findings": self.findings,
            "cost": self.cost,
            "metadata": self.metadata,
            "error": self.error,
        }


class AgentCoordinator:
    """
    Coordinates execution of specialized agents.

    Provides a unified interface for running different types of agents
    (data analysis, literature search, hypothesis generation, etc.) and
    ensures consistent result formatting.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        notebooks_dir: Path = Path("outputs/notebooks"),
        plots_dir: Path = Path("outputs/plots"),
    ):
        """
        Initialize the agent coordinator.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            notebooks_dir: Directory for saving analysis notebooks
            plots_dir: Directory for saving plots
        """
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.notebooks_dir = notebooks_dir
        self.plots_dir = plots_dir

        # Ensure directories exist
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)

    async def execute_data_analysis(
        self,
        task: Task,
        world_model: WorldModel,
    ) -> TaskResult:
        """
        Execute a data analysis task.

        Args:
            task: Task object with objective and context
            world_model: World model for context and storing results

        Returns:
            TaskResult with analysis findings
        """
        try:
            # Extract parameters from task context
            dataset_path = task.context.get("dataset_path")
            if not dataset_path:
                return TaskResult(
                    success=False,
                    task_id=task.task_id,
                    task_type=task.task_type.value,
                    findings=[],
                    cost=0.0,
                    metadata={},
                    error="No dataset_path provided in task context",
                )

            # Get world model context for the agent
            world_model_context = self._extract_world_model_context(world_model)

            # Create and run data analysis agent
            agent_config = AgentConfig(
                api_key=self.api_key,
                model="claude-sonnet-4-20250514",
                use_extended_thinking=True,
                max_iterations=task.context.get("max_iterations", 5),
            )

            agent = DataAnalysisAgent(
                config=agent_config,
                notebooks_dir=self.notebooks_dir,
                plots_dir=self.plots_dir,
            )

            # Run analysis in thread pool to avoid blocking
            result = await asyncio.to_thread(
                agent.analyze,
                objective=task.objective,
                dataset_path=dataset_path,
                world_model_context=world_model_context,
            )

            # Extract findings
            findings = result.get("findings", [])

            # Calculate cost (rough estimate based on tokens)
            # TODO: Get actual cost from agent
            estimated_cost = 0.05  # Placeholder

            return TaskResult(
                success=True,
                task_id=task.task_id,
                task_type=task.task_type.value,
                findings=findings,
                cost=estimated_cost,
                metadata={
                    "notebook_path": result.get("notebook_path"),
                    "steps": len(result.get("steps", [])),
                    "world_model_updates": result.get("world_model_updates", []),
                },
            )

        except Exception as e:
            return TaskResult(
                success=False,
                task_id=task.task_id,
                task_type=task.task_type.value,
                findings=[],
                cost=0.0,
                metadata={},
                error=str(e),
            )

    async def execute_literature_search(
        self,
        task: Task,
        world_model: WorldModel,
    ) -> TaskResult:
        """
        Execute a literature search task.

        Args:
            task: Task object with search query and context
            world_model: World model for context

        Returns:
            TaskResult with papers and claims found
        """
        try:
            if LiteratureAgent is None:
                return TaskResult(
                    success=False,
                    task_id=task.task_id,
                    task_type=task.task_type.value,
                    findings=[],
                    cost=0.0,
                    metadata={},
                    error="LiteratureAgent not available",
                )

            # Create literature agent
            agent = LiteratureAgent()

            # Determine search approach
            hypothesis = task.context.get("hypothesis")
            max_papers = task.context.get("max_papers", 5)

            # Run in thread pool to avoid blocking (if agent has blocking operations)
            if hypothesis:
                # Search for hypothesis validation
                result = await asyncio.to_thread(agent.search_for_hypothesis, hypothesis)
            else:
                # General search based on objective
                papers = await asyncio.to_thread(agent.search, task.objective, max_results=max_papers)
                result = {
                    "task": "literature_search",
                    "query": task.objective,
                    "papers": papers,
                    "findings": [f"Found {len(papers)} relevant papers"],
                }

            # Format findings
            findings = []
            for paper in result.get("papers", []):
                findings.append(
                    {
                        "type": "paper",
                        "title": paper.get("title"),
                        "authors": paper.get("authors"),
                        "year": paper.get("year"),
                        "relevance_score": paper.get("relevance_score", 0.0),
                        "abstract": paper.get("abstract"),
                    }
                )

            # Add text findings
            for finding_text in result.get("findings", []):
                findings.append(
                    {
                        "type": "insight",
                        "text": finding_text,
                    }
                )

            return TaskResult(
                success=True,
                task_id=task.task_id,
                task_type=task.task_type.value,
                findings=findings,
                cost=0.0,  # Mock agent, no cost
                metadata={
                    "papers_found": len(result.get("papers", [])),
                    "query": task.objective,
                },
            )

        except Exception as e:
            return TaskResult(
                success=False,
                task_id=task.task_id,
                task_type=task.task_type.value,
                findings=[],
                cost=0.0,
                metadata={},
                error=str(e),
            )

    async def execute_hypothesis_generation(
        self,
        task: Task,
        world_model: WorldModel,
    ) -> TaskResult:
        """
        Execute hypothesis generation task.

        Args:
            task: Task object with context
            world_model: World model with current findings

        Returns:
            TaskResult with generated hypotheses
        """
        try:
            # Extract parameters from task context
            current_cycle = task.context.get("current_cycle")
            max_hypotheses = task.context.get("max_hypotheses", 5)
            min_finding_confidence = task.context.get("min_finding_confidence", 0.6)

            # Create hypothesis agent
            agent = HypothesisAgent(
                world_model=world_model,
                api_key=self.api_key,
                max_hypotheses=max_hypotheses,
            )

            # Generate hypotheses in thread pool
            result = await asyncio.to_thread(
                agent.generate_hypotheses,
                current_cycle=current_cycle,
                min_finding_confidence=min_finding_confidence,
            )

            # Format findings
            findings = []
            for hyp_id, hyp_data in zip(result.hypothesis_ids, result.raw_hypotheses):
                findings.append(
                    {
                        "type": "hypothesis",
                        "id": hyp_id,
                        "text": hyp_data.get("statement", ""),
                        "rationale": hyp_data.get("rationale", ""),
                        "testability": hyp_data.get("testability", ""),
                        "confidence": hyp_data.get("novelty_score", 0.0),
                    }
                )

            return TaskResult(
                success=True,
                task_id=task.task_id,
                task_type=task.task_type.value,
                findings=findings,
                cost=result.cost,
                metadata={
                    "hypotheses_generated": result.hypotheses_generated,
                    "hypothesis_ids": result.hypothesis_ids,
                },
            )

        except Exception as e:
            return TaskResult(
                success=False,
                task_id=task.task_id,
                task_type=task.task_type.value,
                findings=[],
                cost=0.0,
                metadata={},
                error=str(e),
            )

    async def execute_hypothesis_test(
        self,
        task: Task,
        world_model: WorldModel,
    ) -> TaskResult:
        """
        Execute hypothesis testing task.

        This is a stub for future HypothesisTesterAgent implementation.

        Args:
            task: Task object with hypothesis to test
            world_model: World model with data and context

        Returns:
            TaskResult with test results
        """
        # TODO: Implement actual HypothesisTesterAgent
        # For now, return a stub result
        hypothesis = task.context.get("hypothesis", "Unknown hypothesis")

        return TaskResult(
            success=True,
            task_id=task.task_id,
            task_type=task.task_type.value,
            findings=[
                {
                    "type": "test_result",
                    "hypothesis": hypothesis,
                    "result": "not_tested",
                    "note": "HypothesisTesterAgent not yet implemented",
                }
            ],
            cost=0.0,
            metadata={
                "status": "stub_implementation",
                "hypothesis": hypothesis,
            },
        )

    def _extract_world_model_context(
        self,
        world_model: WorldModel,
    ) -> Dict[str, Any]:
        """
        Extract relevant context from world model for agent use.

        Args:
            world_model: The world model

        Returns:
            Dictionary with relevant context
        """
        context = {
            "total_nodes": world_model.graph.number_of_nodes(),
            "total_edges": world_model.graph.number_of_edges(),
            "hypotheses": [],
            "recent_findings": [],
        }

        # Get recent hypotheses
        for node_id, data in world_model.graph.nodes(data=True):
            if data.get("node_type") == "hypothesis":
                context["hypotheses"].append(
                    {
                        "id": node_id,
                        "text": data.get("text", ""),
                        "confidence": data.get("confidence", 0.0),
                    }
                )

        # Get recent findings
        for node_id, data in world_model.graph.nodes(data=True):
            if data.get("node_type") == "finding":
                context["recent_findings"].append(
                    {
                        "id": node_id,
                        "text": data.get("text", ""),
                        "confidence": data.get("confidence", 0.0),
                    }
                )

        # Limit to most recent/relevant
        context["hypotheses"] = context["hypotheses"][:10]
        context["recent_findings"] = context["recent_findings"][:10]

        return context
