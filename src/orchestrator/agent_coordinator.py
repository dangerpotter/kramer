"""
AgentCoordinator - Provides unified interface for executing agents.

This module coordinates the execution of different specialized agents
(data analysis, literature search, hypothesis generation, etc.) and
returns structured results.
"""

import asyncio
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.kramer.data_analysis_agent import AgentConfig, DataAnalysisAgent
from src.kramer.hypothesis_agent import HypothesisAgent
from src.kramer.hypothesis_tester_agent import HypothesisTesterAgent
from src.orchestrator.cycle_manager import Task
from src.world_model.graph import EdgeType, NodeType, WorldModel
from src.utils.embedding_service import get_embedding_service, EmbeddingService

# Import LiteratureAgent (prefer multi-source if available)
from src.kramer.literature_agent import LiteratureAgent

try:
    from kramer.agents.literature import LiteratureAgent as MultiSourceLiteratureAgent
except ImportError:
    MultiSourceLiteratureAgent = None


def _tokenize(text: str) -> Set[str]:
    """
    Simple tokenization: lowercase and split on non-alphanumeric characters.
    Removes common stopwords for better similarity comparison.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "under",
        "and", "but", "or", "nor", "so", "yet", "both", "either", "neither",
        "not", "only", "same", "than", "too", "very", "just", "that", "this",
        "these", "those", "it", "its", "they", "their", "them", "we", "our",
    }
    words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
    return words - stopwords


def _compute_text_similarity(text1: str, text2: str) -> float:
    """
    Compute Jaccard similarity between two texts.
    Returns value between 0 (completely different) and 1 (identical).
    """
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


def calculate_finding_novelty(
    finding_text: str,
    world_model: WorldModel,
    use_embeddings: bool = True,
    weight_by_recency: bool = True,
    weight_by_confidence: bool = True,
) -> float:
    """
    Calculate novelty score for a finding based on semantic similarity to existing findings.

    Uses embedding-based similarity for better semantic matching. The novelty score
    considers:
    - Semantic similarity to existing findings (via embeddings)
    - Recency weighting (newer findings count more)
    - Confidence weighting (high-confidence findings count more)

    Novelty is the inverse of the weighted maximum similarity to existing findings.
    A completely unique finding has novelty 1.0, while a duplicate has novelty 0.0.

    Args:
        finding_text: The text of the new finding
        world_model: WorldModel containing existing findings
        use_embeddings: Whether to use embedding-based similarity (True) or Jaccard
        weight_by_recency: Weight similarity by how recent findings are
        weight_by_confidence: Weight similarity by finding confidence

    Returns:
        Novelty score between 0.0 and 1.0
    """
    # Collect existing findings with metadata
    existing_findings: List[Dict[str, Any]] = []
    for node_data in world_model.query_nodes(NodeType.FINDING):
        existing_text = node_data.get("text", "")
        if existing_text:
            existing_findings.append({
                "text": existing_text,
                "confidence": node_data.get("confidence", 0.5),
                "created_at": node_data.get("created_at"),
                "metadata": node_data.get("metadata", {}),
            })

    if not existing_findings:
        # First finding is maximally novel
        return 1.0

    # Get embedding service
    embedding_service = get_embedding_service()

    # Compute similarities
    if use_embeddings and embedding_service.is_available():
        # Use embedding-based semantic similarity
        existing_texts = [f["text"] for f in existing_findings]
        finding_emb = embedding_service.get_embedding(finding_text)

        if finding_emb is not None:
            existing_embs = embedding_service.get_embeddings_batch(existing_texts)
            similarities = []

            for i, (finding, emb) in enumerate(zip(existing_findings, existing_embs)):
                if emb is None:
                    sim = _compute_text_similarity(finding_text, finding["text"])
                else:
                    sim = EmbeddingService._cosine_similarity(finding_emb, emb)

                # Apply weighting
                weight = 1.0

                # Recency weighting: recent findings count more
                if weight_by_recency and finding.get("created_at"):
                    created_at = finding["created_at"]
                    if isinstance(created_at, str):
                        try:
                            created_at = datetime.fromisoformat(created_at)
                        except (ValueError, AttributeError):
                            created_at = None

                    if created_at:
                        age_hours = (datetime.utcnow() - created_at).total_seconds() / 3600
                        # Decay factor: findings older than 24 hours count less
                        recency_weight = max(0.5, 1.0 - (age_hours / 48.0))
                        weight *= recency_weight

                # Confidence weighting: high-confidence findings count more
                if weight_by_confidence:
                    confidence = finding.get("confidence", 0.5)
                    # Scale weight by confidence (0.5 to 1.5)
                    confidence_weight = 0.5 + confidence
                    weight *= confidence_weight

                # Apply weight to similarity
                weighted_sim = sim * weight
                similarities.append(weighted_sim)

            max_similarity = max(similarities) if similarities else 0.0
        else:
            # Fallback to Jaccard if embedding fails
            max_similarity = max(
                _compute_text_similarity(finding_text, f["text"])
                for f in existing_findings
            )
    else:
        # Use Jaccard similarity as fallback
        max_similarity = 0.0
        for finding in existing_findings:
            sim = _compute_text_similarity(finding_text, finding["text"])

            # Apply confidence weighting even for Jaccard
            if weight_by_confidence:
                confidence = finding.get("confidence", 0.5)
                sim *= (0.5 + confidence)

            max_similarity = max(max_similarity, sim)

    # Novelty is inverse of weighted similarity
    # Clamp to [0, 1] range since weighting can push values above 1
    novelty = max(0.0, min(1.0, 1.0 - max_similarity))

    return round(novelty, 3)


def calculate_topic_novelty(
    finding_text: str,
    world_model: WorldModel,
    objective: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calculate topic/domain novelty for a finding.

    Analyzes whether the finding covers a new aspect of the research objective
    or explores a topic not yet covered by existing findings.

    Args:
        finding_text: The text of the new finding
        world_model: WorldModel containing existing findings
        objective: The original research objective (for coverage analysis)

    Returns:
        Dictionary with:
        - topic_novelty: Score 0-1 indicating how novel the topic is
        - covers_new_aspect: Boolean indicating if it covers a new aspect
        - closest_topic: Text of the most similar existing finding
        - objective_relevance: Relevance to original objective (if provided)
    """
    embedding_service = get_embedding_service()

    # Collect existing findings
    existing_findings = []
    for node_data in world_model.query_nodes(NodeType.FINDING):
        existing_text = node_data.get("text", "")
        if existing_text:
            existing_findings.append(existing_text)

    result = {
        "topic_novelty": 1.0,
        "covers_new_aspect": True,
        "closest_topic": None,
        "objective_relevance": None,
    }

    if not existing_findings:
        return result

    if embedding_service.is_available():
        # Find max similarity and closest finding
        max_sim, max_idx = embedding_service.compute_max_similarity(
            finding_text, existing_findings, return_index=True
        )

        # Topic novelty is inverse of max similarity
        # but with a softer threshold (topics can be related but still novel)
        TOPIC_SIMILARITY_THRESHOLD = 0.75
        if max_sim > TOPIC_SIMILARITY_THRESHOLD:
            result["topic_novelty"] = round(1.0 - max_sim, 3)
            result["covers_new_aspect"] = False
        else:
            result["topic_novelty"] = round(max(0.5, 1.0 - max_sim * 0.5), 3)
            result["covers_new_aspect"] = True

        if max_idx >= 0:
            result["closest_topic"] = existing_findings[max_idx][:100]

        # Check relevance to objective
        if objective:
            obj_similarity = embedding_service.compute_similarity(finding_text, objective)
            result["objective_relevance"] = round(obj_similarity, 3)
    else:
        # Fallback to Jaccard
        max_sim = max(
            _compute_text_similarity(finding_text, f) for f in existing_findings
        )
        result["topic_novelty"] = round(1.0 - max_sim, 3)
        result["covers_new_aspect"] = max_sim < 0.5

    return result


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
                model=os.getenv("CLAUDE_MODEL"),
                use_extended_thinking=task.context.get("use_extended_thinking", True),
                max_iterations=task.context.get("max_iterations", 5),
                max_attempts_per_step=task.context.get("max_attempts_per_step", 3),
                quality_threshold=task.context.get("quality_threshold", 0.7),
                step_timeout=task.context.get("step_timeout", 120),
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

            # Extract actual cost from agent
            actual_cost = result.get("cost", 0.0)

            return TaskResult(
                success=True,
                task_id=task.task_id,
                task_type=task.task_type.value,
                findings=findings,
                cost=actual_cost,
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

            # Prefer multi-source literature agent if available and configured
            agent = None
            if MultiSourceLiteratureAgent is not None:
                has_keys = any([
                    os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
                    os.getenv("CORE_API_KEY"),
                    os.getenv("NCBI_API_KEY"),
                ])
                if has_keys:
                    try:
                        agent = MultiSourceLiteratureAgent()
                    except Exception:
                        pass  # Fall back to basic agent

            if agent is None:
                agent = LiteratureAgent()

            # Determine search approach
            hypothesis = task.context.get("hypothesis")
            max_papers = task.context.get("max_papers", 5)

            # Run async methods directly (they're already async)
            if hypothesis:
                # Search for hypothesis validation
                result = await agent.search_for_hypothesis(hypothesis)
            else:
                # General search based on objective
                papers = await agent.search(task.objective, max_results=max_papers)
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

            # Store papers in world model for later retrieval
            # First, collect existing paper IDs to avoid duplicates
            existing_paper_ids = set()
            for node_data in world_model.query_nodes(NodeType.PAPER):
                meta = node_data.get("metadata", {})
                if meta.get("semantic_scholar_id"):
                    existing_paper_ids.add(meta["semantic_scholar_id"])
                if meta.get("arxiv_id"):
                    existing_paper_ids.add(meta["arxiv_id"])
                # Also check DOI for deduplication
                if meta.get("doi"):
                    existing_paper_ids.add(meta["doi"])

            papers_added = 0
            papers_skipped = 0
            for paper in result.get("papers", []):
                # Check if paper already exists in world model
                paper_id = paper.get("paperId") or paper.get("arxiv_id") or paper.get("doi")
                if paper_id and paper_id in existing_paper_ids:
                    papers_skipped += 1
                    continue  # Skip duplicate paper

                try:
                    world_model.add_paper(
                        text=paper.get("abstract", paper.get("title", "No abstract available")),
                        title=paper.get("title", "Unknown Title"),
                        authors=paper.get("authors", []),
                        year=paper.get("year"),
                        doi=paper.get("doi"),
                        metadata={
                            "url": paper.get("url"),
                            "relevance_score": paper.get("relevance_score", 0.0),
                            "source": paper.get("source", "unknown"),
                            "query": task.objective,
                            "arxiv_id": paper.get("arxiv_id"),
                            "semantic_scholar_id": paper.get("paperId"),
                        }
                    )
                    papers_added += 1
                    # Track the paper ID to avoid adding it again in this batch
                    if paper_id:
                        existing_paper_ids.add(paper_id)
                except Exception as e:
                    print(f"Warning: Could not add paper to world model: {e}")

            if papers_skipped > 0:
                print(f"   📚 Papers: {papers_added} added, {papers_skipped} skipped (duplicates)")

            # Extract cost from result (if available)
            actual_cost = result.get("cost", 0.0)

            return TaskResult(
                success=True,
                task_id=task.task_id,
                task_type=task.task_type.value,
                findings=findings,
                cost=actual_cost,
                metadata={
                    "papers_found": len(result.get("papers", [])),
                    "papers_added": papers_added,
                    "papers_skipped_duplicates": papers_skipped,
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
            objective = task.context.get("objective") or task.objective

            # Create hypothesis agent
            agent = HypothesisAgent(
                world_model=world_model,
                api_key=self.api_key,
                max_hypotheses=max_hypotheses,
            )

            # Generate hypotheses in thread pool
            result = await asyncio.to_thread(
                agent.generate_hypotheses,
                objective=objective,
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

        Args:
            task: Task object with hypothesis to test
            world_model: World model with data and context

        Returns:
            TaskResult with test results
        """
        try:
            # Extract test parameters from task context
            # Convert to string in case Claude returned an integer
            raw_hypothesis_id = task.context.get("hypothesis_id")
            hypothesis_id = str(raw_hypothesis_id) if raw_hypothesis_id else None
            dataset_path = task.context.get("dataset_path")
            test_approaches = task.context.get("test_approaches", ["both"])

            if not hypothesis_id:
                print(f"  ✗ Hypothesis test failed: No hypothesis_id provided")
                return TaskResult(
                    success=False,
                    task_id=task.task_id,
                    task_type=task.task_type.value,
                    findings=[],
                    cost=0.0,
                    metadata={},
                    error="No hypothesis_id provided in task context",
                )

            # Get hypothesis text for logging
            hypothesis_text = ""
            if world_model.graph.has_node(hypothesis_id):
                node_data = world_model.graph.nodes[hypothesis_id]
                hypothesis_text = node_data.get("text", "")[:80]
                if len(node_data.get("text", "")) > 80:
                    hypothesis_text += "..."

            print(f"  Testing hypothesis {hypothesis_id[:8]}...")
            if hypothesis_text:
                print(f"    \"{hypothesis_text}\"")

            # Create HypothesisTesterAgent
            tester = HypothesisTesterAgent(
                world_model=world_model,
                api_key=self.api_key,
                model=os.getenv("CLAUDE_MODEL"),
                use_extended_thinking=task.context.get("use_extended_thinking", True),
            )

            # Run hypothesis test
            test_result = tester.test_hypothesis(
                hypothesis_id=hypothesis_id,
                dataset_path=dataset_path,
                test_approaches=test_approaches,
            )

            # Log the test result
            outcome_emoji = {
                "supported": "✓",
                "refuted": "✗",
                "inconclusive": "?",
            }.get(test_result.outcome, "•")
            print(
                f"  {outcome_emoji} Hypothesis {hypothesis_id[:8]}: "
                f"{test_result.outcome.upper()} (confidence: {test_result.confidence:.2f}) "
                f"via {test_result.test_type}"
            )

            # Update world model with test results
            self._update_world_model_with_test_results(
                world_model=world_model,
                test_result=test_result,
            )

            # Convert to findings format
            findings = [
                {
                    "type": "hypothesis_test_result",
                    "hypothesis_id": test_result.hypothesis_id,
                    "outcome": test_result.outcome,
                    "confidence": test_result.confidence,
                    "test_type": test_result.test_type,
                    "reasoning": test_result.reasoning,
                },
            ]

            # Add evidence as separate findings
            for evidence in test_result.evidence:
                findings.append(
                    {
                        "type": "test_evidence",
                        "evidence_type": evidence.get("type"),
                        "supports": evidence.get("supports"),
                        "confidence": evidence.get("confidence", 0.5),
                        "details": evidence,
                    }
                )

            return TaskResult(
                success=True,
                task_id=task.task_id,
                task_type=task.task_type.value,
                findings=findings,
                cost=test_result.cost,
                metadata={
                    "hypothesis_id": test_result.hypothesis_id,
                    "hypothesis_ids": [test_result.hypothesis_id],  # Array format for _schedule_hypothesis_tests
                    "outcome": test_result.outcome,
                    "confidence": test_result.confidence,
                    "test_type": test_result.test_type,
                    "statistical_metrics": test_result.statistical_metrics,
                    "evidence_count": len(test_result.evidence),
                },
            )

        except Exception as e:
            import traceback

            error_msg = f"Error executing hypothesis test: {str(e)}"
            hyp_id_display = str(task.context.get('hypothesis_id', 'unknown'))[:8]
            print(f"  ✗ Hypothesis {hyp_id_display}: FAILED - {str(e)}")
            print(traceback.format_exc())

            return TaskResult(
                success=False,
                task_id=task.task_id,
                task_type=task.task_type.value,
                findings=[],
                cost=0.0,
                metadata={},
                error=error_msg,
            )

    def _update_world_model_with_test_results(
        self,
        world_model: WorldModel,
        test_result,
    ) -> None:
        """
        Update world model with hypothesis test results.

        Args:
            world_model: World model to update
            test_result: TestResult from HypothesisTesterAgent
        """
        hypothesis_id = test_result.hypothesis_id

        # Update hypothesis node with test outcome
        if world_model.graph.has_node(hypothesis_id):
            node_data = world_model.graph.nodes[hypothesis_id]
            metadata = node_data.get("metadata", {})

            # Update metadata with test results
            metadata["tested"] = True
            metadata["tested_at"] = datetime.utcnow().isoformat()  # Track when tested for progress counting
            metadata["test_outcome"] = test_result.outcome
            metadata["test_confidence"] = test_result.confidence
            metadata["test_type"] = test_result.test_type
            metadata["test_reasoning"] = test_result.reasoning

            # Update confidence based on test outcome
            if test_result.outcome == "supported":
                # Increase confidence for supported hypotheses
                new_confidence = min(1.0, test_result.confidence)
            elif test_result.outcome == "refuted":
                # Decrease confidence for refuted hypotheses
                new_confidence = max(0.0, 1.0 - test_result.confidence)
            elif test_result.outcome == "insufficient_evidence":
                # Keep original confidence - hypothesis needs more research
                new_confidence = node_data.get("confidence", 0.5)
            else:
                # Keep original confidence for inconclusive
                new_confidence = node_data.get("confidence", 0.5)

            # Update node
            world_model.graph.nodes[hypothesis_id]["confidence"] = new_confidence
            world_model.graph.nodes[hypothesis_id]["metadata"] = metadata

            # Add evidence findings to world model and link them
            for evidence in test_result.evidence:
                if evidence.get("type") in ["statistical_analysis", "literature_review"]:
                    # Create finding node for evidence
                    finding_text = evidence.get("finding", evidence.get("reasoning", "Test evidence"))

                    # Calculate novelty score for this finding
                    novelty = calculate_finding_novelty(finding_text, world_model)

                    finding_id = world_model.add_finding(
                        text=finding_text,
                        confidence=evidence.get("confidence", 0.5),
                        metadata={
                            "source": evidence.get("source"),
                            "evidence_type": evidence.get("type"),
                            "from_hypothesis_test": hypothesis_id,
                            "novelty": novelty,
                        },
                    )

                    # Link evidence to hypothesis
                    edge_type = EdgeType.SUPPORTS if evidence.get("supports") else EdgeType.REFUTES
                    try:
                        world_model.add_edge(
                            source=finding_id,
                            target=hypothesis_id,
                            edge_type=edge_type,
                            metadata={"test_evidence": True},
                        )
                    except Exception as e:
                        print(f"Warning: Could not add edge from evidence to hypothesis: {e}")

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
