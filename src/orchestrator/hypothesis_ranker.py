"""
Advanced Hypothesis Ranking System with Information Gain and Cost-Benefit Analysis.

Provides intelligent hypothesis prioritization based on:
- Expected information gain
- Cost-benefit analysis
- Novelty scoring
- Testability assessment
- Strategic value
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from src.world_model.graph import WorldModel, NodeType

logger = logging.getLogger(__name__)


class RankingCriterion(Enum):
    """Criteria for ranking hypotheses."""
    INFORMATION_GAIN = "information_gain"
    NOVELTY = "novelty"
    TESTABILITY = "testability"
    COST_BENEFIT = "cost_benefit"
    STRATEGIC_VALUE = "strategic_value"
    CONFIDENCE = "confidence"
    UNCERTAINTY = "uncertainty"


@dataclass
class RankingWeights:
    """Weights for different ranking criteria."""
    information_gain: float = 0.3
    novelty: float = 0.2
    testability: float = 0.2
    cost_benefit: float = 0.15
    strategic_value: float = 0.1
    uncertainty: float = 0.05

    def normalize(self):
        """Normalize weights to sum to 1.0."""
        total = (
            self.information_gain +
            self.novelty +
            self.testability +
            self.cost_benefit +
            self.strategic_value +
            self.uncertainty
        )

        if total > 0:
            self.information_gain /= total
            self.novelty /= total
            self.testability /= total
            self.cost_benefit /= total
            self.strategic_value /= total
            self.uncertainty /= total


@dataclass
class HypothesisScore:
    """Comprehensive score for a hypothesis."""
    hypothesis_id: str
    hypothesis_text: str

    # Individual scores (0-1)
    information_gain: float = 0.0
    novelty: float = 0.0
    testability: float = 0.0
    cost_benefit: float = 0.0
    strategic_value: float = 0.0
    uncertainty: float = 0.0

    # Composite score
    composite_score: float = 0.0
    rank: Optional[int] = None

    # Metadata
    estimated_cost: float = 0.0
    expected_evidence_count: int = 0
    related_findings_count: int = 0
    test_status: str = "untested"

    # Timestamp
    scored_at: datetime = field(default_factory=datetime.utcnow)


class HypothesisRanker:
    """
    Advanced hypothesis ranking system.

    Implements multi-criteria ranking with:
    - Information-theoretic scoring
    - Cost-benefit optimization
    - Strategic prioritization
    - Adaptive weighting
    """

    def __init__(
        self,
        world_model: WorldModel,
        weights: Optional[RankingWeights] = None,
        research_objective: Optional[str] = None
    ):
        """
        Initialize hypothesis ranker.

        Args:
            world_model: WorldModel instance
            weights: Optional custom ranking weights
            research_objective: Optional research objective for alignment scoring
        """
        self.world_model = world_model
        self.weights = weights or RankingWeights()
        self.weights.normalize()
        self.research_objective = research_objective

        # Cache for expensive computations
        self._score_cache: Dict[str, HypothesisScore] = {}

        logger.info("HypothesisRanker initialized with weights: %s", self.weights)

    def rank_hypotheses(
        self,
        hypothesis_ids: Optional[List[str]] = None,
        filter_tested: bool = False,
        top_k: Optional[int] = None
    ) -> List[HypothesisScore]:
        """
        Rank hypotheses by composite score.

        Args:
            hypothesis_ids: Optional list of specific hypothesis IDs to rank
            filter_tested: If True, exclude already tested hypotheses
            top_k: Return only top K hypotheses

        Returns:
            List of HypothesisScore objects, sorted by rank
        """
        # Get hypotheses to rank
        if hypothesis_ids is None:
            hypothesis_nodes = self.world_model.query_nodes(NodeType.HYPOTHESIS)
            hypothesis_ids = [node["node_id"] for node in hypothesis_nodes]

        # Filter tested if requested
        if filter_tested:
            hypothesis_ids = [
                hid for hid in hypothesis_ids
                if not self._is_tested(hid)
            ]

        logger.info(f"Ranking {len(hypothesis_ids)} hypotheses")

        # Score each hypothesis
        scores = []
        for hyp_id in hypothesis_ids:
            score = self.score_hypothesis(hyp_id)
            if score:
                scores.append(score)

        # Sort by composite score (descending)
        scores.sort(key=lambda s: s.composite_score, reverse=True)

        # Assign ranks
        for i, score in enumerate(scores, 1):
            score.rank = i

        # Return top K if specified
        if top_k:
            scores = scores[:top_k]

        logger.info(f"Ranked {len(scores)} hypotheses")

        return scores

    def score_hypothesis(self, hypothesis_id: str) -> Optional[HypothesisScore]:
        """
        Compute comprehensive score for a hypothesis.

        Args:
            hypothesis_id: Hypothesis identifier

        Returns:
            HypothesisScore or None if hypothesis not found
        """
        # Check cache
        if hypothesis_id in self._score_cache:
            return self._score_cache[hypothesis_id]

        # Get hypothesis from world model
        hypothesis_nodes = self.world_model.query_nodes(
            NodeType.HYPOTHESIS,
            filters={"node_id": hypothesis_id}
        )

        if not hypothesis_nodes:
            logger.warning(f"Hypothesis {hypothesis_id} not found")
            return None

        hypothesis = hypothesis_nodes[0]
        hypothesis_text = hypothesis.get("text", "")

        # Compute individual scores
        info_gain = self._compute_information_gain(hypothesis_id)
        novelty = self._compute_novelty(hypothesis_id)
        testability = self._compute_testability(hypothesis_id)
        cost_benefit = self._compute_cost_benefit(hypothesis_id)
        strategic_value = self._compute_strategic_value(hypothesis_id)
        uncertainty = self._compute_uncertainty(hypothesis_id)

        # Compute composite score
        composite = (
            self.weights.information_gain * info_gain +
            self.weights.novelty * novelty +
            self.weights.testability * testability +
            self.weights.cost_benefit * cost_benefit +
            self.weights.strategic_value * strategic_value +
            self.weights.uncertainty * uncertainty
        )

        # Create score object
        score = HypothesisScore(
            hypothesis_id=hypothesis_id,
            hypothesis_text=hypothesis_text,
            information_gain=info_gain,
            novelty=novelty,
            testability=testability,
            cost_benefit=cost_benefit,
            strategic_value=strategic_value,
            uncertainty=uncertainty,
            composite_score=composite,
            test_status=self._get_test_status(hypothesis_id),
            related_findings_count=len(self._get_related_findings(hypothesis_id))
        )

        # Cache result
        self._score_cache[hypothesis_id] = score

        return score

    def _compute_information_gain(self, hypothesis_id: str) -> float:
        """
        Compute expected information gain from testing hypothesis.

        Information gain considers:
        - Reduction in uncertainty about the world
        - Number of related hypotheses that would be informed
        - Potential to resolve contradictions
        - Coverage of unexplored areas
        """
        # Get related nodes
        related_findings = self._get_related_findings(hypothesis_id)
        related_hypotheses = self._get_related_hypotheses(hypothesis_id)

        # Base information gain from testing
        # More related findings = lower gain (already explored)
        # More related hypotheses = higher gain (resolves multiple questions)

        finding_penalty = 1.0 / (1.0 + len(related_findings) * 0.1)
        hypothesis_bonus = min(1.0, len(related_hypotheses) * 0.2)

        # Check for contradictions (high information value)
        has_contradictions = self._has_contradictions(hypothesis_id)
        contradiction_bonus = 0.3 if has_contradictions else 0.0

        # Check if hypothesis covers unexplored area
        unexplored_bonus = self._get_unexplored_bonus(hypothesis_id)

        # Combine factors
        info_gain = min(1.0,
            0.5 * finding_penalty +
            0.3 * hypothesis_bonus +
            contradiction_bonus +
            unexplored_bonus
        )

        return info_gain

    def _compute_novelty(self, hypothesis_id: str) -> float:
        """
        Compute novelty score for hypothesis.

        Novelty considers:
        - Similarity to existing hypotheses
        - Coverage of new concepts
        - Originality of proposed relationships
        """
        # Get all hypotheses
        all_hypotheses = self.world_model.query_nodes(NodeType.HYPOTHESIS)

        if len(all_hypotheses) <= 1:
            return 1.0  # First hypothesis is maximally novel

        # Get hypothesis text
        hypothesis_nodes = self.world_model.query_nodes(
            NodeType.HYPOTHESIS,
            filters={"node_id": hypothesis_id}
        )

        if not hypothesis_nodes:
            return 0.0

        hypothesis_text = hypothesis_nodes[0].get("text", "").lower()

        # Compute similarity to other hypotheses
        similarities = []
        for other_hyp in all_hypotheses:
            if other_hyp["node_id"] == hypothesis_id:
                continue

            other_text = other_hyp.get("text", "").lower()
            similarity = self._compute_text_similarity(hypothesis_text, other_text)
            similarities.append(similarity)

        # Novelty is inverse of max similarity
        if similarities:
            max_similarity = max(similarities)
            novelty = 1.0 - max_similarity
        else:
            novelty = 1.0

        return max(0.0, novelty)

    def _compute_testability(self, hypothesis_id: str) -> float:
        """
        Compute testability score for hypothesis.

        Testability considers:
        - Availability of relevant data
        - Clarity of prediction
        - Measurability of outcomes
        - Feasibility of testing
        """
        # Check if data is available
        datasets = self.world_model.query_nodes(NodeType.DATASET)
        has_data = len(datasets) > 0

        # Check if hypothesis has clear predictions
        # (In practice, would analyze hypothesis text for specific claims)
        hypothesis_nodes = self.world_model.query_nodes(
            NodeType.HYPOTHESIS,
            filters={"node_id": hypothesis_id}
        )

        if not hypothesis_nodes:
            return 0.0

        hypothesis_text = hypothesis_nodes[0].get("text", "")

        # Heuristic: longer hypotheses with specific terms are more testable
        has_quantitative = any(word in hypothesis_text.lower()
                               for word in ["correlation", "increase", "decrease", "effect", "predict"])

        testability = 0.0

        if has_data:
            testability += 0.5

        if has_quantitative:
            testability += 0.3

        # Check if related findings exist (easier to test)
        related_findings = self._get_related_findings(hypothesis_id)
        if related_findings:
            testability += 0.2

        return min(1.0, testability)

    def _compute_cost_benefit(self, hypothesis_id: str) -> float:
        """
        Compute cost-benefit ratio for testing hypothesis.

        Cost-benefit considers:
        - Estimated computational cost
        - Expected value of information
        - Potential impact on research objective
        - Resource requirements
        """
        # Estimate cost (simplified: based on related nodes)
        related_findings = self._get_related_findings(hypothesis_id)
        related_hypotheses = self._get_related_hypotheses(hypothesis_id)

        # More related nodes = higher estimated cost
        estimated_complexity = len(related_findings) + len(related_hypotheses)
        estimated_cost = min(1.0, estimated_complexity * 0.1)

        # Estimate benefit
        info_gain = self._compute_information_gain(hypothesis_id)
        strategic_value = self._compute_strategic_value(hypothesis_id)

        expected_benefit = (info_gain + strategic_value) / 2.0

        # Cost-benefit ratio (higher is better)
        if estimated_cost > 0:
            cost_benefit = expected_benefit / (estimated_cost + 0.1)  # Add small constant to avoid division by zero
        else:
            cost_benefit = expected_benefit

        # Normalize to 0-1
        cost_benefit = min(1.0, cost_benefit)

        return cost_benefit

    def _compute_strategic_value(self, hypothesis_id: str) -> float:
        """
        Compute strategic value relative to research objective.

        Strategic value considers:
        - Alignment with research objective
        - Potential to unblock other hypotheses
        - Impact on overall discovery trajectory
        """
        if not self.research_objective:
            return 0.5  # Default if no objective specified

        # Get hypothesis text
        hypothesis_nodes = self.world_model.query_nodes(
            NodeType.HYPOTHESIS,
            filters={"node_id": hypothesis_id}
        )

        if not hypothesis_nodes:
            return 0.0

        hypothesis_text = hypothesis_nodes[0].get("text", "").lower()
        objective_lower = self.research_objective.lower()

        # Compute keyword overlap
        objective_words = set(objective_lower.split())
        hypothesis_words = set(hypothesis_text.split())

        common_words = objective_words.intersection(hypothesis_words)
        overlap = len(common_words) / max(len(objective_words), 1)

        # Check if hypothesis could unblock others
        dependent_hypotheses = len(self._get_related_hypotheses(hypothesis_id))
        unblocking_value = min(0.3, dependent_hypotheses * 0.05)

        strategic_value = min(1.0, overlap + unblocking_value)

        return strategic_value

    def _compute_uncertainty(self, hypothesis_id: str) -> float:
        """
        Compute uncertainty score for hypothesis.

        Higher uncertainty = higher priority (more to learn).
        """
        # Get hypothesis confidence
        hypothesis_nodes = self.world_model.query_nodes(
            NodeType.HYPOTHESIS,
            filters={"node_id": hypothesis_id}
        )

        if not hypothesis_nodes:
            return 1.0  # Maximum uncertainty if not found

        confidence = hypothesis_nodes[0].get("confidence", 0.5)

        # Uncertainty is inverse of confidence
        uncertainty = 1.0 - confidence

        return uncertainty

    def _is_tested(self, hypothesis_id: str) -> bool:
        """Check if hypothesis has been tested."""
        hypothesis_nodes = self.world_model.query_nodes(
            NodeType.HYPOTHESIS,
            filters={"node_id": hypothesis_id}
        )

        if not hypothesis_nodes:
            return False

        metadata = hypothesis_nodes[0].get("metadata", {})
        return metadata.get("tested", False)

    def _get_test_status(self, hypothesis_id: str) -> str:
        """Get test status for hypothesis."""
        if not self._is_tested(hypothesis_id):
            return "untested"

        hypothesis_nodes = self.world_model.query_nodes(
            NodeType.HYPOTHESIS,
            filters={"node_id": hypothesis_id}
        )

        if not hypothesis_nodes:
            return "unknown"

        metadata = hypothesis_nodes[0].get("metadata", {})
        return metadata.get("test_outcome", "tested")

    def _get_related_findings(self, hypothesis_id: str) -> List[Dict[str, Any]]:
        """Get findings related to hypothesis."""
        # Get edges to/from hypothesis
        neighbors = self.world_model.get_neighbors(hypothesis_id)

        # Filter for findings
        findings = [
            n for n in neighbors
            if n.get("node_type") == NodeType.FINDING.value
        ]

        return findings

    def _get_related_hypotheses(self, hypothesis_id: str) -> List[Dict[str, Any]]:
        """Get hypotheses related to this hypothesis."""
        neighbors = self.world_model.get_neighbors(hypothesis_id)

        hypotheses = [
            n for n in neighbors
            if n.get("node_type") == NodeType.HYPOTHESIS.value
        ]

        return hypotheses

    def _has_contradictions(self, hypothesis_id: str) -> bool:
        """Check if hypothesis has contradictory evidence."""
        # Get related findings
        findings = self._get_related_findings(hypothesis_id)

        # Check for both supporting and refuting evidence
        has_support = any(
            f.get("metadata", {}).get("relationship") == "supports"
            for f in findings
        )

        has_refutation = any(
            f.get("metadata", {}).get("relationship") == "refutes"
            for f in findings
        )

        return has_support and has_refutation

    def _get_unexplored_bonus(self, hypothesis_id: str) -> float:
        """Compute bonus for hypotheses in unexplored areas."""
        # Check if hypothesis has few related findings
        related_findings = self._get_related_findings(hypothesis_id)

        if len(related_findings) == 0:
            return 0.2  # Unexplored area
        elif len(related_findings) <= 2:
            return 0.1  # Lightly explored
        else:
            return 0.0  # Well explored

    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """
        Compute simple text similarity (Jaccard similarity).

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def clear_cache(self):
        """Clear score cache."""
        self._score_cache.clear()
        logger.debug("Score cache cleared")

    def get_ranking_report(self, top_k: int = 10) -> str:
        """
        Generate human-readable ranking report.

        Args:
            top_k: Number of top hypotheses to include

        Returns:
            Formatted report string
        """
        scores = self.rank_hypotheses(top_k=top_k)

        report = "\n" + "="*80 + "\n"
        report += "HYPOTHESIS RANKING REPORT\n"
        report += "="*80 + "\n\n"

        report += f"Research Objective: {self.research_objective or 'Not specified'}\n"
        report += f"Total Hypotheses Ranked: {len(scores)}\n"
        report += f"Ranking Weights: {self.weights}\n\n"

        report += "Top Hypotheses:\n"
        report += "-"*80 + "\n\n"

        for score in scores:
            report += f"Rank #{score.rank}: {score.hypothesis_text[:80]}...\n"
            report += f"  Composite Score: {score.composite_score:.3f}\n"
            report += f"  Information Gain: {score.information_gain:.3f}\n"
            report += f"  Novelty: {score.novelty:.3f}\n"
            report += f"  Testability: {score.testability:.3f}\n"
            report += f"  Cost-Benefit: {score.cost_benefit:.3f}\n"
            report += f"  Strategic Value: {score.strategic_value:.3f}\n"
            report += f"  Status: {score.test_status}\n"
            report += "\n"

        report += "="*80 + "\n"

        return report
