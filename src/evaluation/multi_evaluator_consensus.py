"""
Multi-Evaluator Consensus System for Inter-Rater Reliability.

Provides:
- Collection of evaluations from multiple evaluators
- Inter-rater reliability metrics (Cohen's Kappa, Fleiss' Kappa)
- Consensus resolution strategies
- Disagreement analysis
- Confidence-weighted aggregation
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict, Counter
from enum import Enum
import math

from src.evaluation.evaluation_interface import Verdict, Evaluation

logger = logging.getLogger(__name__)


class ConsensusStrategy(Enum):
    """Strategies for resolving evaluator disagreement."""
    MAJORITY_VOTE = "majority_vote"
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    EXPERT_WEIGHTED = "expert_weighted"
    UNANIMOUS = "unanimous"


@dataclass
class EvaluatorProfile:
    """Profile for an evaluator with expertise and reliability metrics."""
    evaluator_id: str
    name: str
    expertise_level: float = 1.0  # 0-1 scale
    reliability_score: float = 1.0  # 0-1 scale, computed from historical agreement
    evaluations_count: int = 0
    agreement_history: List[float] = field(default_factory=list)


@dataclass
class ConsensusResult:
    """Result of consensus analysis for a claim."""
    claim_id: str
    consensus_verdict: Optional[Verdict] = None
    confidence: float = 0.0
    agreement_level: float = 0.0  # 0-1, percentage of evaluators agreeing
    num_evaluators: int = 0

    # Individual verdicts
    verdicts: Dict[str, Verdict] = field(default_factory=dict)
    evaluator_confidences: Dict[str, float] = field(default_factory=dict)

    # Disagreement info
    has_consensus: bool = False
    disagreement_type: Optional[str] = None  # "minor", "major", "complete"

    # Metadata
    computed_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class InterRaterReliability:
    """Inter-rater reliability metrics."""
    # Cohen's Kappa (for 2 raters)
    cohens_kappa: Optional[float] = None

    # Fleiss' Kappa (for 3+ raters)
    fleiss_kappa: Optional[float] = None

    # Krippendorff's Alpha
    krippendorff_alpha: Optional[float] = None

    # Overall agreement percentage
    overall_agreement: float = 0.0

    # Per-category agreement
    category_agreement: Dict[str, float] = field(default_factory=dict)

    # Interpretation
    interpretation: str = ""


class MultiEvaluatorConsensus:
    """
    Multi-evaluator consensus system with inter-rater reliability.

    Features:
    - Multiple consensus strategies
    - Inter-rater reliability metrics
    - Disagreement analysis
    - Evaluator reliability tracking
    - Confidence-weighted aggregation
    """

    def __init__(
        self,
        consensus_strategy: ConsensusStrategy = ConsensusStrategy.CONFIDENCE_WEIGHTED,
        min_evaluators: int = 2,
        consensus_threshold: float = 0.7
    ):
        """
        Initialize multi-evaluator consensus system.

        Args:
            consensus_strategy: Strategy for resolving disagreements
            min_evaluators: Minimum number of evaluators required
            consensus_threshold: Threshold for consensus (0-1)
        """
        self.consensus_strategy = consensus_strategy
        self.min_evaluators = min_evaluators
        self.consensus_threshold = consensus_threshold

        # Evaluator profiles
        self.evaluators: Dict[str, EvaluatorProfile] = {}

        logger.info(
            f"MultiEvaluatorConsensus initialized: "
            f"strategy={consensus_strategy.value}, "
            f"min_evaluators={min_evaluators}"
        )

    def register_evaluator(
        self,
        evaluator_id: str,
        name: str,
        expertise_level: float = 1.0
    ):
        """
        Register an evaluator.

        Args:
            evaluator_id: Evaluator identifier
            name: Evaluator name
            expertise_level: Expertise level (0-1)
        """
        self.evaluators[evaluator_id] = EvaluatorProfile(
            evaluator_id=evaluator_id,
            name=name,
            expertise_level=expertise_level
        )

        logger.info(f"Registered evaluator: {name} ({evaluator_id})")

    def compute_consensus(
        self,
        claim_id: str,
        evaluations: List[Evaluation]
    ) -> ConsensusResult:
        """
        Compute consensus for a claim from multiple evaluations.

        Args:
            claim_id: Claim identifier
            evaluations: List of evaluations for the claim

        Returns:
            ConsensusResult with aggregated verdict
        """
        if len(evaluations) < self.min_evaluators:
            logger.warning(
                f"Insufficient evaluators for claim {claim_id}: "
                f"{len(evaluations)} < {self.min_evaluators}"
            )
            return ConsensusResult(
                claim_id=claim_id,
                num_evaluators=len(evaluations),
                has_consensus=False
            )

        # Extract verdicts and confidences
        verdicts = {
            eval.evaluator_id: eval.verdict
            for eval in evaluations
        }

        confidences = {
            eval.evaluator_id: eval.confidence_in_verdict
            for eval in evaluations
        }

        # Apply consensus strategy
        if self.consensus_strategy == ConsensusStrategy.MAJORITY_VOTE:
            consensus_verdict, confidence, agreement = self._majority_vote(verdicts)

        elif self.consensus_strategy == ConsensusStrategy.CONFIDENCE_WEIGHTED:
            consensus_verdict, confidence, agreement = self._confidence_weighted(
                verdicts, confidences
            )

        elif self.consensus_strategy == ConsensusStrategy.EXPERT_WEIGHTED:
            consensus_verdict, confidence, agreement = self._expert_weighted(
                verdicts, confidences
            )

        elif self.consensus_strategy == ConsensusStrategy.UNANIMOUS:
            consensus_verdict, confidence, agreement = self._unanimous(verdicts)

        else:
            raise ValueError(f"Unknown consensus strategy: {self.consensus_strategy}")

        # Determine if consensus reached
        has_consensus = agreement >= self.consensus_threshold

        # Analyze disagreement
        disagreement_type = self._analyze_disagreement(verdicts)

        # Create result
        result = ConsensusResult(
            claim_id=claim_id,
            consensus_verdict=consensus_verdict,
            confidence=confidence,
            agreement_level=agreement,
            num_evaluators=len(evaluations),
            verdicts=verdicts,
            evaluator_confidences=confidences,
            has_consensus=has_consensus,
            disagreement_type=disagreement_type
        )

        logger.info(
            f"Consensus for claim {claim_id}: "
            f"verdict={consensus_verdict.value if consensus_verdict else 'None'}, "
            f"agreement={agreement:.2f}, "
            f"consensus={has_consensus}"
        )

        return result

    def compute_inter_rater_reliability(
        self,
        evaluations: List[Evaluation]
    ) -> InterRaterReliability:
        """
        Compute inter-rater reliability metrics.

        Args:
            evaluations: List of all evaluations

        Returns:
            InterRaterReliability with metrics
        """
        # Group evaluations by claim
        by_claim: Dict[str, List[Evaluation]] = defaultdict(list)
        for eval in evaluations:
            by_claim[eval.claim_id].append(eval)

        # Get evaluator IDs
        evaluator_ids = list(set(eval.evaluator_id for eval in evaluations))
        num_evaluators = len(evaluator_ids)

        logger.info(
            f"Computing inter-rater reliability: "
            f"{len(by_claim)} claims, {num_evaluators} evaluators"
        )

        # Compute metrics
        reliability = InterRaterReliability()

        if num_evaluators == 2:
            # Cohen's Kappa for 2 raters
            reliability.cohens_kappa = self._compute_cohens_kappa(by_claim, evaluator_ids)
            reliability.interpretation = self._interpret_kappa(reliability.cohens_kappa)

        elif num_evaluators >= 3:
            # Fleiss' Kappa for 3+ raters
            reliability.fleiss_kappa = self._compute_fleiss_kappa(by_claim, evaluator_ids)
            reliability.interpretation = self._interpret_kappa(reliability.fleiss_kappa)

        # Overall agreement
        reliability.overall_agreement = self._compute_overall_agreement(by_claim)

        # Per-category agreement
        reliability.category_agreement = self._compute_category_agreement(by_claim)

        return reliability

    def _majority_vote(
        self,
        verdicts: Dict[str, Verdict]
    ) -> Tuple[Optional[Verdict], float, float]:
        """Simple majority vote."""
        vote_counts = Counter(verdicts.values())
        most_common = vote_counts.most_common(1)[0]

        majority_verdict = most_common[0]
        majority_count = most_common[1]

        agreement = majority_count / len(verdicts)
        confidence = agreement  # Confidence = agreement percentage

        return majority_verdict, confidence, agreement

    def _confidence_weighted(
        self,
        verdicts: Dict[str, Verdict],
        confidences: Dict[str, float]
    ) -> Tuple[Optional[Verdict], float, float]:
        """Confidence-weighted voting."""
        weighted_votes: Dict[Verdict, float] = defaultdict(float)

        total_confidence = 0.0
        for evaluator_id, verdict in verdicts.items():
            confidence = confidences.get(evaluator_id, 0.5)
            weighted_votes[verdict] += confidence
            total_confidence += confidence

        # Find verdict with highest weighted vote
        consensus_verdict = max(weighted_votes, key=weighted_votes.get)
        consensus_weight = weighted_votes[consensus_verdict]

        # Compute agreement as proportion of weight
        agreement = consensus_weight / total_confidence if total_confidence > 0 else 0.0

        # Compute overall confidence
        confidence = consensus_weight / len(verdicts)

        return consensus_verdict, confidence, agreement

    def _expert_weighted(
        self,
        verdicts: Dict[str, Verdict],
        confidences: Dict[str, float]
    ) -> Tuple[Optional[Verdict], float, float]:
        """Expert and confidence weighted voting."""
        weighted_votes: Dict[Verdict, float] = defaultdict(float)

        total_weight = 0.0
        for evaluator_id, verdict in verdicts.items():
            confidence = confidences.get(evaluator_id, 0.5)

            # Get evaluator expertise
            evaluator = self.evaluators.get(evaluator_id)
            expertise = evaluator.expertise_level if evaluator else 1.0
            reliability = evaluator.reliability_score if evaluator else 1.0

            # Combined weight
            weight = confidence * expertise * reliability

            weighted_votes[verdict] += weight
            total_weight += weight

        # Find verdict with highest weighted vote
        consensus_verdict = max(weighted_votes, key=weighted_votes.get)
        consensus_weight = weighted_votes[consensus_verdict]

        # Compute agreement
        agreement = consensus_weight / total_weight if total_weight > 0 else 0.0

        # Compute confidence
        confidence = consensus_weight / len(verdicts)

        return consensus_verdict, confidence, agreement

    def _unanimous(
        self,
        verdicts: Dict[str, Verdict]
    ) -> Tuple[Optional[Verdict], float, float]:
        """Require unanimous agreement."""
        unique_verdicts = set(verdicts.values())

        if len(unique_verdicts) == 1:
            # Unanimous
            consensus_verdict = list(unique_verdicts)[0]
            return consensus_verdict, 1.0, 1.0
        else:
            # No consensus
            return None, 0.0, 0.0

    def _analyze_disagreement(self, verdicts: Dict[str, Verdict]) -> str:
        """Analyze type of disagreement."""
        unique_verdicts = set(verdicts.values())

        if len(unique_verdicts) == 1:
            return "none"

        # Count votes
        vote_counts = Counter(verdicts.values())
        most_common_count = vote_counts.most_common(1)[0][1]
        total_votes = len(verdicts)

        # Determine disagreement severity
        agreement_ratio = most_common_count / total_votes

        if agreement_ratio >= 0.8:
            return "minor"  # 80%+ agree
        elif agreement_ratio >= 0.5:
            return "moderate"  # 50-80% agree
        else:
            return "major"  # <50% agree (no clear majority)

    def _compute_cohens_kappa(
        self,
        by_claim: Dict[str, List[Evaluation]],
        evaluator_ids: List[str]
    ) -> float:
        """Compute Cohen's Kappa for 2 raters."""
        if len(evaluator_ids) != 2:
            return 0.0

        rater1_id, rater2_id = evaluator_ids

        # Build confusion matrix
        agreements = []
        for claim_id, evals in by_claim.items():
            evals_dict = {e.evaluator_id: e for e in evals}

            if rater1_id not in evals_dict or rater2_id not in evals_dict:
                continue

            verdict1 = evals_dict[rater1_id].verdict
            verdict2 = evals_dict[rater2_id].verdict

            agreements.append((verdict1, verdict2))

        if not agreements:
            return 0.0

        # Compute observed agreement
        po = sum(1 for v1, v2 in agreements if v1 == v2) / len(agreements)

        # Compute expected agreement
        verdict_values = list(Verdict)
        rater1_counts = Counter(v1 for v1, v2 in agreements)
        rater2_counts = Counter(v2 for v1, v2 in agreements)

        pe = sum(
            (rater1_counts[v] / len(agreements)) *
            (rater2_counts[v] / len(agreements))
            for v in verdict_values
        )

        # Cohen's Kappa
        kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0

        return kappa

    def _compute_fleiss_kappa(
        self,
        by_claim: Dict[str, List[Evaluation]],
        evaluator_ids: List[str]
    ) -> float:
        """Compute Fleiss' Kappa for 3+ raters."""
        n_items = len(by_claim)
        n_raters = len(evaluator_ids)
        categories = list(Verdict)

        if n_items == 0 or n_raters < 3:
            return 0.0

        # Build rating matrix
        rating_matrix = []
        for claim_id, evals in by_claim.items():
            evals_dict = {e.evaluator_id: e for e in evals}

            # Count ratings for each category
            row = [
                sum(1 for e in evals if e.verdict == category)
                for category in categories
            ]
            rating_matrix.append(row)

        # Compute P_i (extent of agreement for each item)
        P = []
        for row in rating_matrix:
            sum_squares = sum(n_j ** 2 for n_j in row)
            P_i = (sum_squares - n_raters) / (n_raters * (n_raters - 1))
            P.append(P_i)

        # Compute P_bar (mean agreement)
        P_bar = sum(P) / len(P)

        # Compute P_e (expected agreement)
        p_j = [
            sum(row[j] for row in rating_matrix) / (n_items * n_raters)
            for j in range(len(categories))
        ]
        P_e = sum(p ** 2 for p in p_j)

        # Fleiss' Kappa
        kappa = (P_bar - P_e) / (1 - P_e) if P_e < 1 else 0.0

        return kappa

    def _compute_overall_agreement(
        self,
        by_claim: Dict[str, List[Evaluation]]
    ) -> float:
        """Compute overall percentage agreement."""
        total_claims = 0
        agreement_count = 0

        for claim_id, evals in by_claim.items():
            if len(evals) < 2:
                continue

            total_claims += 1

            # Check if all agree
            verdicts = [e.verdict for e in evals]
            if len(set(verdicts)) == 1:
                agreement_count += 1

        return agreement_count / total_claims if total_claims > 0 else 0.0

    def _compute_category_agreement(
        self,
        by_claim: Dict[str, List[Evaluation]]
    ) -> Dict[str, float]:
        """Compute agreement for each verdict category."""
        category_totals = defaultdict(int)
        category_agreements = defaultdict(int)

        for claim_id, evals in by_claim.items():
            if len(evals) < 2:
                continue

            verdicts = [e.verdict for e in evals]
            most_common = Counter(verdicts).most_common(1)[0][0]

            category_totals[most_common.value] += 1

            # Check agreement
            if len(set(verdicts)) == 1:
                category_agreements[most_common.value] += 1

        # Compute percentages
        return {
            category: (category_agreements[category] / category_totals[category])
            if category_totals[category] > 0 else 0.0
            for category in category_totals
        }

    def _interpret_kappa(self, kappa: Optional[float]) -> str:
        """Interpret Kappa value."""
        if kappa is None:
            return "Not computed"

        if kappa < 0:
            return "Poor (worse than chance)"
        elif kappa < 0.2:
            return "Slight"
        elif kappa < 0.4:
            return "Fair"
        elif kappa < 0.6:
            return "Moderate"
        elif kappa < 0.8:
            return "Substantial"
        else:
            return "Almost perfect"

    def generate_consensus_report(
        self,
        consensus_results: List[ConsensusResult],
        reliability: InterRaterReliability
    ) -> str:
        """
        Generate human-readable consensus report.

        Args:
            consensus_results: List of consensus results
            reliability: Inter-rater reliability metrics

        Returns:
            Formatted report string
        """
        report = "\n" + "="*80 + "\n"
        report += "MULTI-EVALUATOR CONSENSUS REPORT\n"
        report += "="*80 + "\n\n"

        # Overall statistics
        total_claims = len(consensus_results)
        consensus_claims = sum(1 for r in consensus_results if r.has_consensus)
        consensus_rate = consensus_claims / total_claims if total_claims > 0 else 0.0

        report += f"Total Claims Evaluated: {total_claims}\n"
        report += f"Claims with Consensus: {consensus_claims} ({consensus_rate:.1%})\n"
        report += f"Consensus Strategy: {self.consensus_strategy.value}\n"
        report += f"Consensus Threshold: {self.consensus_threshold:.1%}\n\n"

        # Inter-rater reliability
        report += "Inter-Rater Reliability:\n"
        report += "-" * 80 + "\n"

        if reliability.cohens_kappa is not None:
            report += f"  Cohen's Kappa: {reliability.cohens_kappa:.3f} ({reliability.interpretation})\n"

        if reliability.fleiss_kappa is not None:
            report += f"  Fleiss' Kappa: {reliability.fleiss_kappa:.3f} ({reliability.interpretation})\n"

        report += f"  Overall Agreement: {reliability.overall_agreement:.1%}\n"

        if reliability.category_agreement:
            report += "\n  Per-Category Agreement:\n"
            for category, agreement in reliability.category_agreement.items():
                report += f"    {category}: {agreement:.1%}\n"

        report += "\n"

        # Disagreement analysis
        disagreement_types = Counter(r.disagreement_type for r in consensus_results)

        report += "Disagreement Analysis:\n"
        report += "-" * 80 + "\n"
        for disagreement_type, count in disagreement_types.most_common():
            report += f"  {disagreement_type}: {count} claims\n"

        report += "\n"

        # Sample consensus results
        report += "Sample Consensus Results:\n"
        report += "-" * 80 + "\n\n"

        for i, result in enumerate(consensus_results[:5], 1):
            report += f"{i}. Claim: {result.claim_id}\n"
            report += f"   Consensus: {result.consensus_verdict.value if result.consensus_verdict else 'None'}\n"
            report += f"   Agreement: {result.agreement_level:.1%}\n"
            report += f"   Confidence: {result.confidence:.3f}\n"
            report += f"   Evaluators: {result.num_evaluators}\n"
            report += f"   Disagreement: {result.disagreement_type}\n\n"

        report += "="*80 + "\n"

        return report
