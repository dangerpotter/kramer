"""
Metrics Tracker - Track evaluation accuracy metrics over time.

This module computes accuracy metrics from expert evaluations, tracks trends,
and generates reports on system performance.
"""

import json
import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.evaluation.claim_extractor import ClaimType
from src.evaluation.evaluation_interface import Evaluation, Verdict

logger = logging.getLogger(__name__)


@dataclass
class AccuracyMetrics:
    """Accuracy metrics for a set of evaluations."""
    total_claims: int = 0
    evaluated_claims: int = 0
    supported_claims: int = 0
    refuted_claims: int = 0
    unclear_claims: int = 0
    partially_supported_claims: int = 0

    # Accuracy rates
    support_rate: float = 0.0  # % of claims that were supported
    refute_rate: float = 0.0  # % of claims that were refuted
    unclear_rate: float = 0.0  # % of claims that were unclear
    partial_support_rate: float = 0.0  # % of claims that were partially supported

    # Accuracy score (treats supported/partially_supported as correct)
    accuracy: float = 0.0

    # By claim type
    by_type: Dict[str, Dict[str, Any]] = None

    def __post_init__(self):
        """Calculate derived metrics."""
        if self.by_type is None:
            self.by_type = {}

        if self.evaluated_claims > 0:
            self.support_rate = self.supported_claims / self.evaluated_claims
            self.refute_rate = self.refuted_claims / self.evaluated_claims
            self.unclear_rate = self.unclear_claims / self.evaluated_claims
            self.partial_support_rate = self.partially_supported_claims / self.evaluated_claims

            # Calculate accuracy (supported + partially supported)
            self.accuracy = (self.supported_claims + self.partially_supported_claims) / self.evaluated_claims

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_claims": self.total_claims,
            "evaluated_claims": self.evaluated_claims,
            "supported_claims": self.supported_claims,
            "refuted_claims": self.refuted_claims,
            "unclear_claims": self.unclear_claims,
            "partially_supported_claims": self.partially_supported_claims,
            "support_rate": self.support_rate,
            "refute_rate": self.refute_rate,
            "unclear_rate": self.unclear_rate,
            "partial_support_rate": self.partial_support_rate,
            "accuracy": self.accuracy,
            "by_type": self.by_type,
        }


class MetricsTracker:
    """
    Tracks evaluation accuracy metrics over time.

    Computes:
    1. Overall accuracy (% of claims supported/partially supported)
    2. Breakdown by claim type (data analysis, literature, interpretation)
    3. Trends over time
    4. Confidence calibration (system confidence vs expert verdicts)
    """

    def __init__(self, db_path: Path):
        """
        Initialize the metrics tracker.

        Args:
            db_path: Path to SQLite database with evaluations
        """
        self.db_path = db_path

    def compute_accuracy(
        self,
        evaluations: Optional[List[Evaluation]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> AccuracyMetrics:
        """
        Compute accuracy metrics for evaluations.

        Args:
            evaluations: List of evaluations (if None, loads all from database)
            start_date: Filter evaluations after this date
            end_date: Filter evaluations before this date

        Returns:
            AccuracyMetrics object with computed metrics
        """
        # Load evaluations if not provided
        if evaluations is None:
            evaluations = self._load_evaluations(start_date, end_date)

        # Load associated claims
        claim_map = self._load_claims([e.claim_id for e in evaluations])

        # Count verdicts
        verdict_counts = {
            Verdict.SUPPORTED: 0,
            Verdict.REFUTED: 0,
            Verdict.UNCLEAR: 0,
            Verdict.PARTIALLY_SUPPORTED: 0,
        }

        # Count by claim type
        by_type = defaultdict(lambda: {
            "total": 0,
            "supported": 0,
            "refuted": 0,
            "unclear": 0,
            "partially_supported": 0,
        })

        for evaluation in evaluations:
            if evaluation.verdict:
                verdict_counts[evaluation.verdict] += 1

                # Get claim type
                claim = claim_map.get(evaluation.claim_id)
                if claim:
                    claim_type = claim["claim_type"]
                    by_type[claim_type]["total"] += 1

                    if evaluation.verdict == Verdict.SUPPORTED:
                        by_type[claim_type]["supported"] += 1
                    elif evaluation.verdict == Verdict.REFUTED:
                        by_type[claim_type]["refuted"] += 1
                    elif evaluation.verdict == Verdict.UNCLEAR:
                        by_type[claim_type]["unclear"] += 1
                    elif evaluation.verdict == Verdict.PARTIALLY_SUPPORTED:
                        by_type[claim_type]["partially_supported"] += 1

        # Calculate accuracy for each type
        by_type_metrics = {}
        for claim_type, counts in by_type.items():
            total = counts["total"]
            if total > 0:
                by_type_metrics[claim_type] = {
                    "total": total,
                    "supported": counts["supported"],
                    "refuted": counts["refuted"],
                    "unclear": counts["unclear"],
                    "partially_supported": counts["partially_supported"],
                    "support_rate": counts["supported"] / total,
                    "refute_rate": counts["refuted"] / total,
                    "unclear_rate": counts["unclear"] / total,
                    "partial_support_rate": counts["partially_supported"] / total,
                    "accuracy": (counts["supported"] + counts["partially_supported"]) / total,
                }

        # Create metrics object
        metrics = AccuracyMetrics(
            total_claims=len(claim_map),
            evaluated_claims=len(evaluations),
            supported_claims=verdict_counts[Verdict.SUPPORTED],
            refuted_claims=verdict_counts[Verdict.REFUTED],
            unclear_claims=verdict_counts[Verdict.UNCLEAR],
            partially_supported_claims=verdict_counts[Verdict.PARTIALLY_SUPPORTED],
            by_type=by_type_metrics,
        )

        return metrics

    def compute_trends(
        self,
        time_window_days: int = 30,
        num_windows: int = 5
    ) -> List[Tuple[datetime, datetime, AccuracyMetrics]]:
        """
        Compute accuracy trends over time.

        Args:
            time_window_days: Size of each time window in days
            num_windows: Number of time windows to compute

        Returns:
            List of (start_date, end_date, metrics) tuples
        """
        trends = []

        # Get date range
        end_date = datetime.now()

        for i in range(num_windows):
            window_end = end_date - timedelta(days=i * time_window_days)
            window_start = window_end - timedelta(days=time_window_days)

            # Compute metrics for this window
            metrics = self.compute_accuracy(start_date=window_start, end_date=window_end)

            # Only include if there are evaluations
            if metrics.evaluated_claims > 0:
                trends.insert(0, (window_start, window_end, metrics))

        logger.info(f"Computed trends for {len(trends)} time windows")
        return trends

    def compute_confidence_calibration(self) -> Dict[str, Any]:
        """
        Compute confidence calibration metrics.

        Analyzes how well the system's confidence scores correlate with
        expert verdicts.

        Returns:
            Dictionary with calibration metrics
        """
        # Load evaluations and claims
        evaluations = self._load_evaluations()
        claim_map = self._load_claims([e.claim_id for e in evaluations])

        # Group by confidence bins
        bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        binned_data = {f"{bins[i]:.1f}-{bins[i+1]:.1f}": []
                      for i in range(len(bins) - 1)}

        for evaluation in evaluations:
            claim = claim_map.get(evaluation.claim_id)
            if not claim or claim.get("confidence") is None:
                continue

            confidence = claim["confidence"]
            is_accurate = evaluation.verdict in [Verdict.SUPPORTED, Verdict.PARTIALLY_SUPPORTED]

            # Find bin
            for i in range(len(bins) - 1):
                if bins[i] <= confidence < bins[i + 1]:
                    bin_key = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                    binned_data[bin_key].append(is_accurate)
                    break

        # Compute accuracy for each bin
        calibration = {}
        for bin_key, accuracies in binned_data.items():
            if accuracies:
                calibration[bin_key] = {
                    "count": len(accuracies),
                    "accuracy": sum(accuracies) / len(accuracies),
                    "expected_confidence": (float(bin_key.split('-')[0]) + float(bin_key.split('-')[1])) / 2,
                }

        return calibration

    def generate_report(self, output_path: Path) -> Dict[str, Any]:
        """
        Generate a comprehensive accuracy report.

        Args:
            output_path: Path to output markdown file

        Returns:
            Dictionary with report metadata
        """
        logger.info("Generating accuracy report...")

        # Compute overall metrics
        overall_metrics = self.compute_accuracy()

        # Compute trends
        trends = self.compute_trends(time_window_days=7, num_windows=4)

        # Compute calibration
        calibration = self.compute_confidence_calibration()

        # Write markdown report
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write("# Expert Evaluation Accuracy Report\n\n")
            f.write(f"*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            f.write(f"- **Total Claims**: {overall_metrics.total_claims}\n")
            f.write(f"- **Evaluated Claims**: {overall_metrics.evaluated_claims}\n")
            f.write(f"- **Overall Accuracy**: {overall_metrics.accuracy:.1%}\n")
            f.write(f"- **Support Rate**: {overall_metrics.support_rate:.1%}\n")
            f.write(f"- **Refute Rate**: {overall_metrics.refute_rate:.1%}\n")
            f.write(f"- **Unclear Rate**: {overall_metrics.unclear_rate:.1%}\n\n")

            # Verdict Distribution
            f.write("## Verdict Distribution\n\n")
            f.write(f"- Supported: {overall_metrics.supported_claims} ({overall_metrics.support_rate:.1%})\n")
            f.write(f"- Partially Supported: {overall_metrics.partially_supported_claims} ({overall_metrics.partial_support_rate:.1%})\n")
            f.write(f"- Refuted: {overall_metrics.refuted_claims} ({overall_metrics.refute_rate:.1%})\n")
            f.write(f"- Unclear: {overall_metrics.unclear_claims} ({overall_metrics.unclear_rate:.1%})\n\n")

            # By Claim Type
            if overall_metrics.by_type:
                f.write("## Accuracy by Claim Type\n\n")
                f.write("| Claim Type | Total | Accuracy | Support Rate | Refute Rate | Unclear Rate |\n")
                f.write("|------------|-------|----------|--------------|-------------|-------------|\n")

                for claim_type, metrics in overall_metrics.by_type.items():
                    f.write(f"| {claim_type} | {metrics['total']} | "
                           f"{metrics['accuracy']:.1%} | {metrics['support_rate']:.1%} | "
                           f"{metrics['refute_rate']:.1%} | {metrics['unclear_rate']:.1%} |\n")
                f.write("\n")

            # Trends
            if trends:
                f.write("## Accuracy Trends\n\n")
                f.write("| Time Period | Evaluated | Accuracy | Support Rate |\n")
                f.write("|-------------|-----------|----------|-------------|\n")

                for start, end, metrics in trends:
                    period = f"{start.strftime('%Y-%m-%d')} to {end.strftime('%Y-%m-%d')}"
                    f.write(f"| {period} | {metrics.evaluated_claims} | "
                           f"{metrics.accuracy:.1%} | {metrics.support_rate:.1%} |\n")
                f.write("\n")

            # Confidence Calibration
            if calibration:
                f.write("## Confidence Calibration\n\n")
                f.write("Comparison of system confidence scores vs. actual accuracy:\n\n")
                f.write("| Confidence Range | Count | Actual Accuracy | Expected Confidence |\n")
                f.write("|------------------|-------|-----------------|--------------------|\n")

                for bin_key in sorted(calibration.keys()):
                    metrics = calibration[bin_key]
                    f.write(f"| {bin_key} | {metrics['count']} | "
                           f"{metrics['accuracy']:.1%} | {metrics['expected_confidence']:.1%} |\n")
                f.write("\n")

                # Calibration analysis
                f.write("### Calibration Analysis\n\n")
                well_calibrated = []
                over_confident = []
                under_confident = []

                for bin_key, metrics in calibration.items():
                    diff = metrics['accuracy'] - metrics['expected_confidence']
                    if abs(diff) < 0.1:
                        well_calibrated.append(bin_key)
                    elif diff < -0.1:
                        over_confident.append(bin_key)
                    else:
                        under_confident.append(bin_key)

                if well_calibrated:
                    f.write(f"**Well-calibrated ranges**: {', '.join(well_calibrated)}\n\n")
                if over_confident:
                    f.write(f"**Over-confident ranges** (claims less accurate than expected): {', '.join(over_confident)}\n\n")
                if under_confident:
                    f.write(f"**Under-confident ranges** (claims more accurate than expected): {', '.join(under_confident)}\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            if overall_metrics.accuracy < 0.7:
                f.write("- **Low accuracy detected**: Consider reviewing claim extraction and categorization logic.\n")

            if overall_metrics.refute_rate > 0.3:
                f.write("- **High refute rate**: Significant portion of claims are being refuted by experts. "
                       "Review data analysis methods and confidence thresholds.\n")

            if overall_metrics.unclear_rate > 0.2:
                f.write("- **High unclear rate**: Many claims lack sufficient evidence. "
                       "Consider improving context provision and claim specificity.\n")

            # Check for over-confidence
            if calibration:
                avg_overconfidence = sum(
                    max(0, m['expected_confidence'] - m['accuracy'])
                    for m in calibration.values()
                ) / len(calibration)

                if avg_overconfidence > 0.1:
                    f.write("- **Over-confidence detected**: System confidence scores are higher than actual accuracy. "
                           "Consider recalibrating confidence thresholds.\n")

            f.write("\n")

        logger.info(f"Generated accuracy report: {output_path}")

        return {
            "report_path": output_path,
            "overall_metrics": overall_metrics.to_dict(),
            "trends": [
                {
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "metrics": metrics.to_dict()
                }
                for start, end, metrics in trends
            ],
            "calibration": calibration,
        }

    def _load_evaluations(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Evaluation]:
        """Load evaluations from database with optional date filtering."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        query = "SELECT * FROM evaluations WHERE verdict IS NOT NULL"
        params = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())

        if end_date:
            query += " AND timestamp < ?"
            params.append(end_date.isoformat())

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        evaluations = []
        for row in rows:
            evaluation = Evaluation(
                evaluation_id=row[0],
                claim_id=row[1],
                verdict=Verdict(row[2]) if row[2] else None,
                evaluator_id=row[3],
                notes=row[4] or "",
                confidence_in_verdict=row[5],
                timestamp=datetime.fromisoformat(row[6]) if row[6] else None,
            )
            evaluations.append(evaluation)

        return evaluations

    def _load_claims(self, claim_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """Load claims from database."""
        if not claim_ids:
            return {}

        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()

        # Build query with placeholders
        placeholders = ','.join(['?' for _ in claim_ids])
        query = f"SELECT * FROM claims WHERE claim_id IN ({placeholders})"

        cursor.execute(query, claim_ids)
        rows = cursor.fetchall()
        conn.close()

        claim_map = {}
        for row in rows:
            claim_map[row[0]] = {
                "claim_id": row[0],
                "text": row[1],
                "claim_type": row[2],
                "discovery_title": row[3],
                "confidence": row[4],
                "context": row[5],
                "source_section": row[6],
            }

        return claim_map

    def export_metrics(self, output_path: Path) -> None:
        """
        Export all metrics to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        overall_metrics = self.compute_accuracy()
        trends = self.compute_trends()
        calibration = self.compute_confidence_calibration()

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "export_timestamp": datetime.now().isoformat(),
                "overall_metrics": overall_metrics.to_dict(),
                "trends": [
                    {
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "metrics": metrics.to_dict()
                    }
                    for start, end, metrics in trends
                ],
                "calibration": calibration,
            }, f, indent=2)

        logger.info(f"Exported metrics to {output_path}")
