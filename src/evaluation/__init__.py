"""
Expert Evaluation System - Post-hoc expert evaluation as described in Kosmos paper.

This package provides tools for extracting testable claims from research reports,
collecting expert evaluations, and tracking accuracy metrics over time.

Main components:
- ClaimExtractor: Extracts and categorizes testable claims from reports
- EvaluationInterface: Interactive interface for expert review
- MetricsTracker: Computes accuracy metrics and trends
"""

from src.evaluation.claim_extractor import Claim, ClaimExtractor, ClaimType
from src.evaluation.evaluation_interface import (
    Evaluation,
    EvaluationInterface,
    Verdict,
)
from src.evaluation.metrics_tracker import AccuracyMetrics, MetricsTracker

__all__ = [
    "Claim",
    "ClaimExtractor",
    "ClaimType",
    "Evaluation",
    "EvaluationInterface",
    "Verdict",
    "AccuracyMetrics",
    "MetricsTracker",
]
