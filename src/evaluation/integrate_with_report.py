"""
Integration Module - Integrate evaluation system with report generation.

This module provides utilities to automatically run evaluation pipeline
after report generation is complete.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.evaluation.claim_extractor import ClaimExtractor
from src.evaluation.evaluation_interface import EvaluationInterface
from src.evaluation.metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Post-processing pipeline for expert evaluation.

    Can be integrated with ReportGenerator to automatically extract claims
    after report generation.
    """

    def __init__(
        self,
        evaluation_dir: Path = Path("outputs/evaluation"),
        auto_run_interactive: bool = False,
    ):
        """
        Initialize the evaluation pipeline.

        Args:
            evaluation_dir: Directory for evaluation outputs
            auto_run_interactive: Whether to automatically run interactive evaluation
        """
        self.evaluation_dir = evaluation_dir
        self.auto_run_interactive = auto_run_interactive

        self.evaluation_dir.mkdir(parents=True, exist_ok=True)

        # Paths
        self.claims_path = self.evaluation_dir / "claims.json"
        self.db_path = self.evaluation_dir / "evaluations.db"
        self.metrics_path = self.evaluation_dir / "accuracy_report.md"

    def process_report(
        self,
        report_path: Path,
        evaluator_id: str = "default"
    ) -> Dict[str, Any]:
        """
        Process a generated report through the evaluation pipeline.

        Steps:
        1. Extract testable claims
        2. Store claims in database
        3. (Optional) Run interactive evaluation
        4. Generate accuracy report

        Args:
            report_path: Path to the generated markdown report
            evaluator_id: Identifier for the evaluator

        Returns:
            Dictionary with pipeline results
        """
        logger.info(f"Processing report for evaluation: {report_path}")

        results = {
            "report_path": str(report_path),
            "claims_extracted": 0,
            "claims_evaluated": 0,
            "accuracy": None,
        }

        # Step 1: Extract claims
        logger.info("Extracting claims from report...")
        extractor = ClaimExtractor()
        claims = extractor.extract_claims(report_path)
        extractor.save_claims(claims, self.claims_path)

        results["claims_extracted"] = len(claims)
        results["claims_path"] = str(self.claims_path)

        logger.info(f"Extracted {len(claims)} claims")

        # Step 2: Store claims in database
        logger.info("Storing claims in evaluation database...")
        interface = EvaluationInterface(self.db_path)
        interface.store_claims(claims)

        results["database_path"] = str(self.db_path)

        # Step 3: Interactive evaluation (if enabled)
        if self.auto_run_interactive:
            logger.info("Running interactive evaluation session...")

            unevaluated = interface.get_unevaluated_claims(evaluator_id)

            if unevaluated:
                evaluations = interface.interactive_evaluation_session(
                    unevaluated,
                    evaluator_id,
                    auto_save=True
                )
                results["claims_evaluated"] = len(evaluations)
            else:
                logger.info("No unevaluated claims found")
        else:
            logger.info("Skipping interactive evaluation (auto_run_interactive=False)")
            logger.info(f"To evaluate claims, run: python -m src.evaluation.run_evaluation evaluate {self.claims_path}")

        # Step 4: Generate accuracy report (if there are evaluations)
        tracker = MetricsTracker(self.db_path)
        overall_metrics = tracker.compute_accuracy()

        if overall_metrics.evaluated_claims > 0:
            logger.info("Generating accuracy report...")
            report_result = tracker.generate_report(self.metrics_path)

            results["accuracy"] = overall_metrics.accuracy
            results["metrics_path"] = str(self.metrics_path)
            results["accuracy_report"] = report_result["overall_metrics"]

            logger.info(f"Overall accuracy: {overall_metrics.accuracy:.1%}")
        else:
            logger.info("No evaluations available, skipping accuracy report")

        logger.info("Evaluation pipeline complete")
        return results


def integrate_with_orchestrator(
    report_path: Path,
    evaluation_dir: Optional[Path] = None,
    run_interactive: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to integrate evaluation with the orchestrator.

    This can be called after report generation in the orchestrator/main loop.

    Args:
        report_path: Path to generated report
        evaluation_dir: Directory for evaluation outputs (default: outputs/evaluation)
        run_interactive: Whether to run interactive evaluation session

    Returns:
        Dictionary with pipeline results

    Example:
        >>> from src.evaluation.integrate_with_report import integrate_with_orchestrator
        >>> results = integrate_with_orchestrator(
        ...     report_path=Path("outputs/report.md"),
        ...     run_interactive=False  # Don't block on interactive session
        ... )
        >>> print(f"Extracted {results['claims_extracted']} claims")
    """
    if evaluation_dir is None:
        evaluation_dir = Path("outputs/evaluation")

    pipeline = EvaluationPipeline(
        evaluation_dir=evaluation_dir,
        auto_run_interactive=run_interactive,
    )

    return pipeline.process_report(report_path)


# Example integration with ReportGenerator
def post_report_hook(report_result: Dict[str, Any]) -> None:
    """
    Hook to run after ReportGenerator.generate_report() completes.

    This can be called from the orchestrator after report generation.

    Args:
        report_result: Result dictionary from ReportGenerator.generate_report()

    Example in orchestrator:
        >>> from src.reporting.report_generator import ReportGenerator
        >>> from src.evaluation.integrate_with_report import post_report_hook
        >>>
        >>> generator = ReportGenerator(world_model)
        >>> report_result = generator.generate_report(output_path)
        >>>
        >>> # Run evaluation pipeline
        >>> post_report_hook(report_result)
    """
    report_path = report_result.get("report")

    if not report_path:
        logger.warning("No report path in report_result, skipping evaluation")
        return

    logger.info("Running post-report evaluation pipeline...")

    # Run non-interactive pipeline by default
    # Users can manually run interactive evaluation later
    integrate_with_orchestrator(
        report_path=Path(report_path),
        run_interactive=False,
    )
