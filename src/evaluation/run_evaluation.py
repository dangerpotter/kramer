#!/usr/bin/env python3
"""
Run Evaluation - CLI tool for running expert evaluations.

This script provides a command-line interface for:
1. Extracting claims from reports
2. Running interactive evaluation sessions
3. Generating accuracy reports
"""

import argparse
import logging
from pathlib import Path

from src.evaluation.claim_extractor import ClaimExtractor
from src.evaluation.evaluation_interface import EvaluationInterface
from src.evaluation.metrics_tracker import MetricsTracker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_claims(report_path: Path, output_path: Path) -> None:
    """Extract claims from a report and save to JSON."""
    logger.info(f"Extracting claims from {report_path}")

    extractor = ClaimExtractor()
    claims = extractor.extract_claims(report_path)

    extractor.save_claims(claims, output_path)

    logger.info(f"Extracted {len(claims)} claims")
    logger.info(f"Claims saved to {output_path}")


def run_evaluation_session(
    claims_path: Path,
    db_path: Path,
    evaluator_id: str = "default"
) -> None:
    """Run an interactive evaluation session."""
    logger.info("Starting interactive evaluation session")

    # Initialize interface
    interface = EvaluationInterface(db_path)

    # Load claims
    extractor = ClaimExtractor()
    claims = extractor.load_claims(claims_path)

    # Store claims in database
    interface.store_claims(claims)

    # Get unevaluated claims
    unevaluated = interface.get_unevaluated_claims(evaluator_id)

    if not unevaluated:
        logger.info("No unevaluated claims found!")
        return

    logger.info(f"Found {len(unevaluated)} unevaluated claims")

    # Run interactive session
    evaluations = interface.interactive_evaluation_session(
        unevaluated,
        evaluator_id,
        auto_save=True
    )

    logger.info(f"Session complete. Evaluated {len(evaluations)} claims")


def generate_accuracy_report(db_path: Path, output_path: Path) -> None:
    """Generate accuracy report from evaluations."""
    logger.info("Generating accuracy report")

    tracker = MetricsTracker(db_path)
    result = tracker.generate_report(output_path)

    logger.info(f"Report generated: {output_path}")
    logger.info(f"Overall accuracy: {result['overall_metrics']['accuracy']:.1%}")


def full_pipeline(
    report_path: Path,
    output_dir: Path,
    evaluator_id: str = "default",
    interactive: bool = True
) -> None:
    """Run the full evaluation pipeline."""
    logger.info("Running full evaluation pipeline")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    claims_path = output_dir / "claims.json"
    db_path = output_dir / "evaluations.db"
    metrics_path = output_dir / "accuracy_report.md"

    # Step 1: Extract claims
    logger.info("Step 1/3: Extracting claims")
    extract_claims(report_path, claims_path)

    # Step 2: Run evaluation (if interactive)
    if interactive:
        logger.info("Step 2/3: Running evaluation session")
        run_evaluation_session(claims_path, db_path, evaluator_id)
    else:
        logger.info("Step 2/3: Skipping interactive evaluation (non-interactive mode)")
        # Just store claims in database
        interface = EvaluationInterface(db_path)
        extractor = ClaimExtractor()
        claims = extractor.load_claims(claims_path)
        interface.store_claims(claims)

    # Step 3: Generate report
    logger.info("Step 3/3: Generating accuracy report")
    generate_accuracy_report(db_path, metrics_path)

    logger.info("Pipeline complete!")
    logger.info(f"Outputs saved to: {output_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Expert Evaluation System - Post-hoc evaluation of research claims"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        help="Extract claims from a report"
    )
    extract_parser.add_argument(
        "report",
        type=Path,
        help="Path to markdown report file"
    )
    extract_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("outputs/evaluation/claims.json"),
        help="Output path for claims JSON file"
    )

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Run interactive evaluation session"
    )
    evaluate_parser.add_argument(
        "claims",
        type=Path,
        help="Path to claims JSON file"
    )
    evaluate_parser.add_argument(
        "-d", "--database",
        type=Path,
        default=Path("outputs/evaluation/evaluations.db"),
        help="Path to evaluations database"
    )
    evaluate_parser.add_argument(
        "-e", "--evaluator",
        type=str,
        default="default",
        help="Evaluator ID"
    )

    # Report command
    report_parser = subparsers.add_parser(
        "report",
        help="Generate accuracy report"
    )
    report_parser.add_argument(
        "-d", "--database",
        type=Path,
        default=Path("outputs/evaluation/evaluations.db"),
        help="Path to evaluations database"
    )
    report_parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("outputs/evaluation/accuracy_report.md"),
        help="Output path for accuracy report"
    )

    # Full pipeline command
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run full evaluation pipeline"
    )
    pipeline_parser.add_argument(
        "report",
        type=Path,
        help="Path to markdown report file"
    )
    pipeline_parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path("outputs/evaluation"),
        help="Output directory for all files"
    )
    pipeline_parser.add_argument(
        "-e", "--evaluator",
        type=str,
        default="default",
        help="Evaluator ID"
    )
    pipeline_parser.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip interactive evaluation session"
    )

    args = parser.parse_args()

    if args.command == "extract":
        extract_claims(args.report, args.output)
    elif args.command == "evaluate":
        run_evaluation_session(args.claims, args.database, args.evaluator)
    elif args.command == "report":
        generate_accuracy_report(args.database, args.output)
    elif args.command == "pipeline":
        full_pipeline(
            args.report,
            args.output_dir,
            args.evaluator,
            interactive=not args.non_interactive
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
