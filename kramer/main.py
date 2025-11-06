"""Main entry point for Kramer discovery loop"""

import argparse
import json
import logging
from pathlib import Path

from .config import Config, setup_logging, create_output_directory
from .world_model import WorldModel
from .data_agent import DataAgent
from .literature_agent import LiteratureAgent
from .cycle_manager import CycleManager

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Kramer - Autonomous Research Discovery Loop"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset (CSV file)"
    )

    parser.add_argument(
        "--max-cycles",
        type=int,
        default=20,
        help="Maximum number of cycles to run (default: 20)"
    )

    parser.add_argument(
        "--max-time",
        type=float,
        default=6.0,
        help="Maximum time in hours (default: 6.0)"
    )

    parser.add_argument(
        "--stagnation-cycles",
        type=int,
        default=3,
        help="Stop if no new findings in N cycles (default: 3)"
    )

    parser.add_argument(
        "--tasks-per-cycle",
        type=int,
        default=10,
        help="Number of tasks per cycle (default: 10)"
    )

    parser.add_argument(
        "--max-parallel",
        type=int,
        default=4,
        help="Maximum parallel tasks (default: 4)"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    return parser.parse_args()


def print_summary(summary: dict):
    """Print a formatted summary"""
    print("\n" + "="*80)
    print("DISCOVERY LOOP SUMMARY")
    print("="*80)

    print(f"\nTotal Cycles: {summary['total_cycles']}")
    print(f"Elapsed Time: {summary['elapsed_time']}")

    stats = summary['statistics']
    print(f"\nStatistics:")
    print(f"  - Hypotheses: {stats['total_hypotheses']} (Untested: {stats['untested_hypotheses']})")
    print(f"  - Findings: {stats['total_findings']}")
    print(f"  - Papers: {stats['total_papers']}")
    print(f"  - Tasks: {stats['total_tasks']}")

    if summary['supported_hypotheses']:
        print(f"\nSupported Hypotheses ({len(summary['supported_hypotheses'])}):")
        for i, hyp in enumerate(summary['supported_hypotheses'][:5], 1):
            print(f"  {i}. {hyp}")

    if summary['key_findings']:
        print(f"\nKey Findings ({len(summary['key_findings'])}):")
        for i, finding in enumerate(summary['key_findings'][:10], 1):
            print(f"  {i}. {finding}")

    if summary['top_papers']:
        print(f"\nTop Papers ({len(summary['top_papers'])}):")
        for i, paper in enumerate(summary['top_papers'], 1):
            print(f"  {i}. {paper['title']}")
            print(f"     Authors: {paper['authors']} (Relevance: {paper['relevance']:.2f})")

    print("\n" + "="*80)


def main():
    """Main execution function"""
    args = parse_args()

    # Create config
    config = Config(
        max_cycles=args.max_cycles,
        max_time_hours=args.max_time,
        stagnation_cycles=args.stagnation_cycles,
        tasks_per_cycle=args.tasks_per_cycle,
        max_parallel_tasks=args.max_parallel,
        output_dir=args.output_dir,
        log_level=args.log_level
    )

    # Setup logging
    setup_logging(config)
    logger.info("Starting Kramer Discovery Loop")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Config: max_cycles={config.max_cycles}, max_time={config.max_time_hours}h")

    # Create output directory
    output_path = create_output_directory(config)

    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        return 1

    try:
        # Initialize components
        logger.info("Initializing components...")
        world_model = WorldModel()
        data_agent = DataAgent(str(dataset_path))
        literature_agent = LiteratureAgent()

        # Create cycle manager
        cycle_manager = CycleManager(
            world_model=world_model,
            data_agent=data_agent,
            literature_agent=literature_agent,
            max_cycles=config.max_cycles,
            max_time_hours=config.max_time_hours,
            stagnation_cycles=config.stagnation_cycles,
            tasks_per_cycle=config.tasks_per_cycle,
            max_parallel_tasks=config.max_parallel_tasks
        )

        # Run discovery loop
        logger.info("Starting discovery loop...")
        summary = cycle_manager.run()

        # Save results
        logger.info("Saving results...")

        # Save world model
        world_model_path = output_path / "world_model.json"
        world_model.save_to_file(str(world_model_path))
        logger.info(f"World model saved to: {world_model_path}")

        # Save summary
        summary_path = output_path / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {summary_path}")

        # Print summary to console
        print_summary(summary)

        logger.info("Discovery loop completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error during execution: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
