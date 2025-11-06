#!/usr/bin/env python3
"""Example: Run discovery loop on Iris dataset"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kramer.config import Config, setup_logging, create_output_directory
from kramer.world_model import WorldModel
from kramer.data_agent import DataAgent
from kramer.literature_agent import LiteratureAgent
from kramer.cycle_manager import CycleManager


def main():
    """Run discovery loop on Iris dataset"""

    # Configuration
    config = Config(
        max_cycles=5,  # Run 5 cycles for quick demo
        max_time_hours=1.0,
        stagnation_cycles=3,
        tasks_per_cycle=6,
        max_parallel_tasks=3,
        log_level="INFO",
        output_dir="output"
    )

    # Setup
    setup_logging(config)
    output_path = create_output_directory(config)

    # Dataset path
    dataset_path = Path(__file__).parent.parent / "data" / "iris.csv"

    print("=" * 80)
    print("Kramer - Autonomous Research Discovery Loop")
    print("=" * 80)
    print(f"\nDataset: {dataset_path}")
    print(f"Max Cycles: {config.max_cycles}")
    print(f"Tasks per Cycle: {config.tasks_per_cycle}")
    print(f"Output: {output_path}")
    print("\n" + "=" * 80 + "\n")

    # Initialize components
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
    print("Starting discovery loop...\n")
    summary = cycle_manager.run()

    # Print summary
    print("\n" + "=" * 80)
    print("DISCOVERY LOOP COMPLETE")
    print("=" * 80)
    print(f"\nTotal Cycles: {summary['total_cycles']}")
    print(f"Elapsed Time: {summary['elapsed_time']}")

    stats = summary['statistics']
    print(f"\nStatistics:")
    print(f"  - Hypotheses: {stats['total_hypotheses']} (Untested: {stats['untested_hypotheses']})")
    print(f"  - Findings: {stats['total_findings']}")
    print(f"  - Papers: {stats['total_papers']}")

    if summary['key_findings']:
        print(f"\nKey Findings:")
        for i, finding in enumerate(summary['key_findings'][:5], 1):
            print(f"  {i}. {finding}")

    # Save results
    world_model.save_to_file(str(output_path / "world_model.json"))
    print(f"\nResults saved to: {output_path}/")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
