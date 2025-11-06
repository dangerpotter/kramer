"""
Command-line interface for Kramer.
"""

import argparse
import os
import sys
from pathlib import Path
from kramer.data_analysis_agent import DataAnalysisAgent, AgentConfig


def main():
    """Main CLI entry point."""

    parser = argparse.ArgumentParser(
        description="Kramer - AI-Powered Data Analysis Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  kramer analyze data.csv "Calculate summary statistics"

  # Custom iterations
  kramer analyze data.csv "Find correlations" --max-iterations 10

  # Without extended thinking
  kramer analyze data.csv "Explore the data" --no-extended-thinking

Environment Variables:
  ANTHROPIC_API_KEY    Your Claude API key (required)
        """,
    )

    parser.add_argument(
        "dataset",
        type=str,
        help="Path to dataset (CSV, Excel, etc.)",
    )

    parser.add_argument(
        "objective",
        type=str,
        help="Research objective or question to answer",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-20250514",
        help="Claude model to use (default: claude-sonnet-4-20250514)",
    )

    parser.add_argument(
        "--max-iterations",
        type=int,
        default=5,
        help="Maximum number of analysis steps (default: 5)",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per execution step in seconds (default: 300)",
    )

    parser.add_argument(
        "--no-extended-thinking",
        action="store_true",
        help="Disable extended thinking",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for notebooks and plots (default: outputs)",
    )

    parser.add_argument(
        "--api-key",
        type=str,
        help="Claude API key (or set ANTHROPIC_API_KEY env var)",
    )

    args = parser.parse_args()

    # Check for API key
    api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not set", file=sys.stderr)
        print(
            "Please set it with: export ANTHROPIC_API_KEY='your-key'",
            file=sys.stderr,
        )
        print("Or use --api-key argument", file=sys.stderr)
        sys.exit(1)

    # Check dataset exists
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Error: Dataset not found: {args.dataset}", file=sys.stderr)
        sys.exit(1)

    # Configure agent
    config = AgentConfig(
        api_key=api_key,
        model=args.model,
        max_iterations=args.max_iterations,
        timeout=args.timeout,
        use_extended_thinking=not args.no_extended_thinking,
    )

    # Setup output directories
    output_dir = Path(args.output_dir)
    notebooks_dir = output_dir / "notebooks"
    plots_dir = output_dir / "plots"

    # Initialize agent
    print("🚀 Initializing Kramer Data Analysis Agent...")
    print(f"   Model: {config.model}")
    print(f"   Max iterations: {config.max_iterations}")
    print(f"   Extended thinking: {config.use_extended_thinking}")
    print()

    agent = DataAnalysisAgent(
        config=config,
        notebooks_dir=notebooks_dir,
        plots_dir=plots_dir,
    )

    # Run analysis
    print(f"📊 Analyzing: {dataset_path}")
    print(f"🎯 Objective: {args.objective}")
    print("-" * 60)
    print()

    try:
        result = agent.analyze(
            objective=args.objective,
            dataset_path=str(dataset_path),
        )

        # Display results
        print()
        print("=" * 60)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 60)
        print()

        print(f"Steps completed: {result['steps']}")
        print(f"Success: {result['success']}")
        print(f"Findings: {len(result['findings'])}")
        print()

        print(f"📓 Notebook: {result['notebook_path']}")

        # Show key findings
        if result["findings"]:
            print()
            print("🔍 Key Findings:")
            for i, finding in enumerate(result["findings"][:5], 1):
                print(f"   {i}. [{finding['type'].upper()}] {finding['description']}")

            if len(result["findings"]) > 5:
                print(f"   ... and {len(result['findings']) - 5} more")

        # Save trajectory
        traj_path = output_dir / "trajectory.json"
        traj_path.parent.mkdir(parents=True, exist_ok=True)
        agent.save_trajectory(traj_path)
        print()
        print(f"💾 Trajectory: {traj_path}")

        sys.exit(0 if result["success"] else 1)

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user", file=sys.stderr)
        sys.exit(130)

    except Exception as e:
        print(f"\n\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
