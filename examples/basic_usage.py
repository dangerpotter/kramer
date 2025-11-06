"""
Basic usage example for the Kramer Data Analysis Agent.

This script demonstrates how to:
1. Initialize the agent
2. Run an analysis on a dataset
3. Access the results and generated notebook
"""

import os
from pathlib import Path
from kramer.data_analysis_agent import DataAnalysisAgent, AgentConfig


def main():
    """Run a basic data analysis."""

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it with: export ANTHROPIC_API_KEY='your-key-here'")
        return

    # Configure the agent
    config = AgentConfig(
        model="claude-sonnet-4-20250514",
        max_iterations=5,  # Number of analysis steps
        use_extended_thinking=True,  # Use extended thinking for better analysis
        timeout=300,  # Max execution time per step (seconds)
    )

    # Initialize the agent
    print("Initializing Kramer Data Analysis Agent...")
    agent = DataAnalysisAgent(
        config=config,
        notebooks_dir=Path("outputs/notebooks"),
        plots_dir=Path("outputs/plots"),
    )

    # Path to the sample dataset
    dataset_path = "data/sample_data.csv"

    if not Path(dataset_path).exists():
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please ensure the sample_data.csv exists in the data/ directory")
        return

    # Run the analysis
    print(f"\nAnalyzing dataset: {dataset_path}")
    print("Objective: Analyze customer satisfaction drivers")
    print("-" * 60)

    result = agent.analyze(
        objective="Analyze what drives customer satisfaction. "
        "Examine correlations with income, age, and category. "
        "Create visualizations and provide statistical evidence.",
        dataset_path=dataset_path,
    )

    # Display results
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)

    print(f"\n📊 Steps completed: {result['steps']}")
    print(f"✅ Success: {result['success']}")
    print(f"📓 Notebook saved to: {result['notebook_path']}")

    # Display findings
    print(f"\n🔍 Findings ({len(result['findings'])} total):")
    print("-" * 60)

    for i, finding in enumerate(result["findings"][:10], 1):  # Show first 10
        print(f"\n{i}. [{finding['type'].upper()}]")
        print(f"   {finding['description']}")
        if finding.get("value"):
            print(f"   Value: {finding['value']}")

    if len(result["findings"]) > 10:
        print(f"\n   ... and {len(result['findings']) - 10} more findings")

    # Display world model updates
    print(f"\n🌍 World Model Updates: {len(result['world_model_updates'])}")

    # Show trajectory
    print(f"\n📈 Analysis Trajectory:")
    print("-" * 60)
    trajectory = agent.get_trajectory()

    for step in trajectory:
        status = "✅" if step["success"] else "❌"
        print(
            f"{status} Step {step['step_number']}: "
            f"{len(step['findings'])} findings "
            f"({step['execution_time']:.2f}s)"
        )

    # Save trajectory
    traj_path = Path("outputs/trajectory.json")
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    agent.save_trajectory(traj_path)
    print(f"\n💾 Trajectory saved to: {traj_path}")

    print(f"\n🎉 Analysis complete! Open the notebook to see the full results:")
    print(f"   {result['notebook_path']}")


if __name__ == "__main__":
    main()
