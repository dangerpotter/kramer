"""
Advanced usage example for the Kramer Data Analysis Agent.

This script demonstrates:
1. Using world model context
2. Customizing the agent configuration
3. Programmatic access to results
4. Custom result processing
"""

import os
from pathlib import Path
from kramer.data_analysis_agent import DataAnalysisAgent, AgentConfig
from kramer.result_parser import ResultParser


def main():
    """Run an advanced analysis with custom configuration."""

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    # Advanced configuration
    config = AgentConfig(
        model="claude-sonnet-4-20250514",
        max_iterations=7,
        use_extended_thinking=True,
        timeout=600,  # Longer timeout for complex analyses
        temperature=1.0,  # Control randomness
    )

    # Initialize agent with custom directories
    outputs_dir = Path("outputs/advanced_analysis")
    agent = DataAnalysisAgent(
        config=config,
        notebooks_dir=outputs_dir / "notebooks",
        plots_dir=outputs_dir / "plots",
    )

    # Prepare world model context (from previous analyses)
    world_model_context = {
        "previous_findings": [
            "Income is positively correlated with satisfaction",
            "Category A shows highest purchase amounts",
        ],
        "hypotheses": [
            "Age might moderate the income-satisfaction relationship",
            "Purchase patterns differ by category",
        ],
        "constraints": ["Sample size is 100 observations", "Data is cross-sectional"],
    }

    # Run analysis with context
    print("Running advanced analysis with world model context...")

    result = agent.analyze(
        objective=(
            "Building on previous findings that income correlates with satisfaction, "
            "investigate whether age moderates this relationship. "
            "Test if the income-satisfaction correlation differs across age groups. "
            "Also examine if category differences in purchase amounts are statistically significant."
        ),
        dataset_path="data/sample_data.csv",
        world_model_context=world_model_context,
    )

    # Process results programmatically
    print("\n" + "=" * 60)
    print("ADVANCED ANALYSIS RESULTS")
    print("=" * 60)

    # Extract different types of findings
    parser = ResultParser()

    all_findings = result["findings"]

    statistics = [f for f in all_findings if f["type"] == "statistic"]
    insights = [f for f in all_findings if f["type"] == "insight"]
    plots = [f for f in all_findings if f["type"] == "plot"]
    errors = [f for f in all_findings if f["type"] == "error"]

    print(f"\n📊 Statistics extracted: {len(statistics)}")
    for stat in statistics:
        print(f"   - {stat['description']}")

    print(f"\n💡 Insights generated: {len(insights)}")
    for insight in insights:
        print(f"   - {insight['description']}")

    print(f"\n📈 Visualizations created: {len(plots)}")
    for plot in plots:
        print(f"   - {plot['description']}")
        if plot.get("plot_path"):
            print(f"     Path: {plot['plot_path']}")

    if errors:
        print(f"\n⚠️  Errors encountered: {len(errors)}")
        for error in errors:
            print(f"   - {error['description']}")

    # Access world model updates
    print(f"\n🌍 World Model Updates:")
    for i, update in enumerate(result["world_model_updates"][:5], 1):
        print(f"\n{i}. Type: {update['type']}")
        print(f"   Content: {update['content']}")
        if "value" in update:
            print(f"   Value: {update['value']}")
        print(f"   Provenance: {len(update['provenance']['code'])} chars of code")

    # Analyze the trajectory
    trajectory = agent.get_trajectory()

    print(f"\n📋 Trajectory Analysis:")
    total_time = sum(step["execution_time"] for step in trajectory)
    successful_steps = sum(1 for step in trajectory if step["success"])

    print(f"   Total execution time: {total_time:.2f}s")
    print(f"   Successful steps: {successful_steps}/{len(trajectory)}")
    print(f"   Average time per step: {total_time / len(trajectory):.2f}s")

    # Find the most productive step
    if trajectory:
        most_productive = max(trajectory, key=lambda s: len(s["findings"]))
        print(f"\n🏆 Most productive step:")
        print(f"   Step {most_productive['step_number']}: "
              f"{len(most_productive['findings'])} findings")

    # Save structured output
    import json

    output_file = outputs_dir / "analysis_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(
            {
                "objective": (
                    "Investigate age moderation of income-satisfaction relationship"
                ),
                "dataset": "data/sample_data.csv",
                "world_model_context": world_model_context,
                "results": result,
                "trajectory": trajectory,
            },
            f,
            indent=2,
        )

    print(f"\n💾 Full results saved to: {output_file}")
    print(f"📓 Notebook available at: {result['notebook_path']}")


if __name__ == "__main__":
    main()
