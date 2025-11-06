"""Integration tests for DataAnalysisAgent.

Note: These tests require an ANTHROPIC_API_KEY environment variable.
They are skipped if the key is not available.
"""

import pytest
import os
from pathlib import Path
from kramer.data_analysis_agent import DataAnalysisAgent, AgentConfig


# Skip all tests in this file if no API key
pytestmark = pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)


class TestDataAnalysisAgentIntegration:
    """Integration tests for the full agent."""

    def test_agent_initialization(self, temp_outputs_dir):
        """Test initializing the agent."""
        config = AgentConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_iterations=2,
        )

        agent = DataAnalysisAgent(
            config=config,
            notebooks_dir=temp_outputs_dir["notebooks"],
            plots_dir=temp_outputs_dir["plots"],
        )

        assert agent is not None
        assert agent.client is not None

    def test_simple_analysis(self, temp_outputs_dir, sample_csv):
        """Test running a simple analysis."""
        config = AgentConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_iterations=2,
            use_extended_thinking=True,
        )

        agent = DataAnalysisAgent(
            config=config,
            notebooks_dir=temp_outputs_dir["notebooks"],
            plots_dir=temp_outputs_dir["plots"],
        )

        result = agent.analyze(
            objective="Calculate basic statistics (mean, median, std) for age and income",
            dataset_path=str(sample_csv),
        )

        assert result is not None
        assert "notebook_path" in result
        assert Path(result["notebook_path"]).exists()
        assert "findings" in result
        assert "steps" in result
        assert result["steps"] > 0

    def test_analysis_with_visualization(self, temp_outputs_dir, sample_csv):
        """Test analysis that generates visualizations."""
        config = AgentConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_iterations=3,
            use_extended_thinking=True,
        )

        agent = DataAnalysisAgent(
            config=config,
            notebooks_dir=temp_outputs_dir["notebooks"],
            plots_dir=temp_outputs_dir["plots"],
        )

        result = agent.analyze(
            objective="Create a histogram of age distribution and a scatter plot of income vs satisfaction",
            dataset_path=str(sample_csv),
        )

        assert result is not None
        assert Path(result["notebook_path"]).exists()

        # Check for plots in findings
        plot_findings = [
            f for f in result["findings"] if f["type"] == "plot"
        ]
        # Should have generated at least one plot
        # (might not be exactly 2 depending on what the agent does)
        assert len(plot_findings) > 0

    def test_world_model_updates(self, temp_outputs_dir, sample_csv):
        """Test that world model updates are generated."""
        config = AgentConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_iterations=2,
        )

        agent = DataAnalysisAgent(
            config=config,
            notebooks_dir=temp_outputs_dir["notebooks"],
            plots_dir=temp_outputs_dir["plots"],
        )

        result = agent.analyze(
            objective="Analyze the correlation between income and satisfaction",
            dataset_path=str(sample_csv),
        )

        assert "world_model_updates" in result
        assert isinstance(result["world_model_updates"], list)

        # Each update should have required fields
        if result["world_model_updates"]:
            update = result["world_model_updates"][0]
            assert "objective" in update
            assert "provenance" in update

    def test_trajectory_saved(self, temp_outputs_dir, sample_csv):
        """Test that analysis trajectory is properly tracked."""
        config = AgentConfig(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            max_iterations=2,
        )

        agent = DataAnalysisAgent(
            config=config,
            notebooks_dir=temp_outputs_dir["notebooks"],
            plots_dir=temp_outputs_dir["plots"],
        )

        result = agent.analyze(
            objective="Summarize the dataset",
            dataset_path=str(sample_csv),
        )

        trajectory = agent.get_trajectory()

        assert len(trajectory) > 0
        assert all("step_number" in step for step in trajectory)
        assert all("code" in step for step in trajectory)
        assert all("success" in step for step in trajectory)

        # Save trajectory
        traj_path = temp_outputs_dir["outputs"] / "trajectory.json"
        agent.save_trajectory(traj_path)

        assert traj_path.exists()
