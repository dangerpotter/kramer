"""Tests for ResultParser."""

import pytest
from pathlib import Path
from kramer.result_parser import ResultParser, Finding, AnalysisResults
from kramer.code_executor import ExecutionResult


class TestResultParser:
    """Test the ResultParser class."""

    def test_parse_statistics(self):
        """Test parsing statistical outputs."""
        parser = ResultParser()

        # Create mock execution result
        stdout = """
Analysis Results:
Mean: 5.2
Median: 4.8
Std: 1.3
P-value: 0.03
Correlation: 0.85
"""

        exec_result = ExecutionResult(
            success=True,
            stdout=stdout,
            stderr="",
        )

        results = parser.parse(exec_result, code="test code")

        assert results.success is True
        assert len(results.findings) > 0

        # Check that statistics were extracted
        stat_findings = results.get_findings_by_type("statistic")
        assert len(stat_findings) > 0

        # Check specific values
        stat_names = [f.metadata.get("stat_name") for f in stat_findings]
        assert "mean" in stat_names
        assert "p_value" in stat_names

    def test_parse_insights(self):
        """Test parsing insights from output."""
        parser = ResultParser()

        stdout = """
Insight: The data shows a strong positive correlation
Finding: Users with higher income report higher satisfaction
Conclusion: Income is a significant predictor of satisfaction
"""

        exec_result = ExecutionResult(
            success=True,
            stdout=stdout,
            stderr="",
        )

        results = parser.parse(exec_result)

        insight_findings = results.get_findings_by_type("insight")
        assert len(insight_findings) > 0

    def test_parse_plots(self, tmp_path):
        """Test parsing plot outputs."""
        parser = ResultParser()

        # Create mock plots
        plot1 = tmp_path / "plot_001.png"
        plot2 = tmp_path / "plot_002.png"
        plot1.write_bytes(b"fake image data")
        plot2.write_bytes(b"fake image data")

        exec_result = ExecutionResult(
            success=True,
            stdout="",
            stderr="",
            plots=[plot1, plot2],
        )

        results = parser.parse(exec_result, code="plotting code")

        assert len(results.plots) == 2
        plot_findings = results.get_findings_by_type("plot")
        assert len(plot_findings) == 2

    def test_parse_errors(self):
        """Test parsing execution errors."""
        parser = ResultParser()

        exec_result = ExecutionResult(
            success=False,
            stdout="",
            stderr="ValueError: Invalid data",
            error="ValueError: Invalid data",
        )

        results = parser.parse(exec_result)

        assert results.success is False
        assert results.error is not None

        error_findings = results.get_findings_by_type("error")
        assert len(error_findings) > 0

    def test_extract_world_model_updates(self):
        """Test extracting world model updates."""
        parser = ResultParser()

        findings = [
            Finding(
                type="statistic",
                description="Mean income: 50000",
                value=50000,
                code_provenance="df['income'].mean()",
            ),
            Finding(
                type="insight",
                description="Income correlates with satisfaction",
                code_provenance="correlation analysis",
            ),
        ]

        results = AnalysisResults(
            findings=findings,
            plots=[],
            raw_output="",
            execution_time=1.5,
            success=True,
        )

        updates = parser.extract_world_model_updates(
            results=results,
            objective="Analyze customer satisfaction",
        )

        assert len(updates) > 0
        assert all("objective" in u for u in updates)
        assert all("provenance" in u for u in updates)

    def test_format_findings_for_display(self):
        """Test formatting findings for display."""
        parser = ResultParser()

        findings = [
            Finding(
                type="statistic",
                description="Mean",
                value=5.2,
            ),
            Finding(
                type="insight",
                description="Strong correlation found",
            ),
        ]

        results = AnalysisResults(
            findings=findings,
            plots=[],
            raw_output="test output",
            execution_time=2.5,
            success=True,
        )

        display = parser.format_findings_for_display(results)

        assert "2.5" in display  # execution time
        assert "STATISTIC" in display
        assert "INSIGHT" in display
        assert "Mean" in display

    def test_value_parsing(self):
        """Test parsing different value types."""
        parser = ResultParser()

        # Integer
        assert parser._parse_value("42") == 42

        # Float
        assert parser._parse_value("3.14") == 3.14

        # Scientific notation
        assert parser._parse_value("1.5e-3") == 0.0015

        # String (when parsing fails)
        assert parser._parse_value("not a number") == "not a number"
