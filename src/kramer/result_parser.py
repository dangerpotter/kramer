"""
Result parser for extracting structured findings from code execution results.
"""

import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
import json


@dataclass
class Finding:
    """A single quantitative or qualitative finding from analysis."""

    type: str  # "statistic", "correlation", "plot", "insight", "error"
    description: str
    value: Any = None
    code_provenance: str = ""
    plot_path: Optional[Path] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert finding to dictionary."""
        result = asdict(self)
        if self.plot_path:
            result["plot_path"] = str(self.plot_path)
        return result


@dataclass
class AnalysisResults:
    """Collection of findings from an analysis."""

    findings: List[Finding]
    plots: List[Path]
    raw_output: str
    execution_time: float
    success: bool
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary."""
        return {
            "findings": [f.to_dict() for f in self.findings],
            "plots": [str(p) for p in self.plots],
            "raw_output": self.raw_output,
            "execution_time": self.execution_time,
            "success": self.success,
            "error": self.error,
        }

    def get_findings_by_type(self, finding_type: str) -> List[Finding]:
        """Get all findings of a specific type."""
        return [f for f in self.findings if f.type == finding_type]


class ResultParser:
    """
    Parses execution results to extract structured findings.

    Extracts:
    - Statistical results (means, p-values, correlations)
    - Visualizations with descriptions
    - Insights and conclusions
    - Errors and warnings
    """

    # Patterns for common statistical outputs
    PATTERNS = {
        "mean": r"mean[:\s=]+(\d+\.?\d*)",
        "median": r"median[:\s=]+(\d+\.?\d*)",
        "std": r"std(?:dev)?[:\s=]+(\d+\.?\d*)",
        "p_value": r"p(?:-value)?[:\s=<]+(\d+\.?\d*(?:e-?\d+)?)",
        "correlation": r"correlation[:\s=]+(-?\d+\.?\d*)",
        "r_squared": r"r[²2]?[:\s=]+(\d+\.?\d*)",
        "count": r"count[:\s=]+(\d+)",
        "percentage": r"(\d+\.?\d*)%",
    }

    def __init__(self):
        """Initialize the result parser."""
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.PATTERNS.items()
        }

    def parse(
        self,
        execution_result,
        code: str = "",
    ) -> AnalysisResults:
        """
        Parse execution results to extract findings.

        Args:
            execution_result: ExecutionResult from CodeExecutor
            code: The code that was executed (for provenance)

        Returns:
            AnalysisResults with structured findings
        """

        findings = []

        # Extract from stdout
        if execution_result.stdout:
            findings.extend(self._parse_output(execution_result.stdout, code))

        # Add plot findings
        for plot_path in execution_result.plots:
            findings.append(
                Finding(
                    type="plot",
                    description=f"Generated visualization: {plot_path.name}",
                    plot_path=plot_path,
                    code_provenance=code,
                    metadata={"plot_file": str(plot_path)},
                )
            )

        # Add error findings if present
        if not execution_result.success:
            findings.append(
                Finding(
                    type="error",
                    description=execution_result.error or "Execution failed",
                    value=execution_result.stderr,
                    code_provenance=code,
                    metadata={"stderr": execution_result.stderr},
                )
            )

        return AnalysisResults(
            findings=findings,
            plots=execution_result.plots,
            raw_output=execution_result.stdout,
            execution_time=execution_result.execution_time,
            success=execution_result.success,
            error=execution_result.error,
        )

    def _parse_output(self, output: str, code: str) -> List[Finding]:
        """Parse text output to extract findings."""

        findings = []

        # Extract statistical values using patterns
        for stat_name, pattern in self.compiled_patterns.items():
            matches = pattern.findall(output)
            for match in matches:
                # Find the line containing this match for context
                for line in output.split("\n"):
                    if match in line:
                        findings.append(
                            Finding(
                                type="statistic",
                                description=f"{stat_name}: {match}",
                                value=self._parse_value(match),
                                code_provenance=code,
                                metadata={
                                    "stat_name": stat_name,
                                    "context_line": line.strip(),
                                },
                            )
                        )
                        break

        # Extract insights from print statements
        insight_patterns = [
            r"(?:insight|conclusion|finding|result):\s*(.+)",
            r"(?:we found|analysis shows|results indicate):\s*(.+)",
        ]

        for pattern in insight_patterns:
            matches = re.finditer(pattern, output, re.IGNORECASE)
            for match in matches:
                findings.append(
                    Finding(
                        type="insight",
                        description=match.group(1).strip(),
                        code_provenance=code,
                        metadata={"source": "output_text"},
                    )
                )

        # Extract dataframe summaries
        df_summary_pattern = r"(.*?):\s*\n((?:\s+\w+\s+[\d.]+\n)+)"
        matches = re.finditer(df_summary_pattern, output)
        for match in matches:
            findings.append(
                Finding(
                    type="statistic",
                    description=f"DataFrame summary: {match.group(1).strip()}",
                    value=match.group(2).strip(),
                    code_provenance=code,
                    metadata={"source": "dataframe_summary"},
                )
            )

        return findings

    def _parse_value(self, value_str: str) -> Any:
        """Parse a string value to appropriate type."""
        try:
            # Try integer
            if "." not in value_str and "e" not in value_str.lower():
                return int(value_str)
            # Try float
            return float(value_str)
        except ValueError:
            return value_str

    def extract_world_model_updates(
        self,
        results: AnalysisResults,
        objective: str,
    ) -> List[Dict[str, Any]]:
        """
        Extract updates suitable for adding to a world model.

        Args:
            results: Analysis results
            objective: The research objective

        Returns:
            List of world model updates with provenance
        """

        updates = []

        for finding in results.findings:
            if finding.type in ["statistic", "insight", "correlation"]:
                updates.append(
                    {
                        "type": finding.type,
                        "content": finding.description,
                        "value": finding.value,
                        "objective": objective,
                        "provenance": {
                            "code": finding.code_provenance,
                            "execution_time": results.execution_time,
                        },
                        "metadata": finding.metadata,
                    }
                )
            elif finding.type == "plot":
                updates.append(
                    {
                        "type": "visualization",
                        "content": finding.description,
                        "plot_path": str(finding.plot_path),
                        "objective": objective,
                        "provenance": {
                            "code": finding.code_provenance,
                            "execution_time": results.execution_time,
                        },
                    }
                )

        return updates

    def format_findings_for_display(self, results: AnalysisResults) -> str:
        """Format findings as human-readable text."""

        lines = []
        lines.append(f"Analysis Results (execution time: {results.execution_time:.2f}s)")
        lines.append("=" * 60)

        if not results.success:
            lines.append(f"\n❌ Execution failed: {results.error}")
            return "\n".join(lines)

        # Group findings by type
        by_type = {}
        for finding in results.findings:
            if finding.type not in by_type:
                by_type[finding.type] = []
            by_type[finding.type].append(finding)

        # Display by type
        for finding_type, findings in by_type.items():
            lines.append(f"\n{finding_type.upper()}:")
            for finding in findings:
                if finding.value is not None:
                    lines.append(f"  - {finding.description}: {finding.value}")
                else:
                    lines.append(f"  - {finding.description}")

        # Show plots
        if results.plots:
            lines.append(f"\nGENERATED {len(results.plots)} PLOT(S):")
            for plot in results.plots:
                lines.append(f"  - {plot}")

        return "\n".join(lines)
