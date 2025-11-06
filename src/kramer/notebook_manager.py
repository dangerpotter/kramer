"""
Jupyter notebook management for creating and tracking analysis notebooks.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import nbformat
from nbformat.v4 import new_notebook, new_code_cell, new_markdown_cell


class NotebookManager:
    """
    Manages Jupyter notebooks for analysis trajectories.

    Features:
    - Creates notebooks with metadata
    - Adds cells with execution results
    - Saves notebooks with timestamps
    - Tracks analysis provenance
    """

    def __init__(self, notebooks_dir: Path = Path("outputs/notebooks")):
        """
        Initialize the notebook manager.

        Args:
            notebooks_dir: Directory to save notebooks
        """
        self.notebooks_dir = Path(notebooks_dir)
        self.notebooks_dir.mkdir(parents=True, exist_ok=True)

    def create_notebook(
        self,
        objective: str,
        dataset_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> nbformat.NotebookNode:
        """
        Create a new analysis notebook.

        Args:
            objective: Research objective
            dataset_path: Path to dataset being analyzed
            metadata: Additional metadata

        Returns:
            New NotebookNode
        """

        nb = new_notebook()

        # Add metadata
        nb.metadata.update(
            {
                "kramer": {
                    "objective": objective,
                    "dataset_path": dataset_path,
                    "created_at": time.time(),
                    "version": "0.1.0",
                    **(metadata or {}),
                }
            }
        )

        # Add header cell
        header = f"""# Data Analysis: {objective}

**Created:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Dataset:** {dataset_path or 'N/A'}

## Objective
{objective}

---
"""
        nb.cells.append(new_markdown_cell(header))

        # Add setup cell
        setup_code = """# Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline
"""
        nb.cells.append(new_code_cell(setup_code))

        return nb

    def add_code_cell(
        self,
        notebook: nbformat.NotebookNode,
        code: str,
        execution_result=None,
        description: str = "",
    ) -> nbformat.NotebookNode:
        """
        Add a code cell to the notebook.

        Args:
            notebook: Notebook to modify
            code: Code to add
            execution_result: Optional ExecutionResult with outputs
            description: Optional description of the cell

        Returns:
            Updated notebook
        """

        # Add description as markdown if provided
        if description:
            notebook.cells.append(new_markdown_cell(f"### {description}"))

        # Create code cell
        cell = new_code_cell(code)

        # Add execution metadata if available
        if execution_result:
            cell.execution_count = len(
                [c for c in notebook.cells if c.cell_type == "code"]
            )

            # Add outputs
            outputs = []

            # Add stdout
            if execution_result.stdout:
                outputs.append(
                    nbformat.v4.new_output(
                        output_type="stream",
                        name="stdout",
                        text=execution_result.stdout,
                    )
                )

            # Add stderr
            if execution_result.stderr:
                outputs.append(
                    nbformat.v4.new_output(
                        output_type="stream",
                        name="stderr",
                        text=execution_result.stderr,
                    )
                )

            # Add plot outputs
            for plot_path in execution_result.plots:
                if plot_path.exists():
                    # Read image and encode as base64
                    import base64

                    img_data = base64.b64encode(plot_path.read_bytes()).decode("utf-8")
                    outputs.append(
                        nbformat.v4.new_output(
                            output_type="display_data",
                            data={"image/png": img_data},
                            metadata={},
                        )
                    )

            cell.outputs = outputs

            # Add execution metadata
            cell.metadata.update(
                {
                    "execution": {
                        "execution_time": execution_result.execution_time,
                        "success": execution_result.success,
                    }
                }
            )

        notebook.cells.append(cell)
        return notebook

    def add_markdown_cell(
        self,
        notebook: nbformat.NotebookNode,
        content: str,
    ) -> nbformat.NotebookNode:
        """
        Add a markdown cell to the notebook.

        Args:
            notebook: Notebook to modify
            content: Markdown content

        Returns:
            Updated notebook
        """
        notebook.cells.append(new_markdown_cell(content))
        return notebook

    def add_findings_summary(
        self,
        notebook: nbformat.NotebookNode,
        findings: List,
    ) -> nbformat.NotebookNode:
        """
        Add a summary of findings to the notebook.

        Args:
            notebook: Notebook to modify
            findings: List of Finding objects

        Returns:
            Updated notebook
        """

        summary = ["## Analysis Findings\n"]

        # Group by type
        by_type = {}
        for finding in findings:
            if finding.type not in by_type:
                by_type[finding.type] = []
            by_type[finding.type].append(finding)

        # Format each type
        for finding_type, type_findings in by_type.items():
            summary.append(f"\n### {finding_type.title()}s\n")
            for finding in type_findings:
                if finding.value is not None:
                    summary.append(f"- **{finding.description}**: `{finding.value}`")
                else:
                    summary.append(f"- {finding.description}")

        notebook.cells.append(new_markdown_cell("\n".join(summary)))
        return notebook

    def save_notebook(
        self,
        notebook: nbformat.NotebookNode,
        name: Optional[str] = None,
    ) -> Path:
        """
        Save notebook to disk.

        Args:
            notebook: Notebook to save
            name: Optional name (will generate timestamp-based name if not provided)

        Returns:
            Path to saved notebook
        """

        if name is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            name = f"analysis_{timestamp}.ipynb"

        if not name.endswith(".ipynb"):
            name += ".ipynb"

        notebook_path = self.notebooks_dir / name

        # Write notebook
        with open(notebook_path, "w", encoding="utf-8") as f:
            nbformat.write(notebook, f)

        return notebook_path

    def load_notebook(self, path: Path) -> nbformat.NotebookNode:
        """
        Load a notebook from disk.

        Args:
            path: Path to notebook

        Returns:
            NotebookNode
        """
        with open(path, "r", encoding="utf-8") as f:
            return nbformat.read(f, as_version=4)

    def list_notebooks(self) -> List[Path]:
        """
        List all notebooks in the notebooks directory.

        Returns:
            List of notebook paths
        """
        return sorted(self.notebooks_dir.glob("*.ipynb"))

    def get_notebook_metadata(self, path: Path) -> Dict[str, Any]:
        """
        Get metadata from a notebook.

        Args:
            path: Path to notebook

        Returns:
            Metadata dictionary
        """
        notebook = self.load_notebook(path)
        return notebook.metadata.get("kramer", {})

    def create_trajectory_notebook(
        self,
        objective: str,
        dataset_path: str,
        steps: List[Dict[str, Any]],
    ) -> Path:
        """
        Create a complete notebook from a trajectory of analysis steps.

        Args:
            objective: Research objective
            dataset_path: Path to dataset
            steps: List of analysis steps, each with 'code', 'result', 'description'

        Returns:
            Path to saved notebook
        """

        # Create base notebook
        notebook = self.create_notebook(objective, dataset_path)

        # Add each step
        for i, step in enumerate(steps, 1):
            description = step.get("description", f"Step {i}")
            code = step.get("code", "")
            result = step.get("result")

            self.add_code_cell(
                notebook=notebook,
                code=code,
                execution_result=result,
                description=description,
            )

        # Add findings summary if available
        all_findings = []
        for step in steps:
            if "findings" in step:
                all_findings.extend(step["findings"])

        if all_findings:
            self.add_findings_summary(notebook, all_findings)

        # Save notebook
        return self.save_notebook(notebook)
