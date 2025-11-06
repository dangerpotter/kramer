"""Tests for NotebookManager."""

import pytest
from pathlib import Path
import nbformat
from kramer.notebook_manager import NotebookManager
from kramer.code_executor import ExecutionResult
from kramer.result_parser import Finding


class TestNotebookManager:
    """Test the NotebookManager class."""

    def test_create_notebook(self, temp_outputs_dir):
        """Test creating a new notebook."""
        manager = NotebookManager(notebooks_dir=temp_outputs_dir["notebooks"])

        notebook = manager.create_notebook(
            objective="Test analysis",
            dataset_path="/path/to/data.csv",
        )

        assert isinstance(notebook, nbformat.NotebookNode)
        assert "kramer" in notebook.metadata
        assert notebook.metadata["kramer"]["objective"] == "Test analysis"
        assert len(notebook.cells) > 0  # Should have header and setup cells

    def test_add_code_cell(self, temp_outputs_dir):
        """Test adding a code cell to notebook."""
        manager = NotebookManager(notebooks_dir=temp_outputs_dir["notebooks"])

        notebook = manager.create_notebook("Test", "/data.csv")
        initial_cells = len(notebook.cells)

        code = "print('Hello')"
        manager.add_code_cell(notebook, code)

        assert len(notebook.cells) == initial_cells + 1
        assert notebook.cells[-1].cell_type == "code"
        assert notebook.cells[-1].source == code

    def test_add_code_cell_with_execution_result(self, temp_outputs_dir):
        """Test adding a code cell with execution results."""
        manager = NotebookManager(notebooks_dir=temp_outputs_dir["notebooks"])

        notebook = manager.create_notebook("Test", "/data.csv")

        code = "print('test output')"
        exec_result = ExecutionResult(
            success=True,
            stdout="test output\n",
            stderr="",
            execution_time=0.5,
        )

        manager.add_code_cell(
            notebook,
            code,
            execution_result=exec_result,
            description="Test Cell",
        )

        # Should have added description cell + code cell
        assert notebook.cells[-2].cell_type == "markdown"
        assert "Test Cell" in notebook.cells[-2].source

        code_cell = notebook.cells[-1]
        assert code_cell.cell_type == "code"
        assert len(code_cell.outputs) > 0
        assert code_cell.outputs[0]["text"] == "test output\n"

    def test_add_markdown_cell(self, temp_outputs_dir):
        """Test adding a markdown cell."""
        manager = NotebookManager(notebooks_dir=temp_outputs_dir["notebooks"])

        notebook = manager.create_notebook("Test", "/data.csv")
        initial_cells = len(notebook.cells)

        markdown = "## Test Section\nThis is a test."
        manager.add_markdown_cell(notebook, markdown)

        assert len(notebook.cells) == initial_cells + 1
        assert notebook.cells[-1].cell_type == "markdown"
        assert notebook.cells[-1].source == markdown

    def test_add_findings_summary(self, temp_outputs_dir):
        """Test adding findings summary."""
        manager = NotebookManager(notebooks_dir=temp_outputs_dir["notebooks"])

        notebook = manager.create_notebook("Test", "/data.csv")

        findings = [
            Finding(
                type="statistic",
                description="Mean value",
                value=5.2,
            ),
            Finding(
                type="insight",
                description="Strong correlation found",
            ),
        ]

        manager.add_findings_summary(notebook, findings)

        # Check that summary cell was added
        summary_cell = notebook.cells[-1]
        assert summary_cell.cell_type == "markdown"
        assert "Findings" in summary_cell.source
        assert "statistic" in summary_cell.source.lower()

    def test_save_and_load_notebook(self, temp_outputs_dir):
        """Test saving and loading notebooks."""
        manager = NotebookManager(notebooks_dir=temp_outputs_dir["notebooks"])

        notebook = manager.create_notebook("Test", "/data.csv")
        manager.add_code_cell(notebook, "print('test')")

        # Save
        path = manager.save_notebook(notebook, name="test_notebook")

        assert path.exists()
        assert path.suffix == ".ipynb"

        # Load
        loaded_notebook = manager.load_notebook(path)

        assert isinstance(loaded_notebook, nbformat.NotebookNode)
        assert loaded_notebook.metadata["kramer"]["objective"] == "Test"

    def test_list_notebooks(self, temp_outputs_dir):
        """Test listing notebooks."""
        manager = NotebookManager(notebooks_dir=temp_outputs_dir["notebooks"])

        # Create multiple notebooks
        for i in range(3):
            notebook = manager.create_notebook(f"Test {i}", "/data.csv")
            manager.save_notebook(notebook, name=f"notebook_{i}")

        notebooks = manager.list_notebooks()

        assert len(notebooks) == 3
        assert all(p.suffix == ".ipynb" for p in notebooks)

    def test_get_notebook_metadata(self, temp_outputs_dir):
        """Test getting notebook metadata."""
        manager = NotebookManager(notebooks_dir=temp_outputs_dir["notebooks"])

        notebook = manager.create_notebook(
            objective="Test Objective",
            dataset_path="/data.csv",
            metadata={"custom_field": "value"},
        )

        path = manager.save_notebook(notebook)

        metadata = manager.get_notebook_metadata(path)

        assert metadata["objective"] == "Test Objective"
        assert metadata["custom_field"] == "value"

    def test_create_trajectory_notebook(self, temp_outputs_dir):
        """Test creating a complete trajectory notebook."""
        manager = NotebookManager(notebooks_dir=temp_outputs_dir["notebooks"])

        steps = [
            {
                "description": "Load data",
                "code": "import pandas as pd\ndf = pd.read_csv('data.csv')",
                "result": ExecutionResult(
                    success=True,
                    stdout="Data loaded\n",
                    stderr="",
                ),
            },
            {
                "description": "Analyze data",
                "code": "print(df.describe())",
                "result": ExecutionResult(
                    success=True,
                    stdout="Statistics...\n",
                    stderr="",
                ),
                "findings": [
                    Finding(type="statistic", description="Count: 100", value=100)
                ],
            },
        ]

        path = manager.create_trajectory_notebook(
            objective="Test Analysis",
            dataset_path="/data.csv",
            steps=steps,
        )

        assert path.exists()

        # Load and verify
        notebook = manager.load_notebook(path)
        assert "Test Analysis" in notebook.cells[0].source

        # Should have header, setup, and 2 analysis steps
        code_cells = [c for c in notebook.cells if c.cell_type == "code"]
        assert len(code_cells) >= 3  # setup + 2 steps
