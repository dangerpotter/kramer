"""Tests for CodeExecutor."""

import pytest
from pathlib import Path
from kramer.code_executor import CodeExecutor, ExecutionResult


class TestCodeExecutor:
    """Test the CodeExecutor class."""

    def test_simple_execution(self, temp_outputs_dir):
        """Test executing simple Python code."""
        executor = CodeExecutor(plots_dir=temp_outputs_dir["plots"])

        code = """
print("Hello, World!")
result = 2 + 2
print(f"2 + 2 = {result}")
"""

        result = executor.execute(code)

        assert result.success is True
        assert "Hello, World!" in result.stdout
        assert "2 + 2 = 4" in result.stdout
        assert result.execution_time > 0

    def test_execution_with_error(self, temp_outputs_dir):
        """Test handling of execution errors."""
        executor = CodeExecutor(plots_dir=temp_outputs_dir["plots"])

        code = """
# This will raise an error
x = 1 / 0
"""

        result = executor.execute(code)

        assert result.success is False
        assert result.error is not None
        assert "ZeroDivisionError" in result.stderr

    def test_execution_timeout(self, temp_outputs_dir):
        """Test timeout handling."""
        executor = CodeExecutor(timeout=1, plots_dir=temp_outputs_dir["plots"])

        code = """
import time
time.sleep(10)  # Will timeout
"""

        result = executor.execute(code)

        assert result.success is False
        assert "timed out" in result.error.lower()

    def test_plot_capture(self, temp_outputs_dir):
        """Test capturing matplotlib plots."""
        executor = CodeExecutor(plots_dir=temp_outputs_dir["plots"])

        code = """
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("Sine Wave")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
"""

        result = executor.execute(code, capture_plots=True)

        assert result.success is True
        assert len(result.plots) == 1
        assert result.plots[0].exists()
        assert result.plots[0].suffix == ".png"

    def test_multiple_plots(self, temp_outputs_dir):
        """Test capturing multiple plots."""
        executor = CodeExecutor(plots_dir=temp_outputs_dir["plots"])

        code = """
import matplotlib.pyplot as plt
import numpy as np

# Plot 1
plt.figure()
plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Plot 1")
plt.show()

# Plot 2
plt.figure()
plt.plot([1, 2, 3], [1, 2, 3])
plt.title("Plot 2")
plt.show()
"""

        result = executor.execute(code, capture_plots=True)

        assert result.success is True
        assert len(result.plots) == 2

    def test_execution_with_context(self, temp_outputs_dir):
        """Test passing context variables."""
        executor = CodeExecutor(plots_dir=temp_outputs_dir["plots"])

        code = """
print(f"Dataset path: {dataset_path}")
print(f"Config value: {config_value}")
"""

        context = {
            "dataset_path": "/path/to/data.csv",
            "config_value": 42,
        }

        result = executor.execute(code, context=context)

        assert result.success is True
        assert "/path/to/data.csv" in result.stdout
        assert "42" in result.stdout

    def test_dataframe_operations(self, temp_outputs_dir, sample_csv):
        """Test executing pandas operations."""
        executor = CodeExecutor(plots_dir=temp_outputs_dir["plots"])

        code = f"""
import pandas as pd

df = pd.read_csv("{sample_csv}")
print(f"Shape: {{df.shape}}")
print(f"Mean age: {{df['age'].mean():.2f}}")
print(f"Mean income: {{df['income'].mean():.2f}}")
"""

        result = executor.execute(code)

        assert result.success is True
        assert "Shape:" in result.stdout
        assert "Mean age:" in result.stdout
        assert "Mean income:" in result.stdout
