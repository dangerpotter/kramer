"""Pytest configuration and fixtures."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    np.random.seed(42)

    data = {
        "id": range(1, 101),
        "age": np.random.randint(18, 80, 100),
        "income": np.random.normal(50000, 15000, 100),
        "satisfaction": np.random.randint(1, 11, 100),
        "category": np.random.choice(["A", "B", "C"], 100),
        "purchase_amount": np.random.exponential(100, 100),
    }

    df = pd.DataFrame(data)

    # Add some correlation
    df["satisfaction"] = df["satisfaction"] + (df["income"] / 10000).astype(int)
    df["satisfaction"] = df["satisfaction"].clip(1, 10)

    return df


@pytest.fixture
def sample_csv(sample_dataset, tmp_path):
    """Create a temporary CSV file with sample data."""
    csv_path = tmp_path / "sample_data.csv"
    sample_dataset.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def temp_outputs_dir(tmp_path):
    """Create temporary output directories."""
    outputs_dir = tmp_path / "outputs"
    notebooks_dir = outputs_dir / "notebooks"
    plots_dir = outputs_dir / "plots"

    notebooks_dir.mkdir(parents=True)
    plots_dir.mkdir(parents=True)

    return {
        "outputs": outputs_dir,
        "notebooks": notebooks_dir,
        "plots": plots_dir,
    }


@pytest.fixture(autouse=True)
def change_test_dir(tmp_path, monkeypatch):
    """Change to temporary directory for tests."""
    monkeypatch.chdir(tmp_path)
