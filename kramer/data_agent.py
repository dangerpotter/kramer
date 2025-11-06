"""Data Analysis Agent"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)


class DataAgent:
    """Agent for performing data analysis tasks"""

    def __init__(self, dataset_path: str):
        """Initialize with a dataset"""
        self.dataset_path = dataset_path
        self.df: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}
        self._load_dataset()

    def _load_dataset(self):
        """Load and prepare the dataset"""
        try:
            self.df = pd.read_csv(self.dataset_path)
            self._compute_metadata()
            logger.info(f"Loaded dataset: {self.dataset_path} with {len(self.df)} rows and {len(self.df.columns)} columns")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise

    def _compute_metadata(self):
        """Compute dataset metadata"""
        if self.df is None:
            return

        self.metadata = {
            "n_rows": len(self.df),
            "n_cols": len(self.df.columns),
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "numeric_columns": list(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": list(self.df.select_dtypes(include=['object', 'category']).columns),
            "missing_values": {col: int(self.df[col].isna().sum()) for col in self.df.columns}
        }

    def explore_structure(self) -> Dict[str, Any]:
        """Initial exploratory analysis of dataset structure"""
        if self.df is None:
            return {"error": "No dataset loaded"}

        results = {
            "task": "explore_structure",
            "metadata": self.metadata,
            "summary_statistics": {},
            "findings": []
        }

        # Summary statistics for numeric columns
        numeric_cols = self.metadata["numeric_columns"]
        if numeric_cols:
            results["summary_statistics"] = self.df[numeric_cols].describe().to_dict()

        # Generate initial findings
        findings = []
        findings.append(f"Dataset contains {self.metadata['n_rows']} samples with {self.metadata['n_cols']} features")

        if numeric_cols:
            findings.append(f"Found {len(numeric_cols)} numeric features: {', '.join(numeric_cols)}")

        if self.metadata["categorical_columns"]:
            findings.append(f"Found {len(self.metadata['categorical_columns'])} categorical features")

        # Check for missing values
        missing = {k: v for k, v in self.metadata["missing_values"].items() if v > 0}
        if missing:
            findings.append(f"Missing values detected in columns: {', '.join(missing.keys())}")
        else:
            findings.append("No missing values detected")

        results["findings"] = findings
        return results

    def test_correlation(self, feature1: Optional[str] = None,
                        feature2: Optional[str] = None) -> Dict[str, Any]:
        """Test correlations between numeric features"""
        if self.df is None:
            return {"error": "No dataset loaded"}

        numeric_cols = self.metadata["numeric_columns"]
        if len(numeric_cols) < 2:
            return {"error": "Need at least 2 numeric columns for correlation analysis"}

        results = {
            "task": "test_correlation",
            "findings": []
        }

        # Compute correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        results["correlation_matrix"] = corr_matrix.to_dict()

        # Find strong correlations
        findings = []
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.7:
                    findings.append(
                        f"Strong correlation ({corr:.3f}) between {col1} and {col2}"
                    )
                elif abs(corr) > 0.5:
                    findings.append(
                        f"Moderate correlation ({corr:.3f}) between {col1} and {col2}"
                    )

        if not findings:
            findings.append("No strong correlations found between features")

        results["findings"] = findings
        return results

    def test_distribution(self, feature: Optional[str] = None) -> Dict[str, Any]:
        """Test distribution of features"""
        if self.df is None:
            return {"error": "No dataset loaded"}

        numeric_cols = self.metadata["numeric_columns"]
        if not numeric_cols:
            return {"error": "No numeric columns found"}

        # Test all numeric columns if feature not specified
        if feature is None:
            features_to_test = numeric_cols[:3]  # Test first 3
        else:
            features_to_test = [feature] if feature in numeric_cols else []

        results = {
            "task": "test_distribution",
            "findings": [],
            "tests": {}
        }

        findings = []
        for col in features_to_test:
            data = self.df[col].dropna()

            # Normality test (Shapiro-Wilk)
            if len(data) >= 3:
                stat, p_value = stats.shapiro(data[:5000])  # Limit sample size
                results["tests"][col] = {
                    "shapiro_wilk": {"statistic": float(stat), "p_value": float(p_value)},
                    "mean": float(data.mean()),
                    "std": float(data.std()),
                    "skewness": float(data.skew()),
                    "kurtosis": float(data.kurtosis())
                }

                if p_value > 0.05:
                    findings.append(f"{col} appears normally distributed (p={p_value:.3f})")
                else:
                    findings.append(f"{col} is not normally distributed (p={p_value:.3f})")

                # Check for skewness
                skew = data.skew()
                if abs(skew) > 1:
                    findings.append(f"{col} shows {'positive' if skew > 0 else 'negative'} skewness ({skew:.2f})")

        results["findings"] = findings
        return results

    def test_group_differences(self, target_col: Optional[str] = None) -> Dict[str, Any]:
        """Test for differences between groups"""
        if self.df is None:
            return {"error": "No dataset loaded"}

        categorical_cols = self.metadata["categorical_columns"]
        numeric_cols = self.metadata["numeric_columns"]

        if not categorical_cols or not numeric_cols:
            return {"error": "Need both categorical and numeric columns"}

        # Use first categorical column as target if not specified
        if target_col is None:
            target_col = categorical_cols[0]
        elif target_col not in categorical_cols:
            return {"error": f"Column {target_col} not found or not categorical"}

        results = {
            "task": "test_group_differences",
            "target_column": target_col,
            "findings": [],
            "tests": {}
        }

        # Get unique groups
        groups = self.df[target_col].unique()
        results["groups"] = list(map(str, groups))

        findings = []
        findings.append(f"Analyzing differences across {len(groups)} groups in {target_col}")

        # Test each numeric feature
        for num_col in numeric_cols[:3]:  # Test first 3 numeric columns
            group_data = [self.df[self.df[target_col] == g][num_col].dropna() for g in groups]

            # Filter out empty groups
            group_data = [g for g in group_data if len(g) > 0]

            if len(group_data) >= 2:
                # ANOVA test
                f_stat, p_value = stats.f_oneway(*group_data)
                results["tests"][num_col] = {
                    "anova": {"f_statistic": float(f_stat), "p_value": float(p_value)},
                    "group_means": {str(g): float(self.df[self.df[target_col] == g][num_col].mean())
                                  for g in groups}
                }

                if p_value < 0.05:
                    findings.append(
                        f"Significant difference in {num_col} across {target_col} groups (p={p_value:.4f})"
                    )
                else:
                    findings.append(
                        f"No significant difference in {num_col} across {target_col} groups (p={p_value:.4f})"
                    )

        results["findings"] = findings
        return results

    def perform_pca(self) -> Dict[str, Any]:
        """Perform principal component analysis"""
        if self.df is None:
            return {"error": "No dataset loaded"}

        numeric_cols = self.metadata["numeric_columns"]
        if len(numeric_cols) < 2:
            return {"error": "Need at least 2 numeric columns for PCA"}

        # Prepare data
        X = self.df[numeric_cols].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA
        n_components = min(len(numeric_cols), 4)
        pca = PCA(n_components=n_components)
        pca.fit(X_scaled)

        results = {
            "task": "pca",
            "n_components": n_components,
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "findings": []
        }

        findings = []
        findings.append(f"First {n_components} components explain {pca.explained_variance_ratio_.sum():.1%} of variance")

        for i, var in enumerate(pca.explained_variance_ratio_):
            findings.append(f"PC{i+1} explains {var:.1%} of variance")

        # Find dominant features for each component
        components = pd.DataFrame(
            pca.components_,
            columns=numeric_cols,
            index=[f"PC{i+1}" for i in range(n_components)]
        )
        results["component_loadings"] = components.to_dict()

        for i in range(n_components):
            loadings = components.iloc[i].abs().sort_values(ascending=False)
            top_features = loadings.head(2)
            findings.append(
                f"PC{i+1} dominated by: {', '.join(f'{feat} ({loadings[feat]:.2f})' for feat in top_features.index)}"
            )

        results["findings"] = findings
        return results

    def test_hypothesis(self, hypothesis_text: str) -> Dict[str, Any]:
        """Test a specific hypothesis"""
        results = {
            "task": "test_hypothesis",
            "hypothesis": hypothesis_text,
            "findings": []
        }

        # Extract key terms from hypothesis
        lower_hyp = hypothesis_text.lower()

        # Check for correlation hypothesis
        if "correlat" in lower_hyp:
            return self.test_correlation()

        # Check for distribution hypothesis
        if "distribut" in lower_hyp or "normal" in lower_hyp:
            return self.test_distribution()

        # Check for group difference hypothesis
        if "differ" in lower_hyp or "group" in lower_hyp:
            return self.test_group_differences()

        # Default: perform general analysis
        results["findings"] = ["Hypothesis testing: performing general correlation and distribution analysis"]
        corr_results = self.test_correlation()
        dist_results = self.test_distribution()

        results["findings"].extend(corr_results.get("findings", []))
        results["findings"].extend(dist_results.get("findings", []))

        return results

    def execute_task(self, task_description: str, task_type: str = "general") -> Dict[str, Any]:
        """Execute a data analysis task"""
        logger.info(f"Executing task: {task_description}")

        try:
            # Route to appropriate analysis method
            if "structure" in task_description.lower() or "explore" in task_description.lower():
                return self.explore_structure()
            elif "correlation" in task_description.lower():
                return self.test_correlation()
            elif "distribution" in task_description.lower():
                return self.test_distribution()
            elif "difference" in task_description.lower() or "group" in task_description.lower():
                return self.test_group_differences()
            elif "pca" in task_description.lower() or "component" in task_description.lower():
                return self.perform_pca()
            elif "hypothesis" in task_description.lower():
                return self.test_hypothesis(task_description)
            else:
                # Default: explore structure
                return self.explore_structure()

        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "error": str(e),
                "task": task_description,
                "findings": [f"Error during analysis: {str(e)}"]
            }
