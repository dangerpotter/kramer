"""
Computational Analysis Agent for Simulation and Modeling.

Goes beyond basic data analysis to perform:
- Mathematical modeling
- Numerical simulations
- Parameter estimation
- Sensitivity analysis
- Model fitting and validation
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from anthropic import Anthropic

from src.kramer.code_executor import CodeExecutor
from src.utils.cost_tracker import CostTracker

logger = logging.getLogger(__name__)


@dataclass
class ModelingConfig:
    """Configuration for computational analysis."""
    api_key: Optional[str] = None
    model: str = "claude-sonnet-4-20250514"
    timeout: int = 300  # 5 minutes for complex simulations
    max_iterations: int = 3
    use_extended_thinking: bool = True


@dataclass
class SimulationResult:
    """Result from a simulation or modeling task."""
    success: bool
    model_type: str  # e.g., "differential_equations", "monte_carlo", "regression"
    code: str
    results: Dict[str, Any]
    plots: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    interpretation: str = ""
    error: Optional[str] = None


class ComputationalAnalysisAgent:
    """
    Agent for advanced computational analysis and modeling.

    Capabilities:
    - Differential equation modeling
    - Monte Carlo simulations
    - Parameter optimization
    - Statistical modeling
    - Time series forecasting
    - Agent-based modeling
    - Network analysis
    - Stochastic processes
    """

    def __init__(self, config: ModelingConfig):
        """
        Initialize computational analysis agent.

        Args:
            config: ModelingConfig instance
        """
        self.config = config
        self.client = Anthropic(api_key=config.api_key)
        self.executor = CodeExecutor(timeout=config.timeout)

        logger.info("ComputationalAnalysisAgent initialized")

    async def run_differential_equation_model(
        self,
        system_description: str,
        initial_conditions: Dict[str, float],
        time_span: tuple,
        dataset_path: Optional[str] = None
    ) -> SimulationResult:
        """
        Create and solve differential equation model.

        Args:
            system_description: Description of the system to model
            initial_conditions: Initial values for variables
            time_span: (t_start, t_end) for simulation
            dataset_path: Optional data path for validation

        Returns:
            SimulationResult with model and outputs
        """
        prompt = self._create_differential_equation_prompt(
            system_description,
            initial_conditions,
            time_span,
            dataset_path
        )

        return await self._execute_modeling_task(prompt, "differential_equations")

    async def run_monte_carlo_simulation(
        self,
        process_description: str,
        parameters: Dict[str, Any],
        n_simulations: int = 1000,
        dataset_path: Optional[str] = None
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation.

        Args:
            process_description: Description of stochastic process
            parameters: Parameter distributions and values
            n_simulations: Number of simulation runs
            dataset_path: Optional data path for comparison

        Returns:
            SimulationResult with simulation outputs
        """
        prompt = self._create_monte_carlo_prompt(
            process_description,
            parameters,
            n_simulations,
            dataset_path
        )

        return await self._execute_modeling_task(prompt, "monte_carlo")

    async def fit_statistical_model(
        self,
        dataset_path: str,
        model_type: str,
        target_variable: str,
        predictor_variables: List[str],
        hypothesis: Optional[str] = None
    ) -> SimulationResult:
        """
        Fit statistical model to data.

        Args:
            dataset_path: Path to dataset
            model_type: Type of model (e.g., "regression", "glm", "mixed_effects")
            target_variable: Dependent variable
            predictor_variables: Independent variables
            hypothesis: Optional hypothesis to test

        Returns:
            SimulationResult with fitted model and diagnostics
        """
        prompt = self._create_statistical_model_prompt(
            dataset_path,
            model_type,
            target_variable,
            predictor_variables,
            hypothesis
        )

        return await self._execute_modeling_task(prompt, "statistical_model")

    async def run_sensitivity_analysis(
        self,
        model_code: str,
        parameters: Dict[str, tuple],  # parameter: (min, max)
        output_metrics: List[str],
        n_samples: int = 100
    ) -> SimulationResult:
        """
        Perform sensitivity analysis on a model.

        Args:
            model_code: Python code for model function
            parameters: Dict of parameter names to (min, max) ranges
            output_metrics: List of metrics to track
            n_samples: Number of parameter samples

        Returns:
            SimulationResult with sensitivity results
        """
        prompt = self._create_sensitivity_analysis_prompt(
            model_code,
            parameters,
            output_metrics,
            n_samples
        )

        return await self._execute_modeling_task(prompt, "sensitivity_analysis")

    async def forecast_time_series(
        self,
        dataset_path: str,
        time_column: str,
        value_column: str,
        forecast_periods: int,
        method: str = "auto"  # "auto", "arima", "prophet", "lstm"
    ) -> SimulationResult:
        """
        Forecast time series data.

        Args:
            dataset_path: Path to time series dataset
            time_column: Name of time/date column
            value_column: Name of value column to forecast
            forecast_periods: Number of periods to forecast
            method: Forecasting method

        Returns:
            SimulationResult with forecasts and diagnostics
        """
        prompt = self._create_time_series_prompt(
            dataset_path,
            time_column,
            value_column,
            forecast_periods,
            method
        )

        return await self._execute_modeling_task(prompt, "time_series")

    async def run_agent_based_model(
        self,
        system_description: str,
        agent_rules: Dict[str, Any],
        environment_params: Dict[str, Any],
        n_agents: int,
        n_steps: int
    ) -> SimulationResult:
        """
        Run agent-based simulation.

        Args:
            system_description: Description of system to model
            agent_rules: Rules governing agent behavior
            environment_params: Environment parameters
            n_agents: Number of agents
            n_steps: Number of simulation steps

        Returns:
            SimulationResult with simulation outputs
        """
        prompt = self._create_agent_based_model_prompt(
            system_description,
            agent_rules,
            environment_params,
            n_agents,
            n_steps
        )

        return await self._execute_modeling_task(prompt, "agent_based_model")

    async def optimize_parameters(
        self,
        objective_description: str,
        parameters: Dict[str, tuple],  # parameter: (min, max, initial)
        constraints: List[str],
        dataset_path: Optional[str] = None
    ) -> SimulationResult:
        """
        Optimize model parameters.

        Args:
            objective_description: Description of optimization objective
            parameters: Parameters to optimize with bounds
            constraints: List of constraint descriptions
            dataset_path: Optional data for fitting

        Returns:
            SimulationResult with optimized parameters
        """
        prompt = self._create_optimization_prompt(
            objective_description,
            parameters,
            constraints,
            dataset_path
        )

        return await self._execute_modeling_task(prompt, "optimization")

    async def _execute_modeling_task(
        self,
        prompt: str,
        model_type: str
    ) -> SimulationResult:
        """
        Execute a modeling task using Claude and code execution.

        Args:
            prompt: Prompt for Claude
            model_type: Type of modeling task

        Returns:
            SimulationResult
        """
        total_cost = 0.0

        try:
            # Generate code using Claude
            logger.info(f"Generating {model_type} code with Claude...")

            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=4000,
                thinking={
                    "type": "enabled",
                    "budget_tokens": 10000
                } if self.config.use_extended_thinking else None,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )

            # Track cost
            cost = CostTracker.track_call(self.config.model, response)
            total_cost += cost

            # Extract code
            code = self._extract_code_from_response(response)

            if not code:
                return SimulationResult(
                    success=False,
                    model_type=model_type,
                    code="",
                    results={},
                    error="No code generated"
                )

            logger.info(f"Executing {model_type} code...")

            # Execute code
            exec_result = await self.executor.execute(code)

            if not exec_result.success:
                logger.error(f"Code execution failed: {exec_result.error}")
                return SimulationResult(
                    success=False,
                    model_type=model_type,
                    code=code,
                    results={},
                    error=exec_result.error
                )

            # Parse results
            results = self._parse_execution_results(exec_result)

            # Get interpretation
            interpretation = self._extract_interpretation_from_response(response)

            return SimulationResult(
                success=True,
                model_type=model_type,
                code=code,
                results=results,
                plots=exec_result.plots,
                parameters=results.get("parameters", {}),
                metrics=results.get("metrics", {}),
                interpretation=interpretation
            )

        except Exception as e:
            logger.error(f"Modeling task failed: {e}", exc_info=True)
            return SimulationResult(
                success=False,
                model_type=model_type,
                code="",
                results={},
                error=str(e)
            )

    def _create_differential_equation_prompt(
        self,
        system_description: str,
        initial_conditions: Dict[str, float],
        time_span: tuple,
        dataset_path: Optional[str]
    ) -> str:
        """Create prompt for differential equation modeling."""
        return f"""
You are a computational scientist. Create a differential equation model for the following system:

{system_description}

Initial conditions: {initial_conditions}
Time span: {time_span[0]} to {time_span[1]}
{f"Dataset for validation: {dataset_path}" if dataset_path else ""}

Write Python code that:
1. Defines the system of differential equations
2. Solves the equations using scipy.integrate.solve_ivp
3. Visualizes the results
{f"4. Compares model output to data from {dataset_path}" if dataset_path else ""}
5. Prints key findings and model parameters

Use matplotlib for plotting and pandas for data handling if needed.
Store results in a dict called 'results' with keys: 'solution', 'parameters', 'metrics', 'findings'.
"""

    def _create_monte_carlo_prompt(
        self,
        process_description: str,
        parameters: Dict[str, Any],
        n_simulations: int,
        dataset_path: Optional[str]
    ) -> str:
        """Create prompt for Monte Carlo simulation."""
        return f"""
You are a computational scientist. Create a Monte Carlo simulation for:

{process_description}

Parameters: {parameters}
Number of simulations: {n_simulations}
{f"Dataset for comparison: {dataset_path}" if dataset_path else ""}

Write Python code that:
1. Implements the stochastic process
2. Runs {n_simulations} Monte Carlo simulations
3. Computes summary statistics (mean, median, confidence intervals)
4. Visualizes distributions and trajectories
{f"5. Compares simulation results to data from {dataset_path}" if dataset_path else ""}
6. Prints key findings

Use numpy for random number generation and matplotlib for plotting.
Store results in a dict called 'results' with keys: 'simulations', 'statistics', 'metrics', 'findings'.
"""

    def _create_statistical_model_prompt(
        self,
        dataset_path: str,
        model_type: str,
        target_variable: str,
        predictor_variables: List[str],
        hypothesis: Optional[str]
    ) -> str:
        """Create prompt for statistical modeling."""
        return f"""
You are a statistical modeler. Fit a {model_type} model to the data.

Dataset: {dataset_path}
Target variable: {target_variable}
Predictor variables: {', '.join(predictor_variables)}
{f"Hypothesis to test: {hypothesis}" if hypothesis else ""}

Write Python code that:
1. Loads the dataset
2. Fits a {model_type} model
3. Performs model diagnostics
4. Tests model assumptions
5. Computes fit metrics (R², AIC, BIC, etc.)
6. Visualizes results (fitted vs actual, residuals, etc.)
{f"7. Tests the hypothesis: {hypothesis}" if hypothesis else ""}
8. Prints interpretation and conclusions

Use statsmodels or sklearn for modeling, pandas for data, matplotlib for plotting.
Store results in a dict called 'results' with keys: 'model_summary', 'parameters', 'metrics', 'diagnostics', 'findings'.
"""

    def _create_sensitivity_analysis_prompt(
        self,
        model_code: str,
        parameters: Dict[str, tuple],
        output_metrics: List[str],
        n_samples: int
    ) -> str:
        """Create prompt for sensitivity analysis."""
        return f"""
You are performing sensitivity analysis on a computational model.

Model function:
```python
{model_code}
```

Parameters to vary: {parameters}
Output metrics to track: {output_metrics}
Number of samples: {n_samples}

Write Python code that:
1. Implements the model function
2. Samples parameters using Latin Hypercube Sampling or similar
3. Runs model for each parameter set
4. Computes sensitivity indices (Sobol, Morris, or variance-based)
5. Visualizes parameter effects (tornado plot, scatter plots)
6. Identifies most influential parameters
7. Prints sensitivity analysis results

Use SALib or implement custom sensitivity analysis.
Store results in a dict called 'results' with keys: 'sensitivity_indices', 'rankings', 'metrics', 'findings'.
"""

    def _create_time_series_prompt(
        self,
        dataset_path: str,
        time_column: str,
        value_column: str,
        forecast_periods: int,
        method: str
    ) -> str:
        """Create prompt for time series forecasting."""
        return f"""
You are a time series analyst. Forecast future values using {method} method.

Dataset: {dataset_path}
Time column: {time_column}
Value column: {value_column}
Forecast periods: {forecast_periods}
Method: {method}

Write Python code that:
1. Loads and prepares the time series data
2. Performs exploratory analysis (trend, seasonality, stationarity tests)
3. Fits appropriate forecasting model ({method})
4. Generates {forecast_periods} period forecast
5. Computes forecast accuracy metrics
6. Visualizes historical data and forecasts with confidence intervals
7. Prints forecasting results and diagnostics

Use appropriate libraries (statsmodels, prophet, scikit-learn).
Store results in a dict called 'results' with keys: 'forecasts', 'confidence_intervals', 'metrics', 'diagnostics', 'findings'.
"""

    def _create_agent_based_model_prompt(
        self,
        system_description: str,
        agent_rules: Dict[str, Any],
        environment_params: Dict[str, Any],
        n_agents: int,
        n_steps: int
    ) -> str:
        """Create prompt for agent-based modeling."""
        return f"""
You are building an agent-based model for:

{system_description}

Agent rules: {agent_rules}
Environment parameters: {environment_params}
Number of agents: {n_agents}
Simulation steps: {n_steps}

Write Python code that:
1. Defines agent class with specified rules
2. Defines environment with parameters
3. Initializes {n_agents} agents
4. Runs simulation for {n_steps} steps
5. Collects metrics at each step
6. Visualizes agent behavior and system evolution
7. Analyzes emergent patterns
8. Prints simulation findings

Use object-oriented programming for agents and environment.
Store results in a dict called 'results' with keys: 'trajectories', 'metrics', 'patterns', 'findings'.
"""

    def _create_optimization_prompt(
        self,
        objective_description: str,
        parameters: Dict[str, tuple],
        constraints: List[str],
        dataset_path: Optional[str]
    ) -> str:
        """Create prompt for parameter optimization."""
        return f"""
You are optimizing parameters for:

{objective_description}

Parameters to optimize: {parameters}
Constraints: {constraints}
{f"Dataset: {dataset_path}" if dataset_path else ""}

Write Python code that:
1. Defines objective function
2. Defines constraints
3. Sets parameter bounds
4. Runs optimization (scipy.optimize, genetic algorithm, or similar)
5. Validates optimized parameters
6. Visualizes optimization trajectory
7. Compares initial vs optimized results
8. Prints optimization results

Use scipy.optimize or similar optimization library.
Store results in a dict called 'results' with keys: 'optimal_parameters', 'objective_value', 'metrics', 'convergence', 'findings'.
"""

    def _extract_code_from_response(self, response) -> str:
        """Extract Python code from Claude's response."""
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        # Extract code blocks
        import re
        code_blocks = re.findall(r"```python\n(.*?)```", content, re.DOTALL)

        if code_blocks:
            return code_blocks[0].strip()

        # Fallback: try to find code without markers
        return content.strip()

    def _extract_interpretation_from_response(self, response) -> str:
        """Extract interpretation text from Claude's response."""
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        # Remove code blocks
        import re
        interpretation = re.sub(r"```python\n.*?```", "", content, flags=re.DOTALL)

        return interpretation.strip()

    def _parse_execution_results(self, exec_result) -> Dict[str, Any]:
        """Parse results from code execution."""
        results = {
            "stdout": exec_result.stdout,
            "stderr": exec_result.stderr,
        }

        # Try to extract 'results' dict from stdout
        import re
        results_match = re.search(
            r"results\s*=\s*({.*})",
            exec_result.stdout,
            re.DOTALL
        )

        if results_match:
            try:
                import ast
                results_dict = ast.literal_eval(results_match.group(1))
                results.update(results_dict)
            except:
                pass

        return results
